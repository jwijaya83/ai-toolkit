"""Fizgig's MiniMax H3 character-LoRA recipe (github.com/shootthesound/Fizgig,
"MiniMax H3 (rank 16, 60 epochs)" preset, v9.7.9), behind one switch:

    model:
      arch: "minimax_h3"
      model_kwargs:
        fizgig: true      # false / absent = the job trains exactly as its YAML says

On, FIZGIG_TRAIN and the block window replace the YAML's values (each change
prints at startup); data, run length and saves stay the YAML's. The model-side
pieces are plain model_kwargs knobs the recipe only defaults, so a YAML value
wins: train_blocks (read here), train_lownoise_pct, tread, clip_stills (read by
minimax_h3.py and the dataloader).
"""

import copy
import heapq
import os

FIZGIG_TRAIN = {
    "optimizer": "automagic3",
    "lr": 1e-6,  # Automagic's start; it sets its own rate from here
    # non-fused: a fused Automagic updates inside backward, before the trainer's
    # max_grad_norm clip runs, so clipping would silently do nothing
    "optimizer_params": {"weight_decay": 0.0, "fused": False},
    "max_grad_norm": 1.0,
    "timestep_type": "shift",  # train_lownoise_pct only acts on the shift schedule
    "content_or_style": "balanced",
    # decay ramps in as min(0.98, (1+n)/(10+n)), as Fizgig's EMAWeights
    "ema_config": {"use_ema": True, "ema_decay": 0.98, "use_num_updates": True},
}
FIZGIG_MODEL_KWARGS = {
    "train_blocks": "20-49",  # identity blocks; "6-49" is Fizgig's More Blocks mode
    "train_lownoise_pct": 60,  # % of steps below sigma 0.5 (stock shift 12 is ~7)
    "tread": "0.5@2-47",  # clip steps: half the video tokens skip blocks 2-46
    "clip_stills": True,  # each clip's sharpest face frame also trains as a photo
}
FIZGIG_ADAPTER = (
    "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v3.safetensors"
)
H3_NUM_BLOCKS = 50


def parse_blocks(spec) -> list:
    """'20-49' or '3-12, 22, 31-33' -> sorted block ids."""
    ids = set()
    try:
        for part in str(spec).split(","):
            lo, _, hi = part.strip().partition("-")
            ids.update(range(int(lo), int(hi or lo) + 1))
    except ValueError:
        raise ValueError(f"model_kwargs.train_blocks: can't read {spec!r}; use e.g. '20-49'")
    if not ids or min(ids) < 0 or max(ids) >= H3_NUM_BLOCKS:
        raise ValueError(
            f"model_kwargs.train_blocks: {spec!r} is outside blocks 0-{H3_NUM_BLOCKS - 1}"
        )
    return sorted(ids)


def apply_fizgig_recipe(process_config: dict) -> dict:
    """The process config with Fizgig's recipe applied when
    model.model_kwargs.fizgig is set; otherwise the config itself, untouched."""
    model = process_config.get("model") or {}
    if not (model.get("model_kwargs") or {}).get("fizgig"):
        return process_config
    if model.get("arch") != "minimax_h3":
        raise ValueError(
            "model_kwargs.fizgig is the MiniMax H3 fl2va recipe (arch: minimax_h3); "
            f"this config's arch is {model.get('arch')!r}"
        )
    if not process_config.get("network"):
        raise ValueError("model_kwargs.fizgig trains a LoRA: add a network section")

    cfg = copy.deepcopy(process_config)
    model = cfg["model"]
    kw = model["model_kwargs"]
    for k, v in FIZGIG_MODEL_KWARGS.items():
        kw.setdefault(k, v)
    # the adapter is half the method; an explicit `assistant_lora_path: null` keeps it off
    model.setdefault("assistant_lora_path", FIZGIG_ADAPTER)

    lines = []
    train = cfg.setdefault("train", {})
    for k, v in FIZGIG_TRAIN.items():
        if train.get(k) != v:
            lines.append(f"train.{k}: {train.get(k)!r} -> {v!r}")
            train[k] = copy.deepcopy(v)

    blocks = parse_blocks(kw["train_blocks"])
    network = cfg["network"]
    nk = network["network_kwargs"] = dict(network.get("network_kwargs") or {})
    # dotted-path match: also leaves out both token-refiner blocks, which Fizgig never trains
    nk["only_if_contains"] = [f"blocks.{i}." for i in blocks]
    nk["ignore_if_contains"] = list(
        dict.fromkeys([*(nk.get("ignore_if_contains") or []), "adaln_proj"])
    )
    lines.append(
        f"network: blocks {kw['train_blocks']} only ({len(blocks) * 4} LoRA Linears), "
        "no token refiner, no adaln_proj"
    )
    lines.append(
        f"model_kwargs: train_lownoise_pct={kw['train_lownoise_pct']} "
        f"tread={kw['tread']} clip_stills={kw['clip_stills']}"
    )
    lines.append(f"assistant LoRA: {model['assistant_lora_path']}")
    print("Fizgig recipe (model_kwargs.fizgig: true):\n  " + "\n  ".join(lines))
    return cfg


# ----------------------------------------------------------------------------
# Clip stills: each clip's sharpest face frame trains as a photo with the
# clip's caption (Fizgig's clip_still_as_photo / minimax/still_pick.py).
# ----------------------------------------------------------------------------


def _sharpness(gray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _largest_face(cascade, gray, min_h: int):
    """(x, y, w, h) of the largest face in frame coordinates, or None. A face
    that fills the frame is beyond the cascade's largest window, so a miss
    retries once on a border-padded copy (Fizgig's pad retry)."""
    import cv2

    H, W = gray.shape
    for pad in (0.0, 0.25):
        py, px = int(H * pad), int(W * pad)
        g = cv2.copyMakeBorder(gray, py, py, px, px, cv2.BORDER_REPLICATE) if pad else gray
        faces = cascade.detectMultiScale(
            g, scaleFactor=1.1, minNeighbors=5, minSize=(min_h, min_h)
        )
        if len(faces):
            x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
            x0, y0 = max(0, x - px), max(0, y - py)
            x1, y1 = min(W, x - px + w), min(H, y - py + h)
            return x0, y0, x1 - x0, y1 - y0
    return None


def pick_still_frame(path: str, max_faces: int = 6, top_k: int = 24, min_face_frac: float = 0.08):
    """(index, BGR frame with long edge <= 1024, found_face) for a clip's
    sharpest frame that shows a face: the frames are visited sharpest first and
    the pick is the one whose face crop is sharpest (Laplacian variance) among
    the first `max_faces` faces found. No face in the `top_k` sharpest frames ->
    the sharpest frame.

    ponytail: OpenCV's frontal Haar cascade (the only detector installed) misses
    profiles that Fizgig's insightface finds; those clips get the sharpest frame
    instead. Swap in a stronger detector here if that shows up in the stills."""
    import cv2

    cap = cv2.VideoCapture(path)
    heap, i = [], 0  # the top_k sharpest frames as (sharpness, index, frame)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scale = 1024 / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        item = (_sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), i, frame)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)
        i += 1
    cap.release()
    if not heap:
        raise ValueError(f"no readable frames in {path}")

    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    scored = []
    for _, idx, frame in sorted(heap, reverse=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_h = int(min_face_frac * gray.shape[0])
        face = _largest_face(cascade, gray, min_h)
        if face is None or face[3] < min_h:
            continue
        x, y, w, h = face
        mx, my = int(0.15 * w), int(0.15 * h)
        crop = gray[max(0, y - my) : y + h + my, max(0, x - mx) : x + w + mx]
        if crop.size >= 64:
            scored.append((_sharpness(crop), idx, frame))
        if len(scored) >= max_faces:
            break
    _, idx, frame = max(scored) if scored else max(heap)
    return idx, frame, bool(scored)


def add_clip_stills(videos, root: str, caption_ext: str) -> list:
    """Paths of the clip stills for `videos`, extracting any that are missing
    or older than their clip into <root>/.clip_stills/ (mirroring the folder
    layout). The clip's caption is mirrored beside each still on every scan, so
    caption edits reach it. The dot folder keeps stills out of the dataset's own
    scan: with clip_stills off they are never trained."""
    import cv2

    stills, picked, faces = [], 0, 0
    for video in videos:
        still = os.path.join(
            root, ".clip_stills", os.path.splitext(os.path.relpath(video, root))[0] + ".png"
        )
        try:
            if not os.path.exists(still) or os.path.getmtime(still) < os.path.getmtime(video):
                os.makedirs(os.path.dirname(still), exist_ok=True)
                _, frame, has_face = pick_still_frame(video)
                if not cv2.imwrite(still, frame):
                    raise OSError(f"could not write {still}")
                picked += 1
                faces += has_face
        except Exception as e:  # one bad clip loses its still, not the run
            print(f"Clip still skipped for {video}: {e}")
            continue
        src = os.path.splitext(video)[0] + caption_ext
        dst = os.path.splitext(still)[0] + caption_ext
        text = open(src, encoding="utf-8").read() if os.path.exists(src) else None
        old = open(dst, encoding="utf-8").read() if os.path.exists(dst) else None
        if text is None and old is not None:
            os.remove(dst)
        elif text is not None and text != old:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(text)
        stills.append(still)
    if picked:
        print(f"  -  Clip stills: picked {picked} ({faces} with a face, the rest sharpest frame)")
    return stills
