# CLAUDE.md

Working notes for this fork of ai-toolkit. Keep the **Fixes** section at the bottom
appended to — it is the reference log for past investigations.

## Running

```bash
./venv/bin/python run.py config/sets/<arch>/<job>.yml     # train from a config
./venv/bin/python run.py <cfg> -r -n <name> -l <logfile>  # recover / name / log
```

The UI (`ui/`, a Next.js app) writes job rows to `aitk_db.db` and shells out to the
same trainer. Configs it launches are snapshotted to
`output/<job_name>/config.yaml` — note that file is **rewritten whenever the job is
edited in the UI**, so its mtime may be newer than the run you are debugging. The
source of truth for a CLI run is the file under `config/sets/`.

## Layout

| Path | What |
|---|---|
| `jobs/process/BaseSDTrainProcess.py` | Trainer driver: model load → network → optimizer → dataloader → train loop |
| `extensions_built_in/sd_trainer/SDTrainer.py` | The default `diffusion_trainer` process |
| `extensions_built_in/diffusion_models/<arch>/` | Per-architecture model classes (ltx2, wan22, qwen_image, flux2, …) |
| `toolkit/models/base_model.py` | `BaseModel` — base for all **modern** archs |
| `toolkit/stable_diffusion_model.py` | Legacy SD1.5/SDXL base. Parallel implementation of the same helpers — **changes to device-state logic usually need to land in both** |
| `toolkit/dataloader_mixins.py` | Dataset caching pipeline (latents, text embeds, CLIP vision, controls) |
| `toolkit/memory_management/` | Layer offloading (`MemoryManager`) — streams weights CPU→GPU per forward |
| `toolkit/util/ostris_quant.py`, `convrot_quant.py` | `OstrisLinear` + the ConvRot int8/fp4 quantizer backends |
| `toolkit/unloader.py` | Hard-unload helpers (`unload_text_encoder`, `MemoryManager.free`) |

## Things that bite

**Setup order in `BaseSDTrainProcess.run()`** — model load → network created and
`force_to(device, fp32)` → optimizer → **dataloader construction, which is what
triggers all dataset caching** → `hook_before_train_loop` (this is where
`unload_text_encoder` runs). So caching happens with the network already resident.

**Caching order** is fixed in `data_loader.py: setup_epoch()`:
buckets → `cache_latents_all_latents()` → `cache_clip_vision_to_disk()` →
`cache_text_embeddings()` → `setup_controls()`.

**Cache directories live next to the media**, and the names are not symmetrical:
- latents → `<dataset>/_latent_cache/`
- text embeddings → `<dataset>/_t_e_cache/`  ← note the underscores

Deleting `_te_cache` does nothing. If a caching stage finishes suspiciously fast
(thousands of it/s), it was a cache hit, not a run.

**`MemoryManager.attach()` replaces `module.to`** with `memory_managed_to`, which
only moves the *unmanaged* submodules (norms, embeddings, anything in
`ignore_modules`). Managed linear/conv weights stay pinned on CPU and are streamed
through a depth-4 ring buffer per forward. Consequences:
- `model.to(device)` is silently a partial move for offloaded models.
- `model.device` (first-parameter device) is not a reliable "is it loaded" signal.
- Peak VRAM does **not** grow with layer count — verified: a 30-layer and a 60-layer
  stack both peak at 0.18 GB.
- `OstrisLinear.forward` has a rescue path that pulls a layer onto the GPU
  permanently if it is *not* memory-managed. It prints
  `OstrisLinear: quantized weights found on cpu while the input is on cuda` once.
  Seeing that line means offloading did not attach and VRAM is about to blow up.

**Writing a standalone probe script** that constructs a model outside the trainer:
- `ModelConfig.vae_dtype` defaults to `ModelConfig.dtype`, *not* `train.dtype`.
  Omit it and the VAE runs fp16 against bf16 weights → `Input type (c10::Half) and
  bias type (c10::BFloat16) should be the same`. Pass
  `ModelConfig(**{**cfg["model"], "dtype": cfg["train"]["dtype"]})`.
- `resolution: [640, 640]` in YAML is expanded into one `DatasetConfig` per
  resolution by the trainer. Handing the raw list to `DatasetConfig` breaks bucketing
  (`can't multiply sequence by non-int of type 'list'`). Pass a scalar.
- `cache_text_embeddings` is set on the dataset by `BaseSDTrainProcess` from the
  *train* config, so set it on the `DatasetConfig` kwargs yourself.

## Measured VRAM — LTX 2.5 on a 16 GB RTX 4080

Config: 22B ConvRot-int8 DiT + Gemma-4 12B ConvRot-int8 TE, `low_vram: true`,
`layer_offloading: true` at 100% for both, 640×640×57f video + audio, 225 clips.

| Point | VRAM |
|---|---|
| After `load_model()` | 2.0 GB |
| Latent caching (225 clips, video + audio VAE) | 4.4 GB peak |
| VAE residue once latent caching returns | **0.00 GB** |
| Text-embed caching (225 captions) | 5.9 GB peak |
| Full trainer run incl. 2 training steps | exit 0 |

Weight distribution after load (`cuda` / `cpu`):
transformer 0 / 19.5 GB · text encoder **2.0 / 10.9 GB** · video VAE 0 / 1.45 GB ·
audio VAE 0 / 0.11 GB · connectors 0 / 4.33 GB · vocoder 0 / 0.26 GB.

The text encoder's 10.9 GB is the number to remember: if a crash shows ~16.6 GB
resident on a 16 GB card, that is very close to "TE fully resident (10.9) + normal
text-stage working set (5.8)", i.e. **offloading was not active on the encoder** —
not a leak in the caching pipeline.

---

# Fixes

Append new entries at the top. Format: date — symptom — root cause — change — how it
was verified.

## 2026-08-15 — LTX 2.5 CUDA OOM at the start of text-embedding caching

**Symptom.** Training crashed immediately after latent caching finished:

```
Caching latents to disk: 100%|...| 225/225 [05:46<00:00,  1.54s/it]
Caching text_embeddings for .../datasets/ltx25/songHye/gold
Caching text embeddings to disk:   0%| | 0/225 [00:00<?, ?it/s]
[W CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while
 trying to allocate 31457280 bytes (free: 47579136, total: 16684941312).
```

Hypothesis at the time: the video-latent VAE was not being unloaded before the text
encoder loaded.

**What the investigation actually found.** Not reproducible, and the VAE was already
being released. Reproduced the full pipeline against a 225-clip copy of the dataset
with the exact job config — both the caching sequence alone and the real
`run.py` trainer completed, peaking at 5.9 GB of 16.4 GB (exit 0). Measurements in the
table above. `restore_device_state()` at the end of `cache_latents_all_latents`
already puts the VAE back on CPU; measured residue is 0.00 GB.

The crash arithmetic points elsewhere: 10.9 GB (TE weights that *should* be streamed)
+ ~5.8 GB (normal text-stage working set) ≈ 16.7 GB ≈ the reported 16.68 GB total with
47 MB free. So that run had the text encoder fully resident rather than offloaded, or
~11 GB held by another process on the GPU. If it recurs, check `nvidia-smi` for a
leftover job process and grep the log for the `OstrisLinear: quantized weights found
on cpu` warning before assuming a pipeline leak. The two `[W]` lines are allocator
*retry warnings* — the fatal traceback is below them and is what actually identifies
the failure.

**Changes made** (real defects found along the way, none of them the reported crash):

1. `toolkit/models/base_model.py` — `set_device_state_preset()` never handled
   `'cache_text_encoder'` or `'unload'`. Every modern arch derives from `BaseModel`
   and `cache_text_embeddings` calls exactly that preset, so it fell through to
   `active_modules = []` — pushing the **text encoder to CPU at the moment it was
   needed**, and offloading the VAE/unet only by accident. Both presets are now
   explicit (mirroring `stable_diffusion_model.py`, which already had them).
   `control_generator.py` calls `'unload'` and was relying on the same accident.

2. `toolkit/dataloader_mixins.py` — `cache_text_embeddings()` now drops the VAE to CPU
   and flushes **up front**, before the loop. Previously that only happened lazily
   inside the loop on the first item that needed encoding, so a fully-cached dataset
   kept the VAE resident for the whole stage. Moved unconditionally because a combined
   VAE (LTX's `ComboVae`) reports only its video sub-VAE's device, so a device check
   would leave the audio VAE behind.

3. `toolkit/dataloader_mixins.py` — added `@torch.no_grad()` to
   `cache_text_embeddings()`. `encode_images` and `cache_clip_vision_to_disk` both
   disable grad; this path did not, and `encode_prompt` has no `no_grad` of its own.

4. `DeviceStatePreset` literals in both base classes updated to list all five presets.

**Verified.** Full 225-clip run with both caches deleted, post-fix: exit 0, 3.59 GB
peak allocated, ~6.3 GB driver-reported. No behaviour change to the training step.

**Ruled out** (so nobody re-checks these): layer-offloading ring buffers are bounded
and do not grow with depth; ConvRot's autograd training path is gated on
`x.requires_grad` so it does not retain GPU weights during frozen inference;
latent-cache tensors are moved to CPU and `file_item.cleanup()` runs per item;
`flush()` (`empty_cache` + `gc.collect`) does release the VAE stage's reserved pool.
