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

## 2026-10-31 — LTX 2.5 OOM with `diff_output_preservation: true`

**Symptom.** `alinaSvere_zlxny` (768×768×65f + audio, 16 GB card) aborted after 3 OOMs
in a row with DOP on, even with the split backward from the 2026-10-29 entry. Without
DOP the same size fits, but it still hit an occasional OOM skip.

**Root cause.** Measured per phase. All failures were in the preservation backward,
which started ~2 GB higher than the main one, for two reasons:
1. The main pass's LoRA `.grad` buffers (~1.7 GB) stayed on the GPU during the
   preservation pass. Splitting the backward frees activations, not grads.
2. `LTX2Model.pad_embeds()` mutated the caller's `PromptEmbeds`, so `conditional_embeds`
   kept a padded `(1, 1024, 188160)` tensor (0.36 GB) for the rest of the step, and the
   preservation pass padded a second one.
Also, usable VRAM tops out around 14.5 GB reserved, not 15.6, and the default allocator
fragments badly enough to fail near that ceiling.

**Change.**
- `SDTrainer.py`: `park_grads()` / `restore_grads()` hold the main grads on CPU during
  the preservation pass, then add them back before the optimizer step.
- `ltx2.py`: `pad_embeds()` pads a shallow copy; `get_noise_prediction()` drops the
  padded embeds right after the connectors. Lowers the peak for all LTX training.
- `.env` (gitignored, per machine): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Verified.** 30 steps of the real config on GPU (all 29 clips in one bucket):

| Run | Result | OOM skips | Preservation backward peak |
|---|---|---|---|
| Split backward only | aborted | 3 | ~14.0 GB |
| + grad parking | aborted | 3 | 13.4 GB |
| + grad parking + expandable | finished | 4 | 13.4 GB |
| + pad fix, default allocator | finished | 1 (230 allocator warnings) | 12.8 GB |
| + pad fix + expandable | finished | **0** (0 warnings) | 12.8 GB |

Max allocated 13.2 GB, ~25 s/step vs ~11 s/step without DOP.
`testing/test_park_grads.py` asserts the split equals one backward on the sum. Not run:
a full 3500 steps, sampling through the changed `pad_embeds()`, and DOP on MiniMax-H3.

**Ruled out:** dropping `loss` / `noise_pred` after the main backward. Tried it and the
preservation pass still started 0.49 GB high, so the spent graph is not what holds memory.

## 2026-10-29 — MiniMax-H3 OOM with `diff_output_preservation: true`

*Correction (2026-10-31): "peak is max(main, preservation)" below is wrong without grad
parking, because the main grads stay resident. See the entry above.*

**Symptom.** `songHye_zxqv` (16 GB card, `low_vram`, 100% layer offloading, 640×640×73f
+ audio) trains at ~14.3 GB with DOP off and OOMs with it on.

**Root cause.** `SDTrainer.train_single_accumulation()` summed the losses
(`loss = loss + preservation_loss`) and ran one backward, so the main forward's
autograd graph and the preservation forward's graph were both resident at peak. With
gradient checkpointing those graphs are the per-block activations of a full video
forward — roughly double the non-DOP activation peak. The prior pass is `no_grad` with
the LoRA off and is not the problem.

**Change.** The main loss is now backwarded (and its graph freed) *before* the
preservation forward runs; only `preservation_loss` goes through the shared backward at
the end. Gradients accumulate in `.grad`, so the update is identical to one backward on
the sum, but peak is max(main, preservation) instead of the sum. Return value is still
main + preservation for logging. Applies to `blank_prompt_preservation` too.

**Verified.** `py_compile` only. Not yet run on GPU — the GPU was busy with the non-DOP
run. If a config ever feeds a grad-carrying tensor through `pred_kwargs` into both
passes, the second backward will raise "backward through the graph a second time"
(loud, not silent).

## 2026-08-16 — LTX 2.5 CUDA OOM immediately after the first sample

**Symptom.** `songHye_sxhonzkyx_woman` ran cleanly to step 250 (~15773/16376 MiB,
`low_vram: true`, `layer_offloading: true` at 100% for transformer and text
encoder), saved its checkpoint, generated its first sample (`skip_first_sample:
true` had suppressed step 0's), and then hit repeated allocator OOM warnings for a
suspiciously round 314572800-byte (300 MB) allocation right after "Generating
Samples: 100%" — i.e. after the sample had already finished rendering, not during
it.

**Root cause.** `BaseModel.generate_images()` (`toolkit/models/base_model.py`)
saves and restores device placement only for `vae`, `unet`/`transformer`,
`text_encoder`, `adapter`, and `refiner_unet` — see `save_device_state()` /
`set_device_state()` / `restore_device_state()`. LTX 2.5's generation pipeline
(`extensions_built_in/diffusion_models/ltx2/ltx2.py`,
`LTX2Model.generate_single_image()`) has two components outside that list:
`connectors` and `vocoder`. Both get pulled fully onto the GPU by
`pipeline = pipeline.to(self.device_torch)` for every sample.

- `connectors` (4.33 GB) turned out to be a non-issue: the training forward pass
  (`if self.pipeline.connectors.device != self.transformer.device: ...to(...)`,
  around line 1048) already keeps it GPU-resident on every step, so sampling
  doesn't change its steady-state footprint.
- `vocoder` (0.26 GB) is **only** touched during sample generation (waveform
  synthesis) — nothing in the training step ever references it, so nothing ever
  moves it back to CPU after `pipeline.to()` puts it on the GPU. Before the first
  sample it had never been loaded onto the GPU at all, so this is a fresh ~0.26 GB
  of permanent VRAM growth triggered by exactly the event in the symptom (first
  sample after `skip_first_sample`). With training already within ~600 MB of the
  16 GB ceiling, that's enough on its own to explain a subsequent 300 MB
  allocation failing on the very next step.

**Change.** `LTX2Model.generate_single_image()` — right after the pipeline call
(same spot that already resets VAE tiling for `low_vram`), explicitly move
`self.pipeline.vocoder` back to CPU and flush. Mirrors the existing pattern in
`encode_audio()`'s `finally` block, which does the same thing for the audio VAE
for the same underlying reason (component not covered by the base class's device
state tracking).

**Verified.** Static: read-traced `pipeline.to()` → `DiffusionPipeline.to()` →
per-component `.to(device)`, confirmed `vocoder` isn't wrapped by
`MemoryManager.attach()` (so the move is a full, not partial, residency) and isn't
referenced anywhere in the training-step forward pass. Not yet confirmed with a
live run to step 250+ — checkpoint and optimizer state for step 250 are already on
disk, so resuming with `-r` is the next real test.

**Ruled out:** the `OstrisLinear` "rescue path" that permanently strands a
non-memory-managed quantized layer on GPU (`toolkit/util/ostris_quant.py`) — both
`layer_offloading_transformer_percent` and `..._text_encoder_percent` are `1`, so
every `OstrisLinear` layer should be under `MemoryManager`, and the sample
finished without printing the associated warning. `self.unet.to(self.device_torch,
dtype=self.torch_dtype)` at the end of `generate_images()` — intercepted by
`memory_managed_to` for an attached module, so it's a partial (safe) move, not a
full load.

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
