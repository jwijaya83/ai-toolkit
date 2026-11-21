# Fizgig's MiniMax H3 character-LoRA method in ai-toolkit

*2026-11-19. ai-toolkit: branch `minimax-h3` @ `995ff36` (same commit as `main`). Fizgig: `../Fizgig` @ `3b8dfac0` (v9.7.9, 2026-11-18).*

The Reddit thread itself could not be read: reddit.com refuses automated fetches. A web search matched its
title to the Fizgig 5.2 release, and Fizgig keeps those release notes in its repo
([`docs/RELEASE_NOTES_v5.2.0.md`](../Fizgig/docs/RELEASE_NOTES_v5.2.0.md)), so section 1 is taken from
there. Everything about how the two trainers behave today comes from reading both codebases.

## Summary

- **The thread.** Fizgig 5.2 combined two methods. The first is Ostris's MiniMax H3 **training adapter**, a frozen
  de-distillation LoRA that is on during training. The second is Fizgig's **likeness blocks**: train only blocks 20–49 of the 50.
  In Fizgig's A/B, each method alone ended in about the same place. Together they got there "a quarter sooner" and
  finished higher.
- **ai-toolkit already has half of it.** The adapter is Ostris's own, on by default for `minimax_h3`.
- **The other half needs config only, no code.** `network_kwargs.only_if_contains` set to `blocks.20.` … `blocks.49.` gives
  exactly Fizgig's 120 trainable Linears. I checked this against ai-toolkit's real `LoRASpecialNetwork`. Nothing before block 20
  then has a trainable weight, so autograd stops at block 20 on its own. That is Fizgig's "backward cut", for free.
  This is the part the thread says would be "easy for Ostris".
- **Fizgig's current preset differs in more ways than the thread covers.** Four of them change what the LoRA learns:
  1. 60% of training steps fall below σ 0.5. ai-toolkit puts about 6.5% there.
  2. Automagic v3 starting at 1e-6. ai-toolkit's default is AdamW8bit at 1e-4.
  3. EMA 0.98. ai-toolkit's default is off.
  4. The text token refiner is never trained. ai-toolkit's default trains it.

  The smaller differences are rank 8 or 16, 0.25 MP, and a checkpoint every epoch.
- **Code needed: one knob of about 10 lines**, a training-only timestep shift in `MinimaxH3Model.load_model` (§4.1).
  Everything else is a YAML recipe (§4.2). A UI dropdown, TREAD for clip steps, and sharp stills taken from clips are optional.
- **Not proven here yet.** "Much better" is Fizgig's own face-recognition score on its own datasets. Your Fizgig attempts
  (`../Fizgig/user_data/logs/2026-11-19_*.log`) never reached a training step: they ran out of memory or were stopped on the 16 GB card.
  §5 is the A/B to run.

## 1. The thread: Fizgig 5.2, "combining two MiniMax training methods"

From [`RELEASE_NOTES_v5.2.0.md`](../Fizgig/docs/RELEASE_NOTES_v5.2.0.md):

> Two ways exist to train a LoRA on H3 well, and until now they lived in different trainers. Fizgig's is **Optimised
> Likeness Learning**: H3's identity lives in the back 30 of its 50 blocks, so photo steps train blocks 20–49 only and
> leave the front of the model — composition, anatomy, prompt following — untouched. AI-Toolkit's, by @ostris, is the
> training adapter: H3 is guidance-distilled, so every plain-flow gradient is partly "learn the concept" and partly
> "undo the distillation"; a frozen assistant LoRA under the trainable one pulls the base back toward plain flow, so the
> gradient is all concept from the first step, and it's switched off for sampling.

The test used one dataset and seed and ran 45 epochs per arm. Each epoch's preview was scored against the training photos by face recognition:

| Arm | Fizgig's result (quoted) |
|---|---|
| Blocks 20–49, no adapter | "a likeness-blocks run and an adapter run ended within a point of each other, taking about the same number of epochs to get there" |
| Adapter, all 50 blocks | Same as above, and "the weakest of the three late in the run" |
| Adapter + blocks 20–49 | "got there a quarter sooner, ran clearly ahead through the whole middle of the run, and finished higher than either" |

Their explanation for the adapter-on-all-blocks arm: "its cleaner gradient reaches every block it's allowed to, and for
a likeness LoRA the front blocks have nothing to learn and a lot to damage."

The note to Ostris, which is the "easy for him to implement" part:

> @ostris — the block finding may be useful on your side too: the identity set (20–49) came out of Repair Studio
> ablations on real H3 LoRAs, and it holds for voice as well (the audio zone is 34–49).

Fizgig kept the combination and added to it in later releases:
- 5.4: clip steps are always confined to 20–49 as well.
- 6.1: Automagic v3 became the optimizer.
- 6.2: the text token refiner is no longer trained, and voice also trains 20–49.

Section 2 compares against today's preset, not the 5.2 one.

## 2. Side by side

**ai-toolkit column:** the `minimax_h3` UI defaults ([ui.tsx:1024](extensions_built_in/diffusion_models/ui.tsx#L1024)) plus the
global job defaults ([jobConfig.ts:70](ui/src/app/jobs/new/jobConfig.ts#L70)). Where your
`config/sets/minimaxh3/template.yml` differs, the table notes it. That file is only on branch `ltx2.5-diff-output-preservation`.

**Fizgig column:** the "✨ MiniMax H3 (rank 16, 60 epochs)" preset, which is what your Fizgig runs used
([families/minimax.py:24](../Fizgig/src/fizgig/families/minimax.py#L24)). The "Fast" preset is identical except for rank 8 and 50 epochs.

| | ai-toolkit today | Fizgig preset | Gap |
|---|---|---|---|
| Distillation handling | Ostris training adapter v3: frozen, on for training, off for samples (`assistant_lora_path`) | Same file, same handling (208 Linears at strength 1) | none |
| Trained blocks | all 50 | **20–49** ("Default" training mode). Blocks 0–5 are never trained in any mode | config |
| Trained modules | attn qkv/out + mlp fc1/fc2 in every block **and both token-refiner blocks**: 208 modules, ~77.5M params at rank 16. `adaln_proj` excluded | The same four Linears in blocks 20–49 only: 120 modules, 44.7M params. Refiner and AdaLN excluded | config |
| Rank / alpha | 16 / 16 | 16 / 16 (Fast: 8 / 8) | none |
| Optimizer | AdamW8bit | **Automagic v3** (the same file in both repos), non-fused | config |
| Learning rate | 1e-4, constant | **1e-6 start**. Automagic sets its own rate from there | config |
| Weight decay | 1e-4 | 0 for Automagic (Fizgig adds 1e-4 only to Adam-type optimizers) | config |
| Grad clipping | `max_grad_norm` 1.0, but **silently skipped** when Automagic runs fused (its default) | 1.0, applied before a normal `.step()` | config |
| Timestep density | `timestep_type: shift` at the model's shift 12: about 6.5% of steps below σ 0.5, and nothing below σ ≈ 0.135 | **"Likeness and Style — 60% clean-end"**: shift (1−0.6)/0.6 = 0.667, so 60% of steps fall below σ 0.5 | **code** (§4.1) |
| EMA | off (0.99 when turned on) | **0.98**. Checkpoints and previews use the averaged weights | config |
| Resolution | 512/768/1024 buckets (template: 640) | 0.25 MP (512-class buckets), for stills and clips | config |
| Caption dropout | 0.05 via a cached blank embedding (template: 0.00) | 0.05 via a cached empty-prompt embedding | config |
| Run length / saves | 3000 steps, save every 250 | 60 epochs (Fast: 50), save every epoch, pick by likeness | config |
| LoRA weight dtype | fp32 ([BaseSDTrainProcess.py:1968](jobs/process/BaseSDTrainProcess.py#L1968)) | bf16 with Automagic's stochastic rounding | none (ai-toolkit is finer) |
| Audio | MSE on audio rows, σ remapped to audio shift 3, weight 1 | Same | none |
| Clip steps | all tokens | TREAD: a random half of video tokens skips blocks 2–46 | optional code |
| Clip stills | none | Each clip's sharpest face frame also trains as a photo, with the clip's caption | optional, data prep |
| Base DiT | pruned int8 ConvRot fl2va | Same file. Your Fizgig runs used a custom `minimax_h3_hybrid_fl2va_ref2va_b30-49-int8.safetensors` | none |

## 3. The differences that change the result

### 3.1 Train blocks 20–49 only, and leave the refiner alone (the thread's subject)

Fizgig's findings ([MINIMAX_H3.md](../Fizgig/docs/MINIMAX_H3.md), [v6.2.0](../Fizgig/docs/RELEASE_NOTES_v6.2.0.md)):
- Identity lives in blocks 20–49.
- Blocks 0–5 "deform anatomy and pull the dataset's colour into the render".
- The front of the model holds composition, anatomy, prompt following and motion.
- "The new Default recipe reached **73 by epoch 8**. Earlier H3 runs on this trainer took until epoch 15 and epoch 21 to
  reach 72 and 74." The score then held between 71 and 73 for five more epochs.
- A style LoRA trained on stills alone left the model's motion untouched.

**The refiner.** ai-toolkit's default LoRA also trains `token_refiner.blocks.0/1`, which is 8 of its 208 modules. Fizgig's hint says
training the refiner "softens output and makes previews judder between epochs". v6.2 calls the refiner "the leak" that let audio
gradients corrupt the visual blocks.

**In ai-toolkit,** both name filters match on the dotted module path:
`ignore_if_contains` at [lora_special.py:524](toolkit/lora_special.py#L524), `only_if_contains` at
[:571](toolkit/lora_special.py#L571). I verified the filter by building a tiny real `MiniMaxH3Transformer` and wrapping it with the real
`LoRASpecialNetwork`:

| `network_kwargs` | LoRA modules | Blocks | Refiner |
|---|---|---|---|
| UI default (`ignore_if_contains: [adaln_proj]`) | 208 | 0–49 | yes (8) |
| UI default + `only_if_contains: [blocks.20. … blocks.49.]` | 120 | 20–49 | no |

**The backward cut** follows from four facts:
- The base transformer is frozen ([BaseSDTrainProcess.py:1884](jobs/process/BaseSDTrainProcess.py#L1884)).
- The noisy latents don't require grad ([:1456](jobs/process/BaseSDTrainProcess.py#L1456)).
- The adapter's weights are frozen.
- Checkpointing uses `use_reentrant=False` ([transformer.py:635](extensions_built_in/diffusion_models/minimax_h3/src/transformer.py#L635)).

So the first tensor that needs a gradient is created inside block 20, and blocks 0–19 run forward-only. That should
shorten the backward pass, and under layer offloading it also skips re-streaming blocks 0–19 from CPU during
backward. This has not been measured.

### 3.2 Where the training steps land on the noise schedule

Low σ means a nearly clean image, where skin, eyes and fine detail are made. High σ is where layout, pose and motion are
decided. Fizgig's hint for this setting: "Higher: more detail and likeness. Lower: more movement and composition."

I measured both densities through ai-toolkit's real code path, using 200k draws: `set_train_timesteps('shift')`, the `balanced` index draw
([BaseSDTrainProcess.py:1291](jobs/process/BaseSDTrainProcess.py#L1291)), and σ = t/1000
([minimax_h3.py:866](extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py#L866)).

| | Shift | Steps below σ 0.5 | Median σ | Steps above σ 0.9 | Lowest σ |
|---|---|---|---|---|---|
| ai-toolkit today | 12 | 6.5% | 0.925 | 57.9% | 0.135 |
| Fizgig "Likeness and Style" (60%) | 0.667 | 59.9% | 0.401 | 6.9% | 0.001 |

Fizgig's other setting, "Model default, movement — 8% clean-end", is the same as ai-toolkit's schedule to within rounding
(shift 11.5 versus 12).

- **Side finding: ai-toolkit never trains H3 below σ ≈ 0.135.** The `shift` branch builds its range from `sigma_min`,
  which the scheduler already shifted when it was created, and then shifts it a second time
  ([custom_flowmatch_sampler.py:145](toolkit/samplers/custom_flowmatch_sampler.py#L145)). Fizgig draws σ over the whole
  (0, 1) range. The knob in §4.1 also lowers the floor to 0.001, as a side effect. The double shift affects every `shift`
  model in ai-toolkit, so a general fix is out of scope here.
- **Evidence.** Fizgig publishes less for this setting than for the others. Its docs call 60 "the tuned default for
  stills". The only head-to-head is for voices (v4.0): it "converges much faster and sounds better than Model default".
- **A warning from v3.7.** A density that starves the high-noise end (their old "mid-concentrated" setting left about 0.7% of
  steps there) made LoRAs go soft or distort at full strength without Turbo. 60% still keeps about 40% of steps
  above σ 0.5, so it isn't that failure, but don't push the share to an extreme.
- **The density goes with the optimizer.** Fizgig's sampler notes say that a density which "trains the mid/low-noise zone
  hard" (sigmoid, median σ 0.5), run "at 1e-4 ... visibly overdrove the adapters"
  ([common.py:941](../Fizgig/src/fizgig/minimax/common.py#L941)). The 60% density sits even lower (median σ 0.4), so run it
  with Automagic, not with a flat 1e-4 AdamW.

### 3.3 Automagic v3 from 1e-6

The two repos ship the same code: Fizgig vendored [toolkit/optimizers/automagic3.py](toolkit/optimizers/automagic3.py), and a diff
shows only its license header. Automagic keeps one learning rate for the whole LoRA. It raises the rate while update signs agree and
lowers it while they alternate, so the configured LR is only the starting point. Fizgig v6.1: "In A/B against the flat
rate it replaces it reached likeness sooner and finished ahead".

The code is the same, but the two trainers construct it differently, and both differences matter:

- **`fused`.** ai-toolkit's default `fused=True` ([automagic3.py:189](toolkit/optimizers/automagic3.py#L189)) updates each
  weight inside the backward pass. The trainer's `clip_grad_norm_` runs after backward
  ([SDTrainer.py:2390](extensions_built_in/sd_trainer/SDTrainer.py#L2390)), when the grads are already consumed, so
  clipping silently does nothing. Fused mode also breaks with gradient accumulation and with any step that runs two
  backward passes. Fizgig builds it non-fused ([optimizers.py:409](../Fizgig/src/fizgig/training/optimizers.py#L409)).
  Set `fused: false`.
- **Weight decay.** ai-toolkit passes `optimizer_params.weight_decay` straight in (the UI default is 1e-4). Fizgig gives Automagic
  none ([train.py:1988](../Fizgig/src/fizgig/families/train.py#L1988)). Set `weight_decay: 0.0`.

Automagic keeps its adapted rate in per-parameter state (`state[p]["lr"]`), so ai-toolkit's LR scheduler can't
overwrite it.

### 3.4 EMA 0.98

When EMA is on, ai-toolkit samples and saves from the averaged weights, the same as Fizgig
([BaseSDTrainProcess.py:794](jobs/process/BaseSDTrainProcess.py#L794),
[SDTrainer.py:2400](extensions_built_in/sd_trainer/SDTrainer.py#L2400)). Fizgig's result: "On a four-way A/B at 50
epochs, 0.98 sat five likeness points above EMA-off on the late epochs with half the epoch-to-epoch spread, and reached
50% just as fast."

There is one small difference. Fizgig ramps the decay in as `min(0.98, (1+n)/(10+n))`, which only matters for roughly the first 440
steps. ai-toolkit's `ExponentialMovingAverage` supports the same ramp (`use_num_updates`), but the trainer doesn't pass
it. You can ignore this unless the earliest checkpoints look stuck near the starting weights.

### 3.5 Smaller settings

- **Rank.** 16/16 with 60 epochs, or 8/8 with 50 epochs for the Fast preset. Fizgig says the lower rank "tends to come out more flexible".
- **0.25 MP** (`resolution: [512]`). Fizgig: "four times cheaper per step than 1 MP, and the extra resolution has not
  paid for itself in testing."
- **Caption dropout 0.05.** ai-toolkit already swaps in a cached blank embedding the same way
  ([dataloader_mixins.py:2457](toolkit/dataloader_mixins.py#L2457)).
- **Save every epoch.** In the v6.2 run the likeness window opened at epoch 8. ai-toolkit's own UI text says the adapter
  "will still break down over a long run". Save often and pick the best checkpoint by eye.

### 3.6 Clip-only extras (skip these for photo-only sets)

- **TREAD** (on by default for clip steps): speed only. "The trained LoRA is an ordinary LoRA." It is about 25 lines in Fizgig's
  block loop ([model.py:1022](../Fizgig/src/fizgig/minimax/model.py#L1022)).
- **Sharpest face frame as a still** (on in the character presets): gives each clip a second, sharp training sample at
  the cost of a still.

## 4. What to build

### 4.1 Code: a training-only timestep shift (the only required change)

In [minimax_h3.py:597](extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py#L597), inside `load_model`.
`CustomFlowMatchEulerDiscreteScheduler` and `scheduler_config` are already in that file:

```python
        self.noise_scheduler = self.get_train_scheduler()
        # Fizgig's clean-end share: train_lownoise_pct % of training steps land below
        # sigma 0.5 (shift = (1 - p) / p, so 60 -> 0.667). Training density only:
        # sampling (build_sample_sigmas) and the audio remap keep video_sigma_shift.
        pct = self.model_config.model_kwargs.get("train_lownoise_pct")
        if pct is not None:
            p = float(pct) / 100.0
            if not 0.0 < p < 1.0:
                raise ValueError(f"train_lownoise_pct must be between 0 and 100, got {pct}")
            if getattr(self.noise_scheduler, "fixed_train_sigmas", None) is not None:
                raise ValueError("train_lownoise_pct does not apply to a model trained on fixed sigmas")
            self.noise_scheduler = CustomFlowMatchEulerDiscreteScheduler(
                **{**scheduler_config, "shift": (1.0 - p) / p}
            )
```

Why this is safe:
- `get_noise_prediction` reads σ as `timestep / 1000`, which doesn't depend on the scheduler's shift. The audio σ remap
  uses the class constants (12 to 3), and Fizgig does the same.
- Sampling uses `build_sample_sigmas` with `video_sigma_shift`, not this scheduler.
- The scheduler the trainer builds ([BaseSDTrainProcess.py:1752](jobs/process/BaseSDTrainProcess.py#L1752)) is replaced
  in `load_model` anyway.
- The knob only takes effect with `timestep_type: shift`.
- Fast H3, ref2va and Omni inherit `load_model`, so they get the knob too. The guard stops H3 Image, which trains on
  `fixed_train_sigmas`, from silently losing its schedule.

Add one check with it, `testing/test_minimax_train_shift.py`. It builds the scheduler for p = 0.6, runs the real
`set_train_timesteps('shift')` and balanced draw, and asserts that P(σ < 0.5) ≈ 0.60. The probe I ran for §3.2 does exactly this
and got 0.599.

### 4.2 Config: the Fizgig recipe

Create `config/sets/minimaxh3/<name>_fizgig.yml` from your template
(`git show ltx2.5-diff-output-preservation:config/sets/minimaxh3/template.yml`) and change only these keys. The
step numbers assume 50 photos at batch 1, so one epoch is 50 steps.

```yaml
      network:
        linear: 16                      # Fizgig "rank 16, 60 epochs"; 8 / 8 for its "Fast" preset
        linear_alpha: 16
        network_kwargs:
          ignore_if_contains: ["adaln_proj"]
          only_if_contains: [
            "blocks.20.", "blocks.21.", "blocks.22.", "blocks.23.", "blocks.24.",
            "blocks.25.", "blocks.26.", "blocks.27.", "blocks.28.", "blocks.29.",
            "blocks.30.", "blocks.31.", "blocks.32.", "blocks.33.", "blocks.34.",
            "blocks.35.", "blocks.36.", "blocks.37.", "blocks.38.", "blocks.39.",
            "blocks.40.", "blocks.41.", "blocks.42.", "blocks.43.", "blocks.44.",
            "blocks.45.", "blocks.46.", "blocks.47.", "blocks.48.", "blocks.49."]
      save:
        save_every: 50                  # one checkpoint per epoch (~90 MB each at rank 16)
        max_step_saves_to_keep: 60
      datasets:
        - resolution: [512]             # ~0.25 MP, for stills and clips
          caption_dropout_rate: 0.05
      train:
        steps: 3000                     # epochs x items: 60 x 50
        optimizer: "automagic3"
        lr: 0.000001                    # Automagic's start; it sets its own rate from here
        optimizer_params:
          weight_decay: 0.0
          fused: false                  # keeps max_grad_norm clipping and a normal step, as Fizgig does
        max_grad_norm: 1.0
        timestep_type: "shift"          # required for train_lownoise_pct to act
        content_or_style: "balanced"
        ema_config:
          use_ema: true
          ema_decay: 0.98
      model:
        assistant_lora_path: "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v3.safetensors"
        model_kwargs:
          train_lownoise_pct: 60        # needs §4.1
```

Everything else in the template stays the same: layer offloading, `low_vram`, `convrot8`/`nvfp4`, `cache_text_embeddings`, `do_audio`.
For Fizgig's "More Blocks" mode (6–49, which Fizgig suggests only for motion concepts), extend the list down to `blocks.6.`.

### 4.3 Optional: make it a choice in the UI

[ui.tsx](extensions_built_in/diffusion_models/ui.tsx#L1024) already has the pattern. The `minimax_h3` entry's
`customModelSelectOptions` holds the "Distillation Handling Method" dropdown, and its `onChange` writes several config
keys. A second dropdown, "Training recipe: ai-toolkit default | Fizgig likeness", would write the train, network and
model keys from §4.2 and clear them again for the default. That is about 50 lines. Resolution and caption dropout are per-dataset settings, so
leave those to the user.

### 4.4 Optional extras, most useful first

1. **Clip stills.** For each clip, take its sharpest frame that shows a face and put it in the image dataset with the clip's
   caption. This is data prep, by hand or with a short script, and needs no trainer change.
2. **TREAD for clip steps.** In `MiniMaxH3Transformer.forward` (the block loop at
   [transformer.py:625](extensions_built_in/diffusion_models/minimax_h3/src/transformer.py#L625)), route a random 50% of
   the video rows past blocks 2–46, during training and for clips only. `x`, `adaln_indices`, `rotary_emb` and the mask
   must be gathered to the rows that stay, then scattered back. This is a speed gain only, so do it if clip steps are your bottleneck.
3. **EMA decay ramp.** Add `use_num_updates` to `EMAConfig` and pass it in `setup_ema`. It's two lines and only affects
   the first few hundred steps.

### 4.5 Not worth porting

These are either already covered by ai-toolkit, not part of the character preset, or Fizgig-specific:
- Fizgig's VRAM planner and block swap. ai-toolkit has layer offloading.
- Turbo-LoRA previews.
- The face-scored likeness gallery.
- Adapter-relative LR.
- Noisy-half LR scaling. It is 100% in the preset, which means off.
- Reference distillation, LoRA Isolation and context LoRA.

## 5. How to check that it actually helps

1. Implement §4.1 and its test.
2. Use a photo-only character set, so the clip extras don't muddy the result. The 50-photo `alinaSvere_zlxny` set you prepared
   for Fizgig works. Use the same seed and the same DiT file in every arm. You can point ai-toolkit at a custom DiT with
   `model_kwargs.dit_fl2va_pruned_path`.
3. Arms:
   - **A:** `template.yml` unchanged.
   - **B:** the §4.2 recipe.
   - Optional **B without density:** B minus `train_lownoise_pct`, back to stock shift 12. This isolates the noise schedule.
   - Optional **B without Automagic:** B with AdamW8bit. This isolates the optimizer. Lower the LR if you try it (§3.2).
4. Check the logs. B should create 120 LoRA modules and print "Loading assistant LoRA". It should not print
   `OstrisLinear: quantized weights found on cpu`. Compare seconds per step against A.
5. Judge every epoch's checkpoint in ComfyUI with fixed prompts and seeds: one still and one 56-frame clip, at strength
   1.0, also without Turbo (Fizgig's v3.7 failure only showed without Turbo). Record:
   - the first epoch that looks like the person;
   - the peak, and how long it holds;
   - whether motion and prompt following drift from the base model.

## 6. Gotchas

- Run Automagic with **`fused: false`**. Fused mode silently skips `max_grad_norm`. Each backward pass applies its own update,
  so it breaks with gradient accumulation and with the two-backward DOP step on the `ltx2.5` branches.
- **DOP with Automagic is untested.** Fizgig switches to AdamW whenever its preservation pass is on: that pass runs on every second step,
  and Automagic read the back-and-forth as overshoot and drove the rate to its floor. ai-toolkit runs DOP inside the
  same step, so that exact failure shouldn't happen. Start without DOP; the Fizgig preset doesn't use it.
- `train_lownoise_pct` does nothing unless `timestep_type` is `shift`.
- Leave `max_step_saves_to_keep` high. The best window can arrive early, and the adapter can degrade late.
- ai-toolkit samples at 28 steps without Turbo, which is slow for clips on 16 GB. Keep `disable_sampling: true` and judge
  in ComfyUI.
- The saved LoRA has no keys for blocks 0–19. ComfyUI loads partial LoRAs without complaint.

## 7. Sources

**ai-toolkit** (branch `minimax-h3`)
- [minimax_h3.py](extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py)
- [ui.tsx](extensions_built_in/diffusion_models/ui.tsx#L1024)
- [jobConfig.ts](ui/src/app/jobs/new/jobConfig.ts)
- [custom_flowmatch_sampler.py](toolkit/samplers/custom_flowmatch_sampler.py)
- [lora_special.py](toolkit/lora_special.py)
- [automagic3.py](toolkit/optimizers/automagic3.py)
- [BaseSDTrainProcess.py](jobs/process/BaseSDTrainProcess.py)
- [SDTrainer.py](extensions_built_in/sd_trainer/SDTrainer.py)
- `config/sets/minimaxh3/template.yml` (on `ltx2.5-diff-output-preservation`, commit `efa00d1`)

**Fizgig code**
- [families/minimax.py](../Fizgig/src/fizgig/families/minimax.py): presets and options
- [minimax/driver.py](../Fizgig/src/fizgig/minimax/driver.py): block windows (`step_frozen_blocks`), `_shift`, `training_loss`
- [minimax/common.py](../Fizgig/src/fizgig/minimax/common.py): `sample_sigmas`, `compute_loss`
- [training/optimizers.py](../Fizgig/src/fizgig/training/optimizers.py)
- [training/ema.py](../Fizgig/src/fizgig/training/ema.py)
- [minimax/model.py](../Fizgig/src/fizgig/minimax/model.py): TREAD

**Fizgig docs**
- [MINIMAX_H3.md](../Fizgig/docs/MINIMAX_H3.md)
- Release notes: [v3.7.0](../Fizgig/docs/RELEASE_NOTES_v3.7.0.md), [v4.0.0](../Fizgig/docs/RELEASE_NOTES_v4.0.0.md),
  [v5.2.0](../Fizgig/docs/RELEASE_NOTES_v5.2.0.md), [v5.4.0](../Fizgig/docs/RELEASE_NOTES_v5.4.0.md),
  [v6.1.0](../Fizgig/docs/RELEASE_NOTES_v6.1.0.md), [v6.2.0](../Fizgig/docs/RELEASE_NOTES_v6.2.0.md)

**Your runs**
- `../Fizgig/user_data/logs/2026-11-19_*.log`. The effective settings are confirmed by the log lines: blocks 20–49 for every
  modality, adapter on, rank 16 with 200 modules, automagic3 from 1e-6, EMA 0.98.

**Web**
- [Fizgig on GitHub](https://github.com/shootthesound/Fizgig). The Reddit thread
  (https://www.reddit.com/r/StableDiffusion/comments/1w5mv4j/fizgig_52_combining_two_minimax_training_methods/) was
  not readable by tools.
