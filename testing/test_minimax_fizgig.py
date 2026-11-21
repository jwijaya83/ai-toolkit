"""Checks for the Fizgig MiniMax H3 recipe (toolkit/fizgig.py) and its model
knobs. Run: ./venv/bin/python -m unittest testing/test_minimax_fizgig.py"""

import os
import sys
import tempfile
import time
import unittest

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (  # noqa: E402
    lownoise_shift,
    parse_tread,
    scheduler_config,
)
from extensions_built_in.diffusion_models.minimax_h3.src.packing import (  # noqa: E402
    build_packed_sequence,
    pad_layouts_to_batch,
)
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (  # noqa: E402
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)
from toolkit.config_modules import NetworkConfig  # noqa: E402
from toolkit.fizgig import add_clip_stills, apply_fizgig_recipe, pick_still_frame  # noqa: E402
from toolkit.lora_special import LoRASpecialNetwork  # noqa: E402
from toolkit.samplers.custom_flowmatch_sampler import (  # noqa: E402
    CustomFlowMatchEulerDiscreteScheduler,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_h3(num_layers):
    torch.manual_seed(0)
    return MiniMaxH3Transformer(
        MiniMaxH3TransformerParams(
            hidden_size=32, num_layers=num_layers, num_attention_heads=2,
            attention_head_dim=32, ffn_hidden_size=32, text_dim=16,
            timestep_input_dim=16, time_embed_hidden_size=16, time_embed_dim=8,
            rope_inv_freq_len=4,
        )
    )


def template_process():
    with open(os.path.join(ROOT, "config/sets/minimaxh3/template.yml")) as f:
        return yaml.safe_load(f)["config"]["process"][0]


class RecipeTest(unittest.TestCase):
    def test_off_leaves_config_alone(self):
        proc = template_process()
        proc["model"]["model_kwargs"]["fizgig"] = False
        self.assertIs(apply_fizgig_recipe(proc), proc)

    def test_on_applies_recipe_and_respects_yaml_knobs(self):
        proc = template_process()
        proc["model"]["model_kwargs"]["tread"] = False  # a YAML knob wins over the default
        proc["datasets"][0]["resolution"] = [896]
        cfg = apply_fizgig_recipe(proc)
        self.assertEqual(cfg["datasets"], proc["datasets"])  # data stays the YAML's
        train, kw = cfg["train"], cfg["model"]["model_kwargs"]
        self.assertEqual(train["optimizer"], "automagic3")
        self.assertEqual(train["lr"], 1e-6)
        self.assertEqual(train["optimizer_params"], {"weight_decay": 0.0, "fused": False})
        self.assertEqual(train["ema_config"]["ema_decay"], 0.98)
        self.assertTrue(train["ema_config"]["use_num_updates"])
        nk = cfg["network"]["network_kwargs"]
        self.assertEqual(nk["only_if_contains"][0], "blocks.20.")
        self.assertEqual(len(nk["only_if_contains"]), 30)
        self.assertIn("adaln_proj", nk["ignore_if_contains"])
        self.assertEqual(kw["train_lownoise_pct"], 60)
        self.assertIs(kw["tread"], False)
        self.assertEqual(proc["train"]["optimizer"], "adamw8bit")  # the input is not mutated

    def test_other_arch_refused(self):
        proc = template_process()
        proc["model"]["arch"] = "minimax_h3_vsa"
        with self.assertRaises(ValueError):
            apply_fizgig_recipe(proc)


class DensityTest(unittest.TestCase):
    def test_60_percent_below_half(self):
        # the trainer's own path: shift schedule + uniform ("balanced") index draw
        sched = CustomFlowMatchEulerDiscreteScheduler(
            **{**scheduler_config, "shift": lownoise_shift(60)}
        )
        sched.set_train_timesteps(1000, device="cpu", timestep_type="shift")
        g = torch.Generator().manual_seed(0)
        idx = torch.randint(0, 999, (200_000,), generator=g)
        sigma = sched.timesteps[idx] / 1000.0
        self.assertAlmostEqual(float((sigma < 0.5).float().mean()), 0.60, delta=0.01)
        self.assertLess(float(sigma.min()), 0.002)
        with self.assertRaises(ValueError):
            lownoise_shift(100)


class LoraWindowTest(unittest.TestCase):
    def test_blocks_20_49_without_refiner(self):
        class Sd:  # what the trainer passes as base_model
            use_old_lokr_format = False

            def get_transformer_block_names(self):
                return ["blocks"]

        sd = Sd()
        nk = apply_fizgig_recipe(template_process())["network"]["network_kwargs"]
        net = LoRASpecialNetwork(
            text_encoder=None, unet=tiny_h3(50), lora_dim=4, alpha=4,
            train_unet=True, train_text_encoder=False,
            network_config=NetworkConfig(type="lora", linear=4, linear_alpha=4),
            network_type="lora", transformer_only=True, is_transformer=True,
            base_model=sd, target_lin_modules=["MiniMaxH3Transformer"], **nk,
        )
        names = [m.lora_name for m in net.unet_loras]
        self.assertEqual(len(names), 120)
        self.assertFalse(any("refiner" in n or "adaln" in n for n in names))
        self.assertFalse(any(f"blocks_{i}_" in n.replace("$$", "_") for n in names for i in range(20)))


class TreadTest(unittest.TestCase):
    def test_routed_rows_skip_blocks_and_rejoin_unchanged(self):
        model = tiny_h3(4)
        layout = build_packed_sequence(
            text_token_tags=torch.ones(5, dtype=torch.long), num_latent_frames=2,
            latent_height=8, latent_width=8, num_audio_latents=3,
        )
        pos, tags, vid, aud, txt, _ = pad_layouts_to_batch([layout])
        row_t = torch.full(tags.shape, 0.3)
        row_t[:, aud] = 0.6
        keep = torch.ones(tags.shape[1], dtype=torch.bool)
        drop = vid[torch.randperm(vid.shape[0])[: vid.shape[0] // 2]]
        keep[drop] = False
        keep = keep.nonzero().squeeze(1)

        seen = {}
        for i, block in enumerate(model.blocks):
            block.register_forward_hook(lambda m, a, o, i=i: seen.__setitem__(i, (a, o)))
        out, _ = model(
            hidden_states=torch.randn(1, vid.shape[0], 96),
            audio_hidden_states=torch.randn(1, aud.shape[0], 32),
            encoder_hidden_states=torch.randn(1, 5, 16),
            row_timesteps=row_t, token_tags=tags, position_ids=pos,
            video_indices=vid, audio_indices=aud, text_indices=txt,
            tread=(1, 3, keep),
        )
        (x0, temb, adaln, rope, mask, vsa), out0 = seen[0]
        self.assertEqual(seen[1][0][0].shape[1], keep.shape[0])  # blocks 1-2 see the kept rows only
        x3 = seen[3][0][0]
        torch.testing.assert_close(x3[:, drop], out0[:, drop])  # routed rows rejoin unchanged
        torch.testing.assert_close(x3[:, keep], seen[2][1])  # kept rows carry blocks 1-2
        # block 1 ran on the kept rows with their own rope / adaln rows
        ref = model.blocks[1](
            out0[:, keep], temb, adaln[:, keep], (rope[0][:, keep], rope[1][:, keep]), mask, vsa
        )
        torch.testing.assert_close(seen[1][1], ref)
        out.sum().backward()
        self.assertIsNotNone(model.blocks[1].mlp.fc1.weight.grad)

    def test_parse(self):
        self.assertEqual(parse_tread("0.5@2-47"), (0.5, 2, 47))
        self.assertIsNone(parse_tread(False))
        with self.assertRaises(ValueError):
            parse_tread("1.5@2-47")


class ClipStillTest(unittest.TestCase):
    def test_sharpest_frame_and_caption_mirror(self):
        import cv2
        import numpy as np

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "clip.mp4")
            writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 24, (128, 128))
            for k in range(12):
                frame = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
                writer.write(frame if k == 7 else cv2.GaussianBlur(frame, (9, 9), 3))
            writer.release()
            self.assertEqual(pick_still_frame(clip)[0], 7)

            with open(os.path.join(root, "clip.txt"), "w") as f:
                f.write("zxqv woman")
            (still,) = add_clip_stills([clip], root, ".txt")
            self.assertEqual(still, os.path.join(root, ".clip_stills", "clip.png"))
            self.assertEqual(open(still[:-4] + ".txt").read(), "zxqv woman")
            mtime = os.path.getmtime(still)
            time.sleep(0.05)
            with open(os.path.join(root, "clip.txt"), "w") as f:
                f.write("zxqv woman, smiling")
            add_clip_stills([clip], root, ".txt")
            self.assertEqual(os.path.getmtime(still), mtime)  # not re-picked
            self.assertEqual(open(still[:-4] + ".txt").read(), "zxqv woman, smiling")


if __name__ == "__main__":
    unittest.main()
