import unittest
from unittest.mock import MagicMock, patch
import torch

from irodori_tts.config import ModelConfig
from irodori_tts.rf import sample_euler_rf_cfg


class TestSampleEulerRfCfg(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        # Create mock model
        self.model = MagicMock()
        self.model.device = self.device
        self.model.dtype = self.dtype
        self.model.cfg = ModelConfig(
            latent_dim=16,
            latent_patch_size=1,
            use_caption_condition=True,
            use_speaker_condition=True,
        )

        # Mock encode_conditions
        self.text_state = torch.randn(1, 10, 32)
        self.text_mask = torch.ones(1, 10, dtype=torch.bool)
        self.speaker_state = torch.randn(1, 5, 32)
        self.speaker_mask = torch.ones(1, 5, dtype=torch.bool)
        self.caption_state = torch.randn(1, 8, 32)
        self.caption_mask = torch.ones(1, 8, dtype=torch.bool)

        self.model.encode_conditions.return_value = (
            self.text_state,
            self.text_mask,
            self.speaker_state,
            self.speaker_mask,
            self.caption_state,
            self.caption_mask,
        )

        # Mock build_context_kv_cache
        self.model.build_context_kv_cache.return_value = [
            (torch.randn(1, 4, 10, 8), torch.randn(1, 4, 10, 8), torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8))
        ]

        # Mock forward_with_encoded_conditions
        def fake_forward(x_t, t, **kwargs):
            return torch.zeros_like(x_t)

        self.model.forward_with_encoded_conditions.side_effect = fake_forward

    def test_no_item_calls_in_sampling_loop(self):
        text_input_ids = torch.tensor([[1, 2, 3]])
        text_mask = torch.ones(1, 3, dtype=torch.bool)

        item_call_count = 0
        original_item = torch.Tensor.item

        def spy_item(self, *args, **kwargs):
            nonlocal item_call_count
            item_call_count += 1
            return original_item(self, *args, **kwargs)

        with patch.object(torch.Tensor, "item", spy_item):
            sample_euler_rf_cfg(
                model=self.model,
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                ref_latent=None,
                ref_mask=None,
                sequence_length=12,
                num_steps=10,
                cfg_scale_text=3.0,
                cfg_scale_caption=3.0,
                cfg_scale_speaker=5.0,
                cfg_guidance_mode="independent",
                t_schedule_mode="linear",
                speaker_kv_scale=1.2,
                speaker_kv_min_t=0.5,
                rescale_k=1.0,
                rescale_sigma=0.5,
                seed=42,
            )

        # item() may be called at setup for bool(caption_mask_cond.any().item())
        # but inside the 10-step sampling loop, no .item() calls should occur.
        self.assertLessEqual(item_call_count, 1)

    def test_sampling_modes_run(self):
        text_input_ids = torch.tensor([[1, 2, 3]])
        text_mask = torch.ones(1, 3, dtype=torch.bool)

        for guidance_mode in ["independent", "joint", "alternating"]:
            for schedule_mode in ["linear", "sway"]:
                with self.subTest(guidance_mode=guidance_mode, schedule_mode=schedule_mode):
                    out = sample_euler_rf_cfg(
                        model=self.model,
                        text_input_ids=text_input_ids,
                        text_mask=text_mask,
                        ref_latent=None,
                        ref_mask=None,
                        sequence_length=8,
                        num_steps=5,
                        cfg_scale_text=3.0,
                        cfg_scale_caption=3.0,
                        cfg_scale_speaker=3.0,
                        cfg_guidance_mode=guidance_mode,
                        t_schedule_mode=schedule_mode,
                        sway_coeff=-0.5,
                        seed=123,
                    )
                    self.assertEqual(out.shape, (1, 8, 16))


if __name__ == "__main__":
    unittest.main()
