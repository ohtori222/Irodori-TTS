import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from irodori_tts import inference_runtime


class _FakeRuntime:
    def __init__(self) -> None:
        self.active_requests = 0
        self.unload_count = 0

    def _acquire_request_lease(self) -> None:
        self.active_requests += 1

    def _release_request_lease(self) -> None:
        if self.active_requests <= 0:
            raise RuntimeError("lease underflow")
        self.active_requests -= 1

    def has_active_requests(self) -> bool:
        return self.active_requests > 0

    def unload(self) -> None:
        self.unload_count += 1




class _FakeTokenizer:
    def batch_encode(self, texts, *, max_length):
        del max_length
        batch = len(texts)
        return (
            torch.ones((batch, 2), dtype=torch.long),
            torch.ones((batch, 2), dtype=torch.bool),
        )


class _FakeCodec:
    sample_rate = 16_000
    dtype = torch.float32

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.model = SimpleNamespace(hop_length=320)
        self.infer_lock_states: list[bool] = []

    def decode_latent(self, latent):
        self.infer_lock_states.append(self.runtime._infer_lock.locked())
        return torch.zeros((latent.shape[0], 1, 64), dtype=torch.float32)


class _DisabledWatermarker:
    ready = False
    disabled = True


def _make_synthesize_runtime() -> inference_runtime.InferenceRuntime:
    runtime = object.__new__(inference_runtime.InferenceRuntime)
    runtime.key = SimpleNamespace(
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
        compile_model=False,
    )
    runtime.model_device = torch.device("cpu")
    runtime.codec_device = torch.device("cpu")
    runtime.model_cfg = SimpleNamespace(
        use_caption_condition=False,
        use_speaker_condition_resolved=False,
        use_duration_predictor=False,
        latent_patch_size=1,
        latent_dim=2,
        speaker_patch_size=1,
    )
    runtime.train_cfg = None
    runtime.model = torch.nn.Linear(2, 2)
    runtime.tokenizer = _FakeTokenizer()
    runtime.caption_tokenizer = None
    runtime.default_text_max_len = 16
    runtime.default_caption_max_len = 16
    runtime.default_max_ref_seconds = 30.0
    runtime.watermarker = _DisabledWatermarker()
    runtime._infer_lock = __import__("threading").Lock()
    runtime._codec_lock = __import__("threading").Lock()
    runtime._request_lease_lock = __import__("threading").Lock()
    runtime._active_requests = 0
    runtime._model_dtype = torch.float32
    runtime._lora_adapter_names = {}
    runtime.codec = _FakeCodec(runtime)
    return runtime


class RuntimePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        inference_runtime._RUNTIME_CACHE_KEY = None
        inference_runtime._RUNTIME_CACHE_VALUE = None

    def tearDown(self) -> None:
        inference_runtime._RUNTIME_CACHE_KEY = None
        inference_runtime._RUNTIME_CACHE_VALUE = None

    def test_cpu_codec_pipeline_requires_non_cpu_model(self) -> None:
        self.assertTrue(
            inference_runtime._should_pipeline_cpu_codec(
                torch.device("cuda"),
                torch.device("cpu"),
            )
        )
        self.assertFalse(
            inference_runtime._should_pipeline_cpu_codec(
                torch.device("cuda"),
                torch.device("cuda"),
            )
        )
        self.assertFalse(
            inference_runtime._should_pipeline_cpu_codec(
                torch.device("cpu"),
                torch.device("cpu"),
            )
        )

    def test_runtime_lease_blocks_cache_replacement_and_clear(self) -> None:
        key_a = inference_runtime.RuntimeKey(checkpoint="a", model_device="cpu")
        key_b = inference_runtime.RuntimeKey(checkpoint="b", model_device="cpu")
        runtime_a = _FakeRuntime()
        runtime_b = _FakeRuntime()

        with patch.object(
            inference_runtime.InferenceRuntime,
            "from_key",
            side_effect=[runtime_a, runtime_b],
        ) as load_runtime:
            with inference_runtime.lease_cached_runtime(key_a) as (leased_a, reloaded):
                self.assertIs(leased_a, runtime_a)
                self.assertTrue(reloaded)
                self.assertEqual(runtime_a.active_requests, 1)

                with inference_runtime.lease_cached_runtime(key_a) as (leased_again, reloaded_again):
                    self.assertIs(leased_again, runtime_a)
                    self.assertFalse(reloaded_again)
                    self.assertEqual(runtime_a.active_requests, 2)

                self.assertEqual(runtime_a.active_requests, 1)
                with self.assertRaisesRegex(RuntimeError, "synthesis requests are active"):
                    inference_runtime.get_cached_runtime(key_b)
                with self.assertRaisesRegex(RuntimeError, "synthesis requests are active"):
                    inference_runtime.clear_cached_runtime()

            self.assertEqual(runtime_a.active_requests, 0)
            loaded_b, reloaded_b = inference_runtime.get_cached_runtime(key_b)
            self.assertIs(loaded_b, runtime_b)
            self.assertTrue(reloaded_b)
            self.assertEqual(runtime_a.unload_count, 1)
            self.assertEqual(load_runtime.call_count, 2)

        inference_runtime.clear_cached_runtime()
        self.assertEqual(runtime_b.unload_count, 1)


    def test_cpu_pipeline_releases_model_lock_before_decode(self) -> None:
        runtime = _make_synthesize_runtime()
        request = inference_runtime.SamplingRequest(
            text="test",
            no_ref=True,
            seconds=0.5,
            num_steps=1,
            trim_tail=False,
        )

        def fake_sample(*, sequence_length, **kwargs):
            del kwargs
            return torch.zeros((1, sequence_length, 2), dtype=torch.float32)

        with (
            patch.object(inference_runtime, "_should_pipeline_cpu_codec", return_value=True),
            patch.object(inference_runtime, "sample_euler_rf_cfg", side_effect=fake_sample),
        ):
            runtime.synthesize(request)

        self.assertEqual(runtime.codec.infer_lock_states, [False])

    def test_non_pipeline_decode_keeps_model_lock(self) -> None:
        runtime = _make_synthesize_runtime()
        request = inference_runtime.SamplingRequest(
            text="test",
            no_ref=True,
            seconds=0.5,
            num_steps=1,
            trim_tail=False,
        )

        def fake_sample(*, sequence_length, **kwargs):
            del kwargs
            return torch.zeros((1, sequence_length, 2), dtype=torch.float32)

        with (
            patch.object(inference_runtime, "_should_pipeline_cpu_codec", return_value=False),
            patch.object(inference_runtime, "sample_euler_rf_cfg", side_effect=fake_sample),
        ):
            runtime.synthesize(request)

        self.assertEqual(runtime.codec.infer_lock_states, [True])


if __name__ == "__main__":
    unittest.main()
