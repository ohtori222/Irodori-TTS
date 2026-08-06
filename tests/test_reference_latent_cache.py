import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from irodori_tts import inference_runtime


class _FakeCodec:
    sample_rate = 16_000
    model = SimpleNamespace(hop_length=320)

    def __init__(self) -> None:
        self.encode_count = 0
        self._count_lock = threading.Lock()

    def encode_waveform(self, waveform, *, sample_rate, normalize_db, ensure_max):
        del waveform, sample_rate, normalize_db, ensure_max
        with self._count_lock:
            self.encode_count += 1
            value = float(self.encode_count)
        time.sleep(0.02)
        return torch.full((1, 8, 2), value, dtype=torch.float32)


def _make_runtime(codec: _FakeCodec) -> inference_runtime.InferenceRuntime:
    runtime = object.__new__(inference_runtime.InferenceRuntime)
    runtime.key = SimpleNamespace(
        codec_repo="test-codec",
        codec_precision="fp32",
        codec_deterministic_encode=True,
    )
    runtime.model = torch.nn.Linear(1, 1)
    runtime.model_device = torch.device("cpu")
    runtime.codec_device = torch.device("cuda")
    runtime.model_cfg = SimpleNamespace(
        latent_dim=2,
        latent_patch_size=1,
        speaker_patch_size=1,
        use_speaker_condition_resolved=True,
    )
    runtime.codec = codec
    runtime.default_max_ref_seconds = 30.0
    return runtime


class ReferenceLatentCacheTests(unittest.TestCase):
    def test_first_request_encodes_and_second_request_hits_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            ref_path = Path(temp_dir) / "reference.wav"
            ref_path.write_bytes(b"reference-v1")
            codec = _FakeCodec()
            runtime = _make_runtime(codec)
            request = inference_runtime.SamplingRequest(
                text="test",
                ref_wavs=[str(ref_path)],
                ref_normalize_db=None,
            )
            audio = torch.zeros(1, 16)

            with (
                patch.dict(os.environ, {"IRODORI_REFERENCE_LATENT_CACHE_DIR": str(cache_dir)}),
                patch.object(inference_runtime, "_load_audio", return_value=(audio, 16_000)),
            ):
                first_messages: list[str] = []
                first, first_mask = runtime._load_reference_latent(
                    req=request,
                    batch_size=1,
                    messages=first_messages,
                )
                cache_path = next(cache_dir.glob("*.pt"))
                cache_mtime = cache_path.stat().st_mtime_ns

                second_messages: list[str] = []
                second, second_mask = runtime._load_reference_latent(
                    req=request,
                    batch_size=1,
                    messages=second_messages,
                )

            self.assertEqual(codec.encode_count, 1)
            self.assertEqual(first.dtype, torch.float32)
            self.assertEqual(first.device.type, "cpu")
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(first_mask, second_mask))
            self.assertEqual(cache_path.stat().st_mtime_ns, cache_mtime)
            self.assertTrue(any("cache miss" in message for message in first_messages))
            self.assertTrue(any("cache hit" in message for message in second_messages))

    def test_content_change_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            ref_path = Path(temp_dir) / "reference.wav"
            ref_path.write_bytes(b"reference-v1")
            codec = _FakeCodec()
            runtime = _make_runtime(codec)
            request = inference_runtime.SamplingRequest(
                text="test",
                ref_wavs=[str(ref_path)],
                ref_normalize_db=-16.0,
            )
            audio = torch.zeros(1, 16)

            with (
                patch.dict(os.environ, {"IRODORI_REFERENCE_LATENT_CACHE_DIR": str(cache_dir)}),
                patch.object(inference_runtime, "_load_audio", return_value=(audio, 16_000)),
            ):
                runtime._load_reference_latent(req=request, batch_size=1, messages=[])
                ref_path.write_bytes(b"reference-v2")
                runtime._load_reference_latent(req=request, batch_size=1, messages=[])

            self.assertEqual(codec.encode_count, 2)
            self.assertEqual(len(list(cache_dir.glob("*.pt"))), 2)

    def test_invalid_cache_is_reencoded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            ref_path = Path(temp_dir) / "reference.wav"
            ref_path.write_bytes(b"reference")
            codec = _FakeCodec()
            runtime = _make_runtime(codec)
            request = inference_runtime.SamplingRequest(
                text="test",
                ref_wavs=[str(ref_path)],
                ref_normalize_db=None,
            )
            audio = torch.zeros(1, 16)

            with (
                patch.dict(os.environ, {"IRODORI_REFERENCE_LATENT_CACHE_DIR": str(cache_dir)}),
                patch.object(inference_runtime, "_load_audio", return_value=(audio, 16_000)),
            ):
                runtime._load_reference_latent(req=request, batch_size=1, messages=[])
                cache_path = next(cache_dir.glob("*.pt"))
                torch.save(torch.full((8, 2), float("nan")), cache_path)
                messages: list[str] = []
                runtime._load_reference_latent(req=request, batch_size=1, messages=messages)

            self.assertEqual(codec.encode_count, 2)
            self.assertTrue(any("cache miss" in message for message in messages))

    def test_concurrent_first_requests_encode_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            ref_path = Path(temp_dir) / "reference.wav"
            ref_path.write_bytes(b"reference")
            codec = _FakeCodec()
            runtime = _make_runtime(codec)
            request = inference_runtime.SamplingRequest(
                text="test",
                ref_wavs=[str(ref_path)],
                ref_normalize_db=None,
            )
            audio = torch.zeros(1, 16)

            def load_one() -> torch.Tensor:
                messages: list[str] = []
                result, _ = runtime._load_reference_latent(
                    req=request,
                    batch_size=1,
                    messages=messages,
                )
                return result

            with (
                patch.dict(os.environ, {"IRODORI_REFERENCE_LATENT_CACHE_DIR": str(cache_dir)}),
                patch.object(inference_runtime, "_load_audio", return_value=(audio, 16_000)),
            ):
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=2) as executor:
                    first, second = executor.map(lambda _: load_one(), range(2))

            self.assertEqual(codec.encode_count, 1)
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
