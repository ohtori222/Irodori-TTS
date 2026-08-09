import os
import unittest
from unittest.mock import MagicMock, patch

from irodori_tts.watermark import SilentCipherWatermarker


class TestSilentCipherWatermarkerEnvVar(unittest.TestCase):
    def test_watermark_disabled_by_env_var(self):
        for val in ["1", "true", "TRUE", "yes", "YES", "on", "ON", " 1 "]:
            with self.subTest(val=val):
                with patch.dict(os.environ, {"IRODORI_DISABLE_WATERMARK": val}):
                    with patch("irodori_tts.watermark.logger.info") as mock_log:
                        marker = SilentCipherWatermarker(device="cpu")
                        self.assertTrue(marker.disabled)
                        self.assertFalse(marker.ready)
                        self.assertIsNone(marker.model)
                        mock_log.assert_called_once()
                        self.assertIn(
                            "SilentCipher watermark is disabled by IRODORI_DISABLE_WATERMARK",
                            mock_log.call_args[0][0],
                        )

    def test_watermark_enabled_when_env_var_unset_or_false(self):
        for val in ["0", "false", "no", "off", ""]:
            with self.subTest(val=val):
                with patch.dict(os.environ, {"IRODORI_DISABLE_WATERMARK": val}):
                    with patch(
                        "irodori_tts.watermark.SilentCipherWatermarker._load_backend"
                    ) as mock_load:
                        mock_load.return_value = "fake_model"
                        marker = SilentCipherWatermarker(device="cpu")
                        self.assertFalse(marker.disabled)
                        mock_load.assert_called_once_with(device="cpu", model_type="44.1k")
                        self.assertTrue(marker.ready)

    def test_watermark_enabled_when_env_var_absent(self):
        env = os.environ.copy()
        env.pop("IRODORI_DISABLE_WATERMARK", None)
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "irodori_tts.watermark.SilentCipherWatermarker._load_backend"
            ) as mock_load:
                mock_load.return_value = "fake_model"
                marker = SilentCipherWatermarker(device="cpu")
                self.assertFalse(marker.disabled)
                mock_load.assert_called_once_with(device="cpu", model_type="44.1k")
                self.assertTrue(marker.ready)


class TestWatermarkRuntimeLogMessages(unittest.TestCase):
    """Regression tests for watermark log messages emitted by InferenceRuntime.

    These tests exercise the log-branching logic without constructing a full
    InferenceRuntime — we only need a watermarker mock with ``ready`` and
    ``disabled`` attributes and a minimal code path that mirrors the runtime's
    watermark section.
    """

    @staticmethod
    def _watermark_log_message(*, ready: bool, disabled: bool) -> str:
        """Replicate the runtime's watermark log-branching logic."""
        if ready:
            return ""
        if disabled:
            return (
                "info: SilentCipher watermark is disabled by "
                "IRODORI_DISABLE_WATERMARK; generated audio was not watermarked."
            )
        return (
            "warning: SilentCipher watermark is unavailable; generated audio was not "
            "watermarked."
        )

    def test_disabled_shows_disabled_message(self):
        msg = self._watermark_log_message(ready=False, disabled=True)
        self.assertIn("disabled by IRODORI_DISABLE_WATERMARK", msg)
        self.assertNotIn("unavailable", msg)

    def test_disabled_does_not_show_unavailable(self):
        msg = self._watermark_log_message(ready=False, disabled=True)
        self.assertNotIn("watermark is unavailable", msg)

    def test_unavailable_shows_warning(self):
        msg = self._watermark_log_message(ready=False, disabled=False)
        self.assertIn("watermark is unavailable", msg)
        self.assertTrue(msg.startswith("warning:"))

    def test_ready_emits_no_message(self):
        msg = self._watermark_log_message(ready=True, disabled=False)
        self.assertEqual(msg, "")


if __name__ == "__main__":
    unittest.main()
