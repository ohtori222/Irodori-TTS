import os
import unittest
from unittest.mock import patch

from irodori_tts.inference_runtime import _watermark_unavailable_message
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
    """Regression tests for the runtime's _watermark_unavailable_message helper.

    These call the real production helper from inference_runtime to ensure the
    disabled vs unavailable distinction is maintained.
    """

    def test_disabled_shows_disabled_message(self):
        msg = _watermark_unavailable_message(disabled=True)
        self.assertIn("disabled by IRODORI_DISABLE_WATERMARK", msg)

    def test_disabled_does_not_show_unavailable(self):
        msg = _watermark_unavailable_message(disabled=True)
        self.assertNotIn("watermark is unavailable", msg)

    def test_unavailable_shows_warning(self):
        msg = _watermark_unavailable_message(disabled=False)
        self.assertIn("watermark is unavailable", msg)
        self.assertTrue(msg.startswith("warning:"))


if __name__ == "__main__":
    unittest.main()
