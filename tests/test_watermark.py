import os
import unittest
from unittest.mock import patch

from irodori_tts.watermark import SilentCipherWatermarker


class TestSilentCipherWatermarkerEnvVar(unittest.TestCase):
    def test_watermark_disabled_by_env_var(self):
        for val in ["1", "true", "TRUE", "yes", "YES", "on", "ON", " 1 "]:
            with self.subTest(val=val):
                with patch.dict(os.environ, {"IRODORI_DISABLE_WATERMARK": val}):
                    with patch("irodori_tts.watermark.logger.info") as mock_log:
                        marker = SilentCipherWatermarker(device="cpu")
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
                mock_load.assert_called_once_with(device="cpu", model_type="44.1k")
                self.assertTrue(marker.ready)


if __name__ == "__main__":
    unittest.main()
