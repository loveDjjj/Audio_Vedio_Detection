from pathlib import Path
import sys
import unittest
from unittest import mock

from src.utils.avhubert_env import import_avhubert_modules

try:
    from src.models.avhubert_backbone import load_torch_checkpoint
except ModuleNotFoundError as exc:  # pragma: no cover - local env may not have torch installed
    load_torch_checkpoint = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class AVHubertEnvTest(unittest.TestCase):
    @unittest.skipIf(IMPORT_ERROR is not None, f"AV-HuBERT env imports unavailable: {IMPORT_ERROR}")
    def test_import_avhubert_modules(self) -> None:
        with mock.patch.object(sys, "argv", ["train_avhubert_classifier.py"]):
            fairseq, hubert_pretraining, hubert, hubert_asr = import_avhubert_modules(
                Path("third_party/av_hubert")
            )

        self.assertTrue(getattr(fairseq, "__file__", None))
        self.assertEqual(hubert_pretraining.__name__, "avhubert.hubert_pretraining")
        self.assertEqual(hubert.__name__, "avhubert.hubert")
        self.assertEqual(hubert_asr.__name__, "avhubert.hubert_asr")

    @unittest.skipIf(load_torch_checkpoint is None, f"torch-backed checkpoint loader unavailable: {IMPORT_ERROR}")
    @mock.patch("src.models.avhubert_backbone.torch.load")
    def test_load_torch_checkpoint_uses_weights_only_false_when_supported(self, mock_load) -> None:
        mock_load.return_value = {"ok": True}

        result = load_torch_checkpoint(Path("model/self_large_vox_433h.pt"))

        self.assertEqual(result, {"ok": True})
        mock_load.assert_called_once_with(
            str(Path("model/self_large_vox_433h.pt")),
            map_location="cpu",
            weights_only=False,
        )

    @unittest.skipIf(load_torch_checkpoint is None, f"torch-backed checkpoint loader unavailable: {IMPORT_ERROR}")
    @mock.patch("src.models.avhubert_backbone.torch.load")
    def test_load_torch_checkpoint_falls_back_when_weights_only_kw_is_unsupported(self, mock_load) -> None:
        mock_load.side_effect = [TypeError("unexpected keyword argument 'weights_only'"), {"ok": True}]

        result = load_torch_checkpoint(Path("model/self_large_vox_433h.pt"))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_load.call_count, 2)
        self.assertEqual(
            mock_load.call_args_list[0],
            mock.call(
                str(Path("model/self_large_vox_433h.pt")),
                map_location="cpu",
                weights_only=False,
            ),
        )
        self.assertEqual(
            mock_load.call_args_list[1],
            mock.call(
                str(Path("model/self_large_vox_433h.pt")),
                map_location="cpu",
            ),
        )


if __name__ == "__main__":
    unittest.main()
