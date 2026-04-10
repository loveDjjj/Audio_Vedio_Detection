from pathlib import Path
import sys
import unittest
from unittest import mock

from src.utils.avhubert_env import import_avhubert_modules


class AVHubertEnvTest(unittest.TestCase):
    def test_import_avhubert_modules(self) -> None:
        with mock.patch.object(sys, "argv", ["train_avhubert_classifier.py"]):
            fairseq, hubert_pretraining, hubert, hubert_asr = import_avhubert_modules(
                Path("third_party/av_hubert")
            )

        self.assertTrue(getattr(fairseq, "__file__", None))
        self.assertEqual(hubert_pretraining.__name__, "avhubert.hubert_pretraining")
        self.assertEqual(hubert.__name__, "avhubert.hubert")
        self.assertEqual(hubert_asr.__name__, "avhubert.hubert_asr")


if __name__ == "__main__":
    unittest.main()
