from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest import mock

from src.utils.project import load_config


class FakeAVCelebWrapperTest(unittest.TestCase):
    def test_fakeavceleb_preprocess_config_paths(self) -> None:
        config = load_config(Path("configs/fakeavceleb_preprocess.yaml"))
        self.assertEqual(config["paths"]["raw_video_root"], "dataset/FakeAVCeleb")
        self.assertEqual(config["paths"]["split_dir"], "splits/fakeavceleb_real_fullfake")
        self.assertEqual(
            config["paths"]["mouth_roi_root"],
            "artifacts/avhubert/fakeavceleb_real_fullfake/mouth_roi",
        )

    def test_fakeavceleb_classifier_config_paths(self) -> None:
        config = load_config(Path("configs/fakeavceleb_classifier.yaml"))
        self.assertEqual(config["paths"]["raw_video_root"], "dataset/FakeAVCeleb")
        self.assertEqual(
            config["paths"]["audio_feature_root"],
            "artifacts/avhubert/fakeavceleb_real_fullfake/audio_features",
        )
        self.assertEqual(
            config["paths"]["output_root"],
            "outputs/avhubert/fakeavceleb_real_fullfake",
        )

    @mock.patch("src.preprocess.runtime.run_preprocess_from_config")
    def test_preprocess_wrapper_uses_fakeavceleb_config(self, mock_run) -> None:
        mock_run.return_value = {"ok": True}
        with self.assertRaises(SystemExit) as exc:
            runpy.run_path("scripts/preprocess_fakeavceleb.py", run_name="__main__")
        self.assertEqual(exc.exception.code, 0)
        called_path = mock_run.call_args.args[0]
        self.assertEqual(Path(called_path), Path("configs/fakeavceleb_preprocess.yaml"))

    def test_audio_cache_wrapper_uses_fakeavceleb_config(self) -> None:
        fake_module = types.ModuleType("src.data.audio_cache_runtime")
        mock_run = mock.Mock(return_value={"ok": True})
        fake_module.run_audio_cache_from_config = mock_run
        with mock.patch.dict(sys.modules, {"src.data.audio_cache_runtime": fake_module}):
            with self.assertRaises(SystemExit) as exc:
                runpy.run_path("scripts/cache_fakeavceleb_audio_features.py", run_name="__main__")
        self.assertEqual(exc.exception.code, 0)
        called_path = mock_run.call_args.args[0]
        self.assertEqual(Path(called_path), Path("configs/fakeavceleb_classifier.yaml"))

    @mock.patch("subprocess.call")
    def test_train_wrapper_forwards_config(self, mock_call) -> None:
        mock_call.return_value = 0
        with mock.patch("sys.argv", ["train_fakeavceleb.py"]):
            with self.assertRaises(SystemExit) as exc:
                runpy.run_path("scripts/train_fakeavceleb.py", run_name="__main__")
        self.assertEqual(exc.exception.code, 0)
        command = mock_call.call_args.args[0]
        self.assertEqual(
            command[1],
            "scripts/train_avhubert_classifier.py",
        )
        self.assertEqual(command[2], "--config")
        self.assertEqual(
            Path(command[3]),
            Path("configs/fakeavceleb_classifier.yaml"),
        )


if __name__ == "__main__":
    unittest.main()
