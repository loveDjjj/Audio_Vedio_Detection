from pathlib import Path
import pickle
import runpy
import sys
import types
import unittest
import uuid
from unittest import mock

import numpy as np
import yaml

from src.preprocess import mouth_roi, runtime
from src.utils.project import load_config


class FakeAVCelebWrapperTest(unittest.TestCase):
    def test_fakeavceleb_preprocess_config_paths(self) -> None:
        config = load_config(Path("configs/fakeavceleb_preprocess.yaml"))
        self.assertEqual(config["paths"]["raw_video_root"], "/data/OneDay/FakeAVCeleb")
        self.assertEqual(config["paths"]["split_dir"], "splits/fakeavceleb_real_fullfake")
        self.assertEqual(config["preprocess"]["stage"], "detect")
        self.assertEqual(
            config["paths"]["mouth_roi_root"],
            "/data/OneDay/artifacts/avhubert/fakeavceleb_real_fullfake/mouth_roi",
        )

    def test_fakeavceleb_align_config_paths(self) -> None:
        config = load_config(Path("configs/fakeavceleb_preprocess_align.yaml"))
        self.assertEqual(config["preprocess"]["stage"], "align")
        self.assertEqual(config["runtime"]["devices"], [])
        self.assertEqual(config["runtime"]["workers_per_device"], 6)

    def test_fakeavceleb_classifier_config_paths(self) -> None:
        config = load_config(Path("configs/fakeavceleb_classifier.yaml"))
        self.assertEqual(config["paths"]["raw_video_root"], "/data/OneDay/FakeAVCeleb")
        self.assertEqual(
            config["paths"]["checkpoint_path"],
            "/data/OneDay/model/self_large_vox_433h.pt",
        )
        self.assertEqual(
            config["paths"]["audio_feature_root"],
            "/data/OneDay/artifacts/avhubert/fakeavceleb_real_fullfake/audio_features",
        )
        self.assertEqual(
            config["paths"]["output_root"],
            "outputs/avhubert/fakeavceleb_real_fullfake",
        )

    @mock.patch("src.preprocess.runtime.run_preprocess_from_config")
    def test_preprocess_wrapper_uses_fakeavceleb_config(self, mock_run) -> None:
        mock_run.return_value = {"ok": True}
        with mock.patch("sys.argv", ["preprocess_fakeavceleb.py"]):
            with self.assertRaises(SystemExit) as exc:
                runpy.run_path("scripts/preprocess_fakeavceleb.py", run_name="__main__")
        self.assertEqual(exc.exception.code, 0)
        called_path = mock_run.call_args.args[0]
        self.assertEqual(Path(called_path), Path("configs/fakeavceleb_preprocess.yaml"))

    @mock.patch("src.preprocess.runtime.run_preprocess_from_config")
    def test_preprocess_wrapper_accepts_custom_config(self, mock_run) -> None:
        mock_run.return_value = {"ok": True}
        with mock.patch("sys.argv", ["preprocess_fakeavceleb.py", "--config", "configs/fakeavceleb_preprocess_align.yaml"]):
            with self.assertRaises(SystemExit) as exc:
                runpy.run_path("scripts/preprocess_fakeavceleb.py", run_name="__main__")
        self.assertEqual(exc.exception.code, 0)
        called_path = mock_run.call_args.args[0]
        self.assertEqual(Path(called_path), Path("configs/fakeavceleb_preprocess_align.yaml"))

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


class PreprocessOptimizationTest(unittest.TestCase):
    def _temp_dir(self):
        root = Path(".tmp_unittest")
        root.mkdir(exist_ok=True)
        path = root / f"preprocess-{uuid.uuid4().hex}"
        path.mkdir()
        return path

    def test_process_manifest_all_reuses_loaded_frames_for_detect_and_align(self) -> None:
        root = self._temp_dir()
        raw_root = root / "raw"
        landmark_root = root / "landmarks"
        mouth_root = root / "mouth"
        raw_root.mkdir(parents=True)
        landmark_root.mkdir(parents=True)
        mouth_root.mkdir(parents=True)

        file_id = "sample"
        (raw_root / f"{file_id}.mp4").write_bytes(b"fake")
        manifest_path = root / "all.list"
        manifest_path.write_text(f"{file_id}\n", encoding="utf-8")
        mean_face_path = root / "mean_face.npy"
        np.save(mean_face_path, np.zeros((68, 2), dtype=np.float32))

        frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]
        landmark = np.zeros((68, 2), dtype=np.float32)
        fake_dlib = mock.Mock()
        fake_dlib.shape_predictor.return_value = mock.Mock()

        with mock.patch("src.preprocess.mouth_roi._load_dependencies", return_value=fake_dlib):
            with mock.patch("src.preprocess.mouth_roi.build_cnn_detector", return_value=object()):
                with mock.patch("src.preprocess.mouth_roi.load_video_frames", return_value=frames) as mock_load:
                    with mock.patch(
                        "src.preprocess.mouth_roi.detect_landmarks_for_frames",
                        return_value=[landmark, landmark],
                    ):
                        with mock.patch("src.preprocess.mouth_roi.interpolate_landmarks", return_value=[landmark, landmark]):
                            with mock.patch(
                                "src.preprocess.mouth_roi.crop_mouth_sequence",
                                return_value=[np.zeros((4, 4, 3), dtype=np.uint8)],
                            ):
                                with mock.patch("src.preprocess.mouth_roi.write_video"):
                                    summary = mouth_roi.process_manifest(
                                        raw_video_root=raw_root,
                                        manifest_path=manifest_path,
                                        landmark_root=landmark_root,
                                        mouth_roi_root=mouth_root,
                                        face_predictor_path=root / "shape_predictor.dat",
                                        mean_face_path=mean_face_path,
                                        cnn_detector_path=root / "cnn_detector.dat",
                                        stage="all",
                                        show_progress=False,
                                    )

        self.assertEqual(summary["mouth_roi_written"], 1)
        self.assertEqual(mock_load.call_count, 1)

    def test_detect_landmarks_for_video_batches_cnn_detector_calls(self) -> None:
        class FakePoint:
            def __init__(self, value: int) -> None:
                self.x = value
                self.y = value

        class FakeShape:
            def part(self, index: int) -> FakePoint:
                return FakePoint(index)

        class FakePredictor:
            def __call__(self, _gray, _rect) -> FakeShape:
                return FakeShape()

        class FakeDetection:
            def __init__(self) -> None:
                self.rect = object()

        class FakeBatchDetector:
            def __init__(self) -> None:
                self.calls = []

            def __call__(self, images):
                self.calls.append(images)
                if isinstance(images, list):
                    return [[FakeDetection()] for _ in images]
                return [FakeDetection()]

        frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
        ]
        detector = FakeBatchDetector()

        class FakeCapture:
            def __init__(self, frames):
                self.frames = list(frames)
                self.released = False

            def isOpened(self):
                return not self.released

            def read(self):
                if not self.frames:
                    return False, None
                return True, self.frames.pop(0)

            def release(self):
                self.released = True

        fake_capture = FakeCapture(frames)

        with mock.patch("src.preprocess.mouth_roi.cv2.VideoCapture", return_value=fake_capture):
            landmarks = mouth_roi.detect_landmarks_for_video(
                Path("sample.mp4"),
                detector=None,
                cnn_detector=detector,
                predictor=FakePredictor(),
            )

        self.assertEqual(len(landmarks), 3)
        self.assertEqual(len(detector.calls), 1)
        self.assertIsInstance(detector.calls[0], list)

    def test_process_manifest_detect_streams_video_without_full_frame_load(self) -> None:
        root = self._temp_dir()
        raw_root = root / "raw"
        landmark_root = root / "landmarks"
        mouth_root = root / "mouth"
        raw_root.mkdir(parents=True)
        landmark_root.mkdir(parents=True)
        mouth_root.mkdir(parents=True)

        file_id = "sample"
        (raw_root / f"{file_id}.mp4").write_bytes(b"fake")
        manifest_path = root / "all.list"
        manifest_path.write_text(f"{file_id}\n", encoding="utf-8")
        mean_face_path = root / "mean_face.npy"
        np.save(mean_face_path, np.zeros((68, 2), dtype=np.float32))

        landmark = np.zeros((68, 2), dtype=np.float32)
        fake_dlib = mock.Mock()
        fake_dlib.shape_predictor.return_value = mock.Mock()

        with mock.patch("src.preprocess.mouth_roi._load_dependencies", return_value=fake_dlib):
            with mock.patch("src.preprocess.mouth_roi.build_cnn_detector", return_value=object()):
                with mock.patch(
                    "src.preprocess.mouth_roi.load_video_frames",
                    side_effect=AssertionError("detect stage should not load the full video into memory"),
                ):
                    with mock.patch(
                        "src.preprocess.mouth_roi.detect_landmarks_for_video",
                        return_value=[landmark],
                    ) as mock_detect_video:
                        summary = mouth_roi.process_manifest(
                            raw_video_root=raw_root,
                            manifest_path=manifest_path,
                            landmark_root=landmark_root,
                            mouth_roi_root=mouth_root,
                            face_predictor_path=root / "shape_predictor.dat",
                            mean_face_path=mean_face_path,
                            cnn_detector_path=root / "cnn_detector.dat",
                            stage="detect",
                            show_progress=False,
                        )

        self.assertEqual(summary["landmarks_written"], 1)
        mock_detect_video.assert_called_once()

    def test_process_manifest_align_skips_dlib_initialization(self) -> None:
        root = self._temp_dir()
        raw_root = root / "raw"
        landmark_root = root / "landmarks"
        mouth_root = root / "mouth"
        raw_root.mkdir(parents=True)
        landmark_root.mkdir(parents=True)
        mouth_root.mkdir(parents=True)

        file_id = "sample"
        (raw_root / f"{file_id}.mp4").write_bytes(b"fake")
        manifest_path = root / "all.list"
        manifest_path.write_text(f"{file_id}\n", encoding="utf-8")
        mean_face_path = root / "mean_face.npy"
        np.save(mean_face_path, np.zeros((68, 2), dtype=np.float32))

        landmark = np.zeros((68, 2), dtype=np.float32)
        with (landmark_root / f"{file_id}.pkl").open("wb") as handle:
            pickle.dump([landmark, landmark], handle)

        with mock.patch(
            "src.preprocess.mouth_roi._load_dependencies",
            side_effect=AssertionError("align stage should not initialize dlib"),
        ):
            with mock.patch(
                "src.preprocess.mouth_roi.load_video_frames",
                return_value=[np.zeros((8, 8, 3), dtype=np.uint8)],
            ):
                with mock.patch("src.preprocess.mouth_roi.interpolate_landmarks", return_value=[landmark, landmark]):
                    with mock.patch(
                        "src.preprocess.mouth_roi.crop_mouth_sequence",
                        return_value=[np.zeros((4, 4, 3), dtype=np.uint8)],
                    ):
                        with mock.patch("src.preprocess.mouth_roi.write_video"):
                            summary = mouth_roi.process_manifest(
                                raw_video_root=raw_root,
                                manifest_path=manifest_path,
                                landmark_root=landmark_root,
                                mouth_roi_root=mouth_root,
                                face_predictor_path=root / "shape_predictor.dat",
                                mean_face_path=mean_face_path,
                                cnn_detector_path=root / "cnn_detector.dat",
                                stage="align",
                                show_progress=False,
                            )

        self.assertEqual(summary["mouth_roi_written"], 1)

    def test_run_preprocess_from_config_align_allows_cpu_only_runtime(self) -> None:
        root = self._temp_dir()
        config_path = root / "config.yaml"
        split_dir = root / "splits"
        manifest_dir = root / "manifests"
        artifact_root = root / "artifacts"
        split_dir.mkdir(parents=True)
        artifact_root.mkdir(parents=True)

        config = {
            "paths": {
                "split_dir": str(split_dir),
                "manifest_dir": str(manifest_dir),
                "artifact_root": str(artifact_root),
                "raw_video_root": str(root / "raw"),
                "landmark_dir": str(root / "landmarks"),
                "mouth_roi_root": str(root / "mouth"),
                "face_predictor_path": str(root / "shape_predictor.dat"),
                "mean_face_path": str(root / "mean_face.npy"),
                "cnn_detector_path": str(root / "cnn_detector.dat"),
            },
            "preprocess": {
                "split_names": ["train"],
                "manifest_name": "all",
                "stage": "align",
                "crop_width": 96,
                "crop_height": 96,
                "start_idx": 48,
                "stop_idx": 68,
                "window_margin": 12,
                "fps": 25,
                "save_landmarks": True,
                "strict": True,
            },
            "runtime": {
                "devices": [],
                "workers_per_device": 1,
                "cpu_threads_per_worker": 2,
                "start_method": "spawn",
                "show_main_progress": False,
            },
            "logging": {
                "level": "INFO",
                "main_log_filename": "preprocess.log",
                "worker_log_filename_template": "preprocess_rank{rank}.log",
            },
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        def fake_build_manifests(*, output_dir, **_kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "all.list").write_text("sample\n", encoding="utf-8")

        shard_summary = {
            "stage": "align",
            "manifest": str(manifest_dir / "all.list"),
            "rank": 0,
            "nshard": 1,
            "requested_files": 1,
            "mouth_roi_written": 1,
            "landmarks_written": 0,
            "skipped_existing_mouth_roi": 0,
            "skipped_existing_landmarks": 0,
            "failed_missing_video": 0,
            "failed_read_video": 0,
            "failed_missing_landmarks": 0,
            "failed_no_landmarks": 0,
            "failed_crop": 0,
            "failed_files": [],
        }

        with mock.patch("src.preprocess.runtime.build_manifests", side_effect=fake_build_manifests):
            with mock.patch("src.preprocess.runtime.build_logger", return_value=mock.Mock()):
                with mock.patch("src.preprocess.runtime._run_preprocess_shard", return_value=shard_summary) as mock_run:
                    summary = runtime.run_preprocess_from_config(config_path)

        self.assertEqual(summary["num_procs"], 1)
        self.assertEqual(summary["devices"], [])
        self.assertIsNone(mock_run.call_args.kwargs["device_index"])


if __name__ == "__main__":
    unittest.main()
