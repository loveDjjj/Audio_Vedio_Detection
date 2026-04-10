from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from skimage import transform as tf
from tqdm import tqdm


STABLE_POINT_IDS = [33, 36, 39, 42, 45]
STD_SIZE = (256, 256)


def _load_dependencies():
    try:
        import dlib
    except ImportError as exc:
        raise RuntimeError(
            "dlib is required for mouth ROI preprocessing. "
            "Install it before running preprocess_av1m_mouth_roi.py."
        ) from exc
    return dlib


def build_cnn_detector(dlib, cnn_detector_path: Path):
    if not getattr(dlib, "DLIB_USE_CUDA", False):
        raise RuntimeError(
            "Strict CNN preprocessing requires a CUDA-enabled dlib build, "
            "but `dlib.DLIB_USE_CUDA` is false."
        )
    if dlib.cuda.get_num_devices() < 1:
        raise RuntimeError(
            "Strict CNN preprocessing requires at least one CUDA device visible to dlib."
        )
    return dlib.cnn_face_detection_model_v1(str(cnn_detector_path))


def read_manifest_ids(manifest_path: Path) -> list[str]:
    return [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shard_items(items: list[str], rank: int, nshard: int) -> list[str]:
    if nshard < 1:
        raise ValueError("nshard must be >= 1")
    if rank < 0 or rank >= nshard:
        raise ValueError(f"rank must be in [0, {nshard - 1}], got {rank}")
    num_per_shard = math.ceil(len(items) / nshard)
    start = num_per_shard * rank
    end = num_per_shard * (rank + 1)
    return items[start:end]


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"Unable to read frames from {video_path}")
    return frames


def detect_landmarks_for_frame(frame: np.ndarray, detector, cnn_detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cnn_detector is None:
        raise RuntimeError("Strict CNN preprocessing requires `cnn_detector` to be initialized.")
    rects = [result.rect for result in cnn_detector(gray)]
    if len(rects) == 0:
        return None

    shape = predictor(gray, rects[0])
    coords = np.zeros((68, 2), dtype=np.float32)
    for index in range(68):
        coords[index] = (shape.part(index).x, shape.part(index).y)
    return coords


def detect_landmarks_for_video(video_path: Path, detector, cnn_detector, predictor) -> list[np.ndarray | None]:
    frames = load_video_frames(video_path)
    return [detect_landmarks_for_frame(frame, detector, cnn_detector, predictor) for frame in frames]


def linear_interpolate(landmarks: list[np.ndarray | None], start_idx: int, stop_idx: int):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    if start_landmarks is None or stop_landmarks is None:
        return landmarks
    delta = stop_landmarks - start_landmarks
    for index in range(1, stop_idx - start_idx):
        landmarks[start_idx + index] = start_landmarks + index / float(stop_idx - start_idx) * delta
    return landmarks


def interpolate_landmarks(landmarks: list[np.ndarray | None]) -> list[np.ndarray] | None:
    valid_indices = [index for index, landmark in enumerate(landmarks) if landmark is not None]
    if not valid_indices:
        return None

    for index in range(1, len(valid_indices)):
        current = valid_indices[index]
        previous = valid_indices[index - 1]
        if current - previous > 1:
            landmarks = linear_interpolate(landmarks, previous, current)

    valid_indices = [index for index, landmark in enumerate(landmarks) if landmark is not None]
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    landmarks[:first_valid] = [landmarks[first_valid]] * first_valid
    landmarks[last_valid:] = [landmarks[last_valid]] * (len(landmarks) - last_valid)

    if any(landmark is None for landmark in landmarks):
        raise ValueError("Landmark interpolation failed to fill every frame")
    return [landmark.astype(np.float32) for landmark in landmarks]


def warp_img(src: np.ndarray, dst: np.ndarray, image: np.ndarray, output_shape: tuple[int, int]):
    transform = tf.estimate_transform("similarity", src, dst)
    warped = tf.warp(image, inverse_map=transform.inverse, output_shape=output_shape)
    warped = (warped * 255).astype("uint8")
    return warped, transform


def apply_transform(transform, image: np.ndarray, output_shape: tuple[int, int]):
    warped = tf.warp(image, inverse_map=transform.inverse, output_shape=output_shape)
    warped = (warped * 255).astype("uint8")
    return warped


def cut_patch(image: np.ndarray, landmarks: np.ndarray, height: int, width: int, threshold: int = 5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < -threshold:
        raise ValueError("too much bias in height")
    if center_x - width < 0:
        center_x = width
    if center_x - width < -threshold:
        raise ValueError("too much bias in width")

    if center_y + height > image.shape[0]:
        center_y = image.shape[0] - height
    if center_y + height > image.shape[0] + threshold:
        raise ValueError("too much bias in height")
    if center_x + width > image.shape[1]:
        center_x = image.shape[1] - width
    if center_x + width > image.shape[1] + threshold:
        raise ValueError("too much bias in width")

    return np.copy(
        image[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )


def crop_mouth_sequence(
    frames: list[np.ndarray],
    landmarks: list[np.ndarray],
    mean_face_landmarks: np.ndarray,
    crop_width: int,
    crop_height: int,
    start_idx: int,
    stop_idx: int,
    window_margin: int,
) -> list[np.ndarray]:
    margin = min(len(frames), window_margin)
    queue_frames: deque[np.ndarray] = deque()
    queue_landmarks: deque[np.ndarray] = deque()
    sequence: list[np.ndarray] = []
    transform = None

    for frame_index, frame in enumerate(frames):
        if frame_index >= len(landmarks):
            break
        queue_frames.append(frame)
        queue_landmarks.append(landmarks[frame_index])

        if len(queue_frames) == margin:
            smoothed_landmarks = np.mean(queue_landmarks, axis=0)
            current_landmarks = queue_landmarks.popleft()
            current_frame = queue_frames.popleft()
            transformed_frame, transform = warp_img(
                smoothed_landmarks[STABLE_POINT_IDS, :],
                mean_face_landmarks[STABLE_POINT_IDS, :],
                current_frame,
                STD_SIZE,
            )
            transformed_landmarks = transform(current_landmarks)
            sequence.append(
                cut_patch(
                    transformed_frame,
                    transformed_landmarks[start_idx:stop_idx],
                    crop_height // 2,
                    crop_width // 2,
                )
            )

        if frame_index == len(landmarks) - 1 and transform is not None:
            while queue_frames:
                current_frame = queue_frames.popleft()
                current_landmarks = queue_landmarks.popleft()
                transformed_frame = apply_transform(transform, current_frame, STD_SIZE)
                transformed_landmarks = transform(current_landmarks)
                sequence.append(
                    cut_patch(
                        transformed_frame,
                        transformed_landmarks[start_idx:stop_idx],
                        crop_height // 2,
                        crop_width // 2,
                    )
                )
    return sequence


def write_video(frames: list[np.ndarray], target_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open VideoWriter for {target_path}")

    try:
        for frame in frames:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()


def process_manifest(
    raw_video_root: Path,
    manifest_path: Path,
    landmark_root: Path,
    mouth_roi_root: Path,
    face_predictor_path: Path,
    mean_face_path: Path,
    cnn_detector_path: Path | None = None,
    rank: int = 0,
    nshard: int = 1,
    crop_width: int = 96,
    crop_height: int = 96,
    start_idx: int = 48,
    stop_idx: int = 68,
    window_margin: int = 12,
    fps: int = 25,
    stage: str = "all",
    save_landmarks: bool = True,
    strict: bool = True,
    show_progress: bool = True,
    progress_callback=None,
) -> dict:
    if stage not in {"all", "detect", "align"}:
        raise ValueError(f"Unsupported stage: {stage}")

    dlib = _load_dependencies()
    file_ids = shard_items(read_manifest_ids(manifest_path), rank=rank, nshard=nshard)
    mean_face_landmarks = np.load(mean_face_path)

    detector = None
    if cnn_detector_path is None:
        raise RuntimeError(
            "Strict preprocessing requires `resources/dlib/mmod_human_face_detector.dat` "
            "and does not fall back to the CPU frontal-face detector."
        )
    cnn_detector = build_cnn_detector(dlib, cnn_detector_path)
    predictor = dlib.shape_predictor(str(face_predictor_path))

    summary = {
        "stage": stage,
        "manifest": str(manifest_path),
        "rank": rank,
        "nshard": nshard,
        "requested_files": len(file_ids),
        "mouth_roi_written": 0,
        "landmarks_written": 0,
        "skipped_existing_mouth_roi": 0,
        "skipped_existing_landmarks": 0,
        "failed_missing_video": 0,
        "failed_missing_landmarks": 0,
        "failed_no_landmarks": 0,
        "failed_crop": 0,
        "failed_files": [],
    }

    for file_id in tqdm(file_ids, desc=f"{stage}:{manifest_path.stem}:r{rank}", disable=not show_progress):
        raw_video_path = raw_video_root / f"{file_id}.mp4"
        landmark_path = landmark_root / f"{file_id}.pkl"
        mouth_roi_path = mouth_roi_root / f"{file_id}.mp4"

        if not raw_video_path.exists():
            summary["failed_missing_video"] += 1
            summary["failed_files"].append({"file_id": file_id, "reason": "missing_video"})
            if progress_callback is not None:
                progress_callback(file_id)
            continue

        if stage == "detect":
            if landmark_path.exists():
                summary["skipped_existing_landmarks"] += 1
                if progress_callback is not None:
                    progress_callback(file_id)
                continue
            landmarks = detect_landmarks_for_video(raw_video_path, detector, cnn_detector, predictor)
            landmark_path.parent.mkdir(parents=True, exist_ok=True)
            with landmark_path.open("wb") as handle:
                pickle.dump(landmarks, handle)
            summary["landmarks_written"] += 1
            if progress_callback is not None:
                progress_callback(file_id)
            continue

        if mouth_roi_path.exists():
            summary["skipped_existing_mouth_roi"] += 1
            if progress_callback is not None:
                progress_callback(file_id)
            continue

        if landmark_path.exists():
            with landmark_path.open("rb") as handle:
                landmarks = pickle.load(handle)
        elif stage == "all":
            landmarks = detect_landmarks_for_video(raw_video_path, detector, cnn_detector, predictor)
            if save_landmarks:
                landmark_path.parent.mkdir(parents=True, exist_ok=True)
                with landmark_path.open("wb") as handle:
                    pickle.dump(landmarks, handle)
                summary["landmarks_written"] += 1
        else:
            summary["failed_missing_landmarks"] += 1
            summary["failed_files"].append({"file_id": file_id, "reason": "missing_landmark_file"})
            if progress_callback is not None:
                progress_callback(file_id)
            continue

        try:
            interpolated_landmarks = interpolate_landmarks(landmarks)
        except Exception:
            interpolated_landmarks = None
        if interpolated_landmarks is None:
            summary["failed_no_landmarks"] += 1
            summary["failed_files"].append({"file_id": file_id, "reason": "no_landmarks"})
            if progress_callback is not None:
                progress_callback(file_id)
            continue

        try:
            frames = load_video_frames(raw_video_path)
            cropped_frames = crop_mouth_sequence(
                frames=frames,
                landmarks=interpolated_landmarks,
                mean_face_landmarks=mean_face_landmarks,
                crop_width=crop_width,
                crop_height=crop_height,
                start_idx=start_idx,
                stop_idx=stop_idx,
                window_margin=window_margin,
            )
            if not cropped_frames:
                raise ValueError("empty_cropped_frames")
            write_video(cropped_frames, mouth_roi_path, fps=fps)
            summary["mouth_roi_written"] += 1
        except Exception as exc:
            summary["failed_crop"] += 1
            summary["failed_files"].append({"file_id": file_id, "reason": f"crop_failed:{exc}"})
            if not strict:
                if progress_callback is not None:
                    progress_callback(file_id)
                continue
        if progress_callback is not None:
            progress_callback(file_id)

    return summary
