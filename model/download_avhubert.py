import os
import time
from pathlib import Path

import requests


MODEL_NAME = "self_large_vox_433h.pt"
MODEL_URL = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt"
EXPECTED_SIZE = 5728780791
CHUNK_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 10
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def print_progress(current: int, total: int) -> None:
    percent = current / total * 100 if total else 0
    print(
        f"\r{percent:6.2f}%  {human_size(current)} / {human_size(total)}",
        end="",
        flush=True,
    )


root = Path(__file__).resolve().parent
model_dir = root / "model"
model_dir.mkdir(parents=True, exist_ok=True)

target_path = model_dir / MODEL_NAME
part_path = model_dir / f"{MODEL_NAME}.part"

if target_path.exists() and target_path.stat().st_size == EXPECTED_SIZE:
    print(f"already downloaded: {target_path}")
    raise SystemExit(0)

session = requests.Session()
session.headers.update({"User-Agent": "Audio-Video-Detection/avhubert-downloader"})

for attempt in range(1, MAX_RETRIES + 1):
    try:
        downloaded = part_path.stat().st_size if part_path.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        with session.get(
            MODEL_URL,
            headers=headers,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        ) as response:
            response.raise_for_status()

            if downloaded and response.status_code == 200:
                downloaded = 0
                if part_path.exists():
                    part_path.unlink()

            total = EXPECTED_SIZE
            mode = "ab" if downloaded else "wb"
            with part_path.open(mode) as handle:
                current = downloaded
                print(f"download attempt {attempt}/{MAX_RETRIES}: {MODEL_NAME}")
                print_progress(current, total)
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    current += len(chunk)
                    print_progress(current, total)
            print()

        final_size = part_path.stat().st_size
        if final_size != EXPECTED_SIZE:
            raise RuntimeError(
                f"size mismatch after download: got {final_size}, expected {EXPECTED_SIZE}"
            )

        os.replace(part_path, target_path)
        print(f"saved to: {target_path}")
        raise SystemExit(0)
    except Exception as exc:
        print(f"\nretry {attempt}/{MAX_RETRIES} failed: {exc}")
        if attempt == MAX_RETRIES:
            raise
        time.sleep(5)
