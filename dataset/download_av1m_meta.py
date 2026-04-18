import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "ControlNet/AV-Deepfake1M"
ROOT = Path("/data/OneDay/AV-Deepfake1M")
VAL_DIR = ROOT / "val"
MAX_RETRIES = 10
RETRY_SLEEP_SECONDS = 5

FILES = [
    "README.md",
    "TERMS_AND_CONDITIONS.md",
    "val_metadata.json",
    *[f"val/val.zip.{i:03d}" for i in range(1, 21)],
]


def download_file(filename: str) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[download {attempt}/{MAX_RETRIES}] {filename}")
            path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=str(ROOT),
            )
            print(f"[done] {path}")
            return
        except Exception as exc:
            print(f"[retry] {filename}: {exc}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SECONDS)


def find_7z() -> str | None:
    return (
        shutil.which("7z")
        or shutil.which("7za")
        or shutil.which("7zr")
        or (str(Path("C:/Program Files/7-Zip/7z.exe")) if Path("C:/Program Files/7-Zip/7z.exe").exists() else None)
    )


ROOT.mkdir(parents=True, exist_ok=True)

for file in FILES:
    download_file(file)

tool = find_7z()
if tool is None:
    raise SystemExit("7z is required to extract val.zip.001-020. Please install 7-Zip first.")

already_extracted = any(
    item.name.startswith("id") or item.suffix == ".mp4"
    for item in VAL_DIR.iterdir()
    if not item.name.startswith("val.zip.")
)

if not already_extracted:
    subprocess.run([tool, "x", "val/val.zip.001", "-y"], cwd=ROOT, check=True)
