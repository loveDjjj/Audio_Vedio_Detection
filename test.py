import csv
from collections import Counter
from pathlib import Path

root = Path("/data/OneDay/MAVOS-DD")
missing = Counter()
present = Counter()

for split in ("train", "val", "test"):
      with open(Path("splits/mavos_dd_real_fullfake") / f"{split}.csv", newline="") as f:
          for r in csv.DictReader(f):
              key = (r["label_name"], r["audio_fake"], r["video_fake"])
              if (root / r["relative_path"]).exists():
                  present[key] += 1
              else:
                  missing[key] += 1

print("present:", present)
print("missing:", missing)
