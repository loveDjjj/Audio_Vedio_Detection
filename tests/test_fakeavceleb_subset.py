from pathlib import Path
import shutil
import unittest
import uuid

from src.data.fakeavceleb_subset import (
    FakeAVCelebRecord,
    build_split_rows,
    load_fakeavceleb_records,
    sample_balanced_binary_records,
    split_records,
)


class FakeAVCelebSubsetTest(unittest.TestCase):
    def _write_metadata(self, root: Path) -> None:
        metadata = """source,target1,target2,method,category,type,race,gender,path,
id00001,-,-,real,A,RealVideo-RealAudio,African,men,00001.mp4,FakeAVCeleb/RealVideo-RealAudio/African/men/id00001
id00002,-,-,real,A,RealVideo-RealAudio,African,men,00002.mp4,FakeAVCeleb/RealVideo-RealAudio/African/men/id00002
id00003,id10001,4,wav2lip,D,FakeVideo-FakeAudio,African,men,00003_4_id10001_wavtolip.mp4,FakeAVCeleb/FakeVideo-FakeAudio/African/men/id00003
id00004,id10002,8,fsgan-wav2lip,D,FakeVideo-FakeAudio,African,men,00004_8_id10002_fsgan-wav2lip.mp4,FakeAVCeleb/FakeVideo-FakeAudio/African/men/id00004
id00005,id10003,-,faceswap-wav2lip,D,FakeVideo-FakeAudio,African,women,00005_id10003_faceswap-wav2lip.mp4,FakeAVCeleb/FakeVideo-FakeAudio/African/women/id00005
id99999,-,-,rtvc,B,RealVideo-FakeAudio,African,men,ignore.mp4,FakeAVCeleb/RealVideo-FakeAudio/African/men/id99999
"""
        (root / "meta_data.csv").write_text(metadata, encoding="utf-8")

    def _touch_video(self, root: Path, relative_path: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")

    def test_load_fakeavceleb_records_keeps_only_a_and_d(self) -> None:
        tmp_root = Path(".tmp_unittest")
        tmp_root.mkdir(exist_ok=True)
        root = tmp_root / f"fakeavceleb_{uuid.uuid4().hex}"
        try:
            root.mkdir(parents=True, exist_ok=False)
            self._write_metadata(root)
            self._touch_video(root, "RealVideo-RealAudio/African/men/id00001/00001.mp4")
            self._touch_video(root, "RealVideo-RealAudio/African/men/id00002/00002.mp4")
            self._touch_video(root, "FakeVideo-FakeAudio/African/men/id00003/00003_4_id10001_wavtolip.mp4")
            self._touch_video(root, "FakeVideo-FakeAudio/African/men/id00004/00004_8_id10002_fsgan-wav2lip.mp4")
            self._touch_video(root, "FakeVideo-FakeAudio/African/women/id00005/00005_id10003_faceswap-wav2lip.mp4")

            records = load_fakeavceleb_records(root)
        finally:
            if root.exists():
                shutil.rmtree(root)

        self.assertEqual(len(records), 5)
        self.assertEqual(
            {record.type for record in records},
            {"RealVideo-RealAudio", "FakeVideo-FakeAudio"},
        )
        self.assertEqual(
            records[0].relative_path,
            "FakeVideo-FakeAudio/African/men/id00003/00003_4_id10001_wavtolip.mp4",
        )
        self.assertEqual(
            records[-1].relative_path,
            "RealVideo-RealAudio/African/men/id00002/00002.mp4",
        )

    def test_balanced_sampling_is_deterministic_and_label_balanced(self) -> None:
        records = [
            FakeAVCelebRecord(
                relative_path=f"RealVideo-RealAudio/African/men/id{i:05d}/{i:05d}.mp4",
                label=0,
                label_name="real",
                source=f"id{i:05d}",
                target1="-",
                target2="-",
                method="real",
                category="A",
                type="RealVideo-RealAudio",
                race="African",
                gender="men",
                filename=f"{i:05d}.mp4",
            )
            for i in range(5)
        ]
        records.extend(
            FakeAVCelebRecord(
                relative_path=f"FakeVideo-FakeAudio/African/men/id{i:05d}/{i:05d}_x_id9{i:04d}_{method}.mp4",
                label=1,
                label_name="fake",
                source=f"id{i:05d}",
                target1=f"id9{i:04d}",
                target2="-",
                method=method,
                category="D",
                type="FakeVideo-FakeAudio",
                race="African",
                gender="men",
                filename=f"{i:05d}_x_id9{i:04d}_{method}.mp4",
            )
            for i, method in enumerate(
                [
                    "wav2lip",
                    "wav2lip",
                    "fsgan-wav2lip",
                    "faceswap-wav2lip",
                    "wav2lip",
                    "fsgan-wav2lip",
                    "wav2lip",
                    "faceswap-wav2lip",
                ],
                start=10,
            )
        )

        selected_a = sample_balanced_binary_records(records, seed=7)
        selected_b = sample_balanced_binary_records(records, seed=7)

        self.assertEqual(
            [row.relative_path for row in selected_a],
            [row.relative_path for row in selected_b],
        )
        self.assertEqual(sum(row.label == 0 for row in selected_a), 5)
        self.assertEqual(sum(row.label == 1 for row in selected_a), 5)

    def test_split_records_produces_expected_sizes(self) -> None:
        records = [
            FakeAVCelebRecord(
                relative_path=f"RealVideo-RealAudio/African/men/id{i:05d}/{i:05d}.mp4",
                label=0 if i < 5 else 1,
                label_name="real" if i < 5 else "fake",
                source=f"id{i:05d}",
                target1="-",
                target2="-",
                method="real" if i < 5 else "wav2lip",
                category="A" if i < 5 else "D",
                type="RealVideo-RealAudio" if i < 5 else "FakeVideo-FakeAudio",
                race="African",
                gender="men",
                filename=f"{i:05d}.mp4",
            )
            for i in range(10)
        ]

        split_map = split_records(records, seed=42)

        self.assertEqual(
            {key: len(value) for key, value in split_map.items()},
            {"train": 8, "val": 1, "test": 1},
        )
        self.assertEqual(sum(len(value) for value in split_map.values()), 10)

    def test_build_split_rows_keeps_required_columns(self) -> None:
        rows = build_split_rows(
            [
                FakeAVCelebRecord(
                    relative_path="FakeVideo-FakeAudio/African/men/id00003/00003_4_id10001_wavtolip.mp4",
                    label=1,
                    label_name="fake",
                    source="id00003",
                    target1="id10001",
                    target2="4",
                    method="wav2lip",
                    category="D",
                    type="FakeVideo-FakeAudio",
                    race="African",
                    gender="men",
                    filename="00003_4_id10001_wavtolip.mp4",
                )
            ]
        )

        self.assertEqual(
            rows[0]["relative_path"],
            "FakeVideo-FakeAudio/African/men/id00003/00003_4_id10001_wavtolip.mp4",
        )
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[0]["method"], "wav2lip")


if __name__ == "__main__":
    unittest.main()
