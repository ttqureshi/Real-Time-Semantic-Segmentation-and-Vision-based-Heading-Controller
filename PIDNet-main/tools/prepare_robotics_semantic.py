#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


LABEL_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 255,
    7: 255,
}

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the robotics semantic dataset for PIDNet."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("../analysis/semantic_dataset"),
        help="Directory containing the generated split text files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/robotics_semantic"),
        help="Target dataset directory inside the PIDNet repo",
    )
    parser.add_argument(
        "--list-root",
        type=Path,
        default=Path("data/list/robotics_semantic"),
        help="Directory for PIDNet list files",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously prepared data before exporting",
    )
    return parser.parse_args()


def remap_mask(mask_path: Path, target_path: Path) -> None:
    mask = np.array(Image.open(mask_path).convert("L"))
    remapped = np.ones_like(mask, dtype=np.uint8) * 255
    for raw_label, train_label in LABEL_MAPPING.items():
        remapped[mask == raw_label] = train_label
    Image.fromarray(remapped).save(target_path)


def copy_image(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    shutil.copy2(source, target)


def prepare_split(split_name: str, split_file: Path, output_root: Path, list_root: Path) -> dict[str, object]:
    images_dir = output_root / "images" / split_name
    labels_dir = output_root / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    list_root.mkdir(parents=True, exist_ok=True)

    list_lines = []
    frame_ids = []
    with split_file.open(encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            match = re.match(r"^(.*\.jpg)\s+(.*_mask\.png)$", raw_line)
            if match is None:
                raise ValueError(f"Could not parse split entry: {raw_line}")
            image_src, mask_src = match.groups()
            image_src = Path(image_src)
            mask_src = Path(mask_src)
            if not image_src.is_absolute():
                image_src = WORKSPACE_ROOT / image_src
            if not mask_src.is_absolute():
                mask_src = WORKSPACE_ROOT / mask_src
            frame_id = image_src.stem.split("_png")[0]
            frame_ids.append(frame_id)
            image_target = images_dir / f"{frame_id}.jpg"
            mask_target = labels_dir / f"{frame_id}.png"
            copy_image(image_src, image_target)
            remap_mask(mask_src, mask_target)
            rel_image = image_target.relative_to(output_root.parent).as_posix()
            rel_mask = mask_target.relative_to(output_root.parent).as_posix()
            list_lines.append(f"{rel_image} {rel_mask}\n")

    with (list_root / f"{split_name}.lst").open("w", encoding="utf-8") as handle:
        handle.writelines(list_lines)

    return {
        "count": len(frame_ids),
        "frames": frame_ids,
    }


def write_manifest(output_root: Path, summary: dict[str, object]) -> None:
    manifest_path = output_root / "dataset_summary.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    if args.clean and args.list_root.exists():
        shutil.rmtree(args.list_root)

    summary = {"label_mapping": LABEL_MAPPING, "splits": {}}
    split_dir = args.analysis_dir / "splits"
    for split_name in ("train", "val", "test"):
        split_file = split_dir / f"{split_name}.txt"
        summary["splits"][split_name] = prepare_split(
            split_name=split_name,
            split_file=split_file,
            output_root=args.output_root,
            list_root=args.list_root,
        )

    write_manifest(args.output_root, summary)
    print("Prepared robotics semantic dataset for PIDNet.")
    print(f"Dataset root: {args.output_root.resolve()}")
    print(f"List root: {args.list_root.resolve()}")
    for split_name, split_summary in summary["splits"].items():
        print(f"{split_name}: {split_summary['count']} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
