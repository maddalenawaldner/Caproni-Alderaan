#!/usr/bin/env python3
"""Batch image tagging with RAM (Recognize Anything), mirroring extract_album_text_vllm.py.

Recursively scans ``--input-dir`` for images, runs the tagging model once per image,
and writes UTF-8 ``.txt`` files under ``--output-dir`` mirroring the directory layout
(relative paths; only the extension changes). Each file contains a comma-separated
tag list, same style as ``pytorch-workshop/examples/image_tagging.py`` single-image mode.

Model loading uses pytorch-workshop ``get_model("ram", ...)`` (see
``pytorch-workshop/src/models/_tagging/_ram.py``). Add that tree to ``PYTHONPATH`` or use
``scripts/run_tag_album_ram.sh``.

Defaults for ``--input-dir`` / ``--output-dir`` match ``extract_album_text_vllm.py``
(``<repo>/datasets/Album-Trento JPG`` and ``<repo>/outputs``).

Requires: pytorch-workshop dependencies (torch, ram, pillow, …); Python version should
match the workshop (see ``pytorch-workshop/pyproject.toml``).
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKSHOP_ROOT = _REPO_ROOT / "pytorch-workshop"

if _WORKSHOP_ROOT.is_dir():
    workshop = str(_WORKSHOP_ROOT)
    if workshop not in sys.path:
        sys.path.insert(0, workshop)
else:
    print(
        f"error: pytorch-workshop not found at {_WORKSHOP_ROOT}; "
        "clone it next to this repo or set PYTHONPATH.",
        file=sys.stderr,
    )
    raise SystemExit(1)

# Register RAM only (avoid importing every workshop model stack).
import src.models._tagging  # noqa: F401, E402
from src.models._api import get_model  # noqa: E402
from src.models._tagging import DEFAULT_TAGGING_MODEL, TAGGING_MODELS  # noqa: E402

import torch  # noqa: E402

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_INPUT_DIR = _REPO_ROOT / "datasets" / "Album-Trento JPG"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs_ram"


def collect_images(root: Path) -> list[Path]:
    root = root.resolve()
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path)
    found.sort()
    return found


def output_txt_path(input_root: Path, output_root: Path, image_path: Path) -> Path:
    rel = image_path.resolve().relative_to(input_root.resolve())
    return (output_root / rel).with_suffix(".txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tag images with RAM (Recognize Anything), mirrored .txt outputs."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Root folder to scan recursively (default: {DEFAULT_INPUT_DIR})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root for mirrored .txt outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TAGGING_MODEL,
        choices=list(TAGGING_MODELS.keys()),
        help=f"Tagging model id (default: {DEFAULT_TAGGING_MODEL})",
    )
    p.add_argument(
        "--device",
        type=str,
        default=_DEFAULT_DEVICE,
        choices=["cpu", "cuda", "mps"],
        help=f"Torch device for RAM (default: {_DEFAULT_DEVICE} if CUDA available, else cpu).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="RAM input image size (default: 384).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many images to process between progress logs and gc.collect() (default: 8). "
        "RAM runs one image per forward pass.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output .txt already exists.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    if not input_root.is_dir():
        print(f"error: input dir does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 1

    images = collect_images(input_root)
    if not images:
        print(f"no images found under {input_root}", file=sys.stderr)
        return 1

    print(f"loading model {args.model!r} on {args.device!r} …", flush=True)
    model = get_model(args.model, device=args.device, image_size=args.image_size)

    total = len(images)
    done = 0
    skipped_existing = 0
    skipped_read = 0

    for start in range(0, total, args.batch_size):
        batch_paths = images[start : start + args.batch_size]

        for img_path in batch_paths:
            out_path = output_txt_path(input_root, output_root, img_path)
            if args.skip_existing and out_path.exists():
                skipped_existing += 1
                continue

            if not img_path.is_file():
                print(f"warn: skip (not a file) {img_path}", file=sys.stderr)
                skipped_read += 1
                continue

            try:
                tags = model(str(img_path.resolve()))
            except OSError as e:
                print(f"warn: skip (cannot read image) {img_path}: {e}", file=sys.stderr)
                skipped_read += 1
                continue
            except Exception as e:
                print(f"error: RAM failed for {img_path}: {e}", file=sys.stderr)
                raise

            if not isinstance(tags, list):
                text = str(tags).strip()
            else:
                text = ", ".join(tags).strip()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
            done += 1

        gc.collect()
        print(
            f"progress: {done} written, {skipped_existing} skip-existing, "
            f"{skipped_read} unreadable, {total} images",
            flush=True,
        )

    print(
        f"done: {done} files written, {skipped_existing} skip-existing, "
        f"{skipped_read} unreadable, output root {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
