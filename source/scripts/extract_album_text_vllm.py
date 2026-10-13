from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Resolve repo root from this file (…/caproni/scripts/ → …/caproni/) so Docker (/work) and host paths both work.
_REPO_ROOT = Path("/data/caproni/maddalena_waldner/workspace/") 
DEFAULT_INPUT_DIR = _REPO_ROOT / "data" / "Album Trento JPG_split"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "Album Trento JPG_split"

DEFAULT_OCR_INSTRUCTION = (
    "Transcribe all visible text in this image. Preserve reading order and line breaks. "
    "If there is no text, respond with exactly: NO_TEXT."
)


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


def build_qwen3_vl_moe_prompt(user_question: str) -> str:
    """Chat + vision placeholders matching vLLM ``run_qwen3_vl_moe`` (image modality)."""
    placeholder = "<|image_pad|>"
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{user_question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract text from images with vLLM + Qwen3-VL-MoE (offline)."
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
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1). Increase for large MoE on multi-GPU.",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max_model_len (default: 4096).",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="vLLM max_num_seqs (default: 4). Lower if OOM.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="How many images to load and submit per generate() call (default: 4).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Sampling max_tokens (default: 2048).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help='Model dtype, e.g. "auto", "float16", "bfloat16" (default: auto).',
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization (default: 0.9).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM when required by the checkpoint.",
    )
    p.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallelism (MoE); useful on multi-GPU when supported.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output .txt already exists.",
    )
    p.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_OCR_INSTRUCTION,
        help="User instruction appended after the vision placeholder.",
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

    prompt_template = build_qwen3_vl_moe_prompt(args.instruction)

    from vllm import LLM, SamplingParams

    llm_kwargs: dict = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "limit_mm_per_prompt": {"image": 1},
        "mm_processor_kwargs": {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.dtype and args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True
    if args.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True

    print(f"loading model {args.model!r} …", flush=True)
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    total = len(images)
    done = 0
    skipped_existing = 0
    skipped_read = 0

    for start in range(0, total, args.batch_size):
        batch_paths = images[start : start + args.batch_size]
        inputs: list[dict] = []
        batch_targets: list[Path] = []

        for img_path in batch_paths:
            out_path = output_txt_path(input_root, output_root, img_path)
            if args.skip_existing and out_path.exists():
                skipped_existing += 1
                continue
            try:
                pil = Image.open(img_path).convert("RGB")
            except OSError as e:
                print(f"warn: skip (cannot read image) {img_path}: {e}", file=sys.stderr)
                skipped_read += 1
                continue

            inputs.append(
                {
                    "prompt": prompt_template,
                    "multi_modal_data": {"image": pil},
                }
            )
            batch_targets.append(out_path)

        if not inputs:
            continue

        try:
            outputs = llm.generate(inputs, sampling_params=sampling)
        except Exception as e:
            print(f"error: vLLM generate failed for batch starting at {start}: {e}", file=sys.stderr)
            for inp in inputs:
                pil = inp.get("multi_modal_data", {}).get("image")
                if pil is not None and hasattr(pil, "close"):
                    pil.close()
            raise

        if len(outputs) != len(batch_targets):
            print(
                f"error: expected {len(batch_targets)} outputs, got {len(outputs)}",
                file=sys.stderr,
            )
            return 1

        for req_out, out_path, inp in zip(outputs, batch_targets, inputs, strict=True):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            text = req_out.outputs[0].text.strip()
            out_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
            pil = inp["multi_modal_data"]["image"]
            if hasattr(pil, "close"):
                pil.close()
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
