import asyncio
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from src.models import DEFAULT_TAGGING_MODEL, TAGGING_MODELS, get_model

try:
    import py360convert

    PANORAMIC_SUPPORT = True
except ImportError:
    PANORAMIC_SUPPORT = False


def convert_panorama_to_cubemap(
    panorama_path: str, face_size: int = 512
) -> dict[str, Image.Image]:
    """Convert an equirectangular panorama to cubemap faces.

    Args:
    ----
        panorama_path (str): Path to the panoramic image.
        face_size (int): Size of each cubemap face. Defaults to 512.

    Returns:
    -------
        dict[str, Image.Image]: Dictionary mapping face names to PIL Images.

    """
    if not PANORAMIC_SUPPORT:
        raise ImportError(
            "Panoramic support requires py360convert. Install it with: pip install py360convert"
        )

    # Load panoramic image and convert to numpy array
    panorama = Image.open(panorama_path)
    panorama_np = np.array(panorama)
    panorama_array = py360convert.e2c(
        panorama_np, face_w=face_size, mode="bilinear", cube_format="dict"
    )

    # Convert face names and arrays to PIL Images
    faces = {}
    face_mapping = {
        "F": "front",
        "R": "right",
        "B": "back",
        "L": "left",
        "U": "top",
        "D": "bottom",
    }

    for key, name in face_mapping.items():
        face_array = panorama_array[key]
        faces[name] = Image.fromarray(face_array)

    return faces


async def run_serverless_tagging_model(args: Namespace) -> None:
    """Run a tagging model in serverless mode.

    Args:
    ----
        args (Namespace): The parsed arguments.

    """
    if args.model not in TAGGING_MODELS:
        avail_models = list(TAGGING_MODELS.keys())
        raise ValueError(
            f"{args.model} not available. Available models are: {', '.join(avail_models)}"
        )
    model = get_model(args.model, device=args.device)

    # Handle panoramic images
    if args.panoramic:
        if not PANORAMIC_SUPPORT:
            raise ImportError(
                "Panoramic support requires py360convert and Pillow."
                " Install them with: `uv pip install py360convert Pillow`"
            )

        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] Processing panoramic image...")
        print(f"Converting panorama to cubemap faces (size: {args.face_size}x{args.face_size})...")

        # Convert panorama to cubemap faces
        faces = convert_panorama_to_cubemap(args.input, face_size=args.face_size)

        # Create output directory with timestamp if saving views
        output_dir = None
        if args.save_views:
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            base_output = args.output_dir if args.output_dir else "output"
            output_dir = Path(base_output) / f"panorama_tagging_{timestamp}"
            faces_dir = output_dir / "faces"
            faces_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving extracted faces to '{faces_dir}/'...")
            for face_name, face_image in faces.items():
                face_path = faces_dir / f"{face_name}.jpg"
                face_image.save(face_path)
                print(f"  - Saved {face_name} face to {face_path}")

        # Process each face and collect tags
        all_tags = []
        face_tags = {}

        for face_name, face_image in faces.items():
            # Save face temporarily for processing
            temp_face_path = f"temp_{face_name}.jpg"
            face_image.save(temp_face_path)

            try:
                print(f"Tagging {face_name} face...")
                user_input = {"messages": [("user", temp_face_path)]}
                output = await model.ainvoke(user_input)
                _, message = output["messages"][-1]
                tags = message.split(", ")
                all_tags.extend(tags)
                face_tags[face_name] = tags
            finally:
                # Clean up temporary file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

        # Calculate statistics
        from collections import Counter

        unique_tags = sorted(set(all_tags))
        tag_counts = Counter(all_tags)
        face_unique_tags = {name: set(tags) for name, tags in face_tags.items()}

        # Display results
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("PANORAMIC IMAGE TAGGING RESULTS")
        print("=" * 60)
        print(f"Processing completed in {processing_time:.2f} seconds")

        print("\nTags by Face:")
        for face_name, tags in face_tags.items():
            print(f"  {face_name.capitalize()}: {', '.join(tags)}")

        print(f"\nGlobal Unique Tags ({len(unique_tags)}):")
        print(", ".join(unique_tags))

        print("\nPer-Face Unique Tag Counts:")
        for face_name in face_tags:
            unique_count = len(face_unique_tags[face_name])
            print(f"  {face_name.capitalize()}: {unique_count} unique tags")

        # Show tag frequency if verbose
        if args.verbose:
            print("\nTag Frequency (appearing in multiple faces):")
            for tag, count in tag_counts.most_common():
                if count > 1:
                    print(f"  {tag}: {count} faces")

        # Save results to file if output directory exists
        if output_dir:
            results_file = output_dir / "results.txt"
            with open(results_file, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("PANORAMIC IMAGE TAGGING RESULTS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Input: {args.input}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Face size: {args.face_size}x{args.face_size}\n")
                f.write(f"Processing time: {processing_time:.2f} seconds\n")
                f.write(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("Tags by Face:\n")
                for face_name, tags in face_tags.items():
                    f.write(f"  {face_name.capitalize()}: {', '.join(tags)}\n")

                f.write(f"\nGlobal Unique Tags ({len(unique_tags)}):\n")
                f.write(", ".join(unique_tags) + "\n")

                f.write("\nPer-Face Unique Tag Counts:\n")
                for face_name in face_tags:
                    unique_count = len(face_unique_tags[face_name])
                    f.write(f"  {face_name.capitalize()}: {unique_count} unique tags\n")

                f.write("\nTag Frequency (appearing in multiple faces):\n")
                for tag, count in tag_counts.most_common():
                    if count > 1:
                        f.write(f"  {tag}: {count} faces\n")

            print(f"\n✓ Results saved to '{results_file}'")

    else:
        # Original single image tagging
        user_input = args.input
        user_input = {"messages": [("user", user_input)]}
        output = await model.ainvoke(user_input)
        _, message = output["messages"][-1]
        print(message)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Tag images with the RAM model. Supports both regular and panoramic images."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TAGGING_MODEL,
        choices=TAGGING_MODELS,
        help=f"The model to execute (default: {DEFAULT_TAGGING_MODEL})",
    )
    parser.add_argument("--input", "-i", required=True, help="The image to use for inference")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The device to use for inference.",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--panoramic",
        "-p",
        action="store_true",
        help="Process the input as a panoramic (equirectangular) image",
    )
    parser.add_argument(
        "--face-size",
        type=int,
        default=512,
        help="Size of each cubemap face when processing panoramic images (default: 512)",
    )
    parser.add_argument(
        "--save-views",
        action="store_true",
        help="Save the extracted cubemap faces to disk (only for panoramic images)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for saving outputs (default: 'output'). Only used with --save-views",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=(
            "Show detailed output including tags per face and frequency (only for"
            " panoramic images)"
        ),
    )
    args = parser.parse_args()

    asyncio.run(run_serverless_tagging_model(args))
