import asyncio
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

from src.models import DEFAULT_DETECTION_MODEL, DETECTION_MODELS, get_model

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


def draw_detections_on_image(image: Image.Image, detections: list[str]) -> Image.Image:
    """Draw bounding boxes and labels on an image.

    Args:
    ----
        image (Image.Image): The input image.
        detections (list[str]): List of detection strings.

    Returns:
    -------
        Image.Image: Image with bounding boxes and labels drawn.

    """
    # Create a copy to avoid modifying the original
    img_annotated = image.copy()
    draw = ImageDraw.Draw(img_annotated)

    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()

    # Color palette for bounding boxes
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#FFA07A",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
        "#85C1E2",
        "#F8B739",
        "#52B788",
    ]

    for idx, detection in enumerate(detections):
        # Parse detection string: "label (confidence X.XXX, location [x1, y1, x2, y2])"
        try:
            label_part = detection.split(" (confidence ")[0]
            conf_part = detection.split("confidence ")[1].split(",")[0]
            loc_part = detection.split("location ")[1].strip("[])")
            coords = [float(x.strip()) for x in loc_part.split(",")]

            x1, y1, x2, y2 = coords
            color = colors[idx % len(colors)]

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label background
            label_text = f"{label_part} ({conf_part})"
            bbox = draw.textbbox((x1, y1 - 15), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 15), label_text, fill="white", font=font)

        except Exception:
            continue

    return img_annotated


async def run_serverless_detection_model(args: Namespace) -> None:
    """Run a detection model in serverless mode.

    Args:
    ----
        args (Namespace): The parsed arguments.

    """
    if args.model not in DETECTION_MODELS:
        avail_models = list(DETECTION_MODELS.keys())
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
        faces_dir = None
        if args.save_views:
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            base_output = args.output_dir if args.output_dir else "output"
            output_dir = Path(base_output) / f"panorama_detection_{timestamp}"
            faces_dir = output_dir / "faces"
            faces_dir.mkdir(parents=True, exist_ok=True)

        # Process each face and collect detections
        face_detections = {}
        all_object_types = []

        for face_name, face_image in faces.items():
            # Save face temporarily for processing
            temp_face_path = f"temp_{face_name}.jpg"
            face_image.save(temp_face_path)

            try:
                print(f"Detecting objects in {face_name} face...")
                user_input = {"messages": [("user", temp_face_path)]}
                output = await model.ainvoke(user_input)
                _, message = output["messages"][-1]

                # Message is a string with detections separated by newlines
                detections = message.split("\n") if message else []
                face_detections[face_name] = detections

                # Extract object types for summary
                for detection in detections:
                    try:
                        obj_type = detection.split(" (confidence ")[0]
                        all_object_types.append(obj_type)
                    except Exception:
                        continue

                # Save face images if requested
                if faces_dir:
                    # Save original face
                    face_path = faces_dir / f"{face_name}.jpg"
                    face_image.save(face_path)

                    # Save annotated face with bounding boxes
                    if detections:
                        annotated_image = draw_detections_on_image(face_image, detections)
                        annotated_path = faces_dir / f"{face_name}_annotated.jpg"
                        annotated_image.save(annotated_path)

            finally:
                # Clean up temporary file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

        # Calculate statistics
        from collections import Counter

        total_detections = sum(len(dets) for dets in face_detections.values())
        object_type_counts = Counter(all_object_types)
        unique_object_types = sorted(set(all_object_types))

        # Display results
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("PANORAMIC IMAGE DETECTION RESULTS")
        print("=" * 60)
        print(f"Processing completed in {processing_time:.2f} seconds")

        print("\nDetections by Face:\n")
        for face_name, detections in face_detections.items():
            print(f"{face_name.capitalize()} ({len(detections)} objects):")
            if detections:
                for detection in detections:
                    print(f"  - {detection}")
            else:
                print("  - No objects detected")
            print()

        print("Summary:")
        print(f"  Total objects detected: {total_detections}")
        print(f"  Unique object types: {len(unique_object_types)}")
        if unique_object_types:
            print(f"  Object types: {', '.join(unique_object_types)}")

        print("\n  Objects per face:")
        for face_name, detections in face_detections.items():
            print(f"    {face_name.capitalize()}: {len(detections)}")

        # Show object frequency if verbose
        if args.verbose and object_type_counts:
            print("\n  Most common objects:")
            for obj_type, count in object_type_counts.most_common(10):
                print(f"    {obj_type}: {count} detection(s)")

        # Save results to file if output directory exists
        if output_dir:
            results_file = output_dir / "results.txt"
            with open(results_file, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("PANORAMIC IMAGE DETECTION RESULTS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Input: {args.input}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Face size: {args.face_size}x{args.face_size}\n")
                f.write(f"Processing time: {processing_time:.2f} seconds\n")
                f.write(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("Detections by Face:\n\n")
                for face_name, detections in face_detections.items():
                    f.write(f"{face_name.capitalize()} ({len(detections)} objects):\n")
                    if detections:
                        for detection in detections:
                            f.write(f"  - {detection}\n")
                    else:
                        f.write("  - No objects detected\n")
                    f.write("\n")

                f.write("Summary:\n")
                f.write(f"  Total objects detected: {total_detections}\n")
                f.write(f"  Unique object types: {len(unique_object_types)}\n")
                if unique_object_types:
                    f.write(f"  Object types: {', '.join(unique_object_types)}\n")

                f.write("\n  Objects per face:\n")
                for face_name, detections in face_detections.items():
                    f.write(f"    {face_name.capitalize()}: {len(detections)}\n")

                if object_type_counts:
                    f.write("\n  Most common objects:\n")
                    for obj_type, count in object_type_counts.most_common():
                        f.write(f"    {obj_type}: {count} detection(s)\n")

            print(f"\n✓ Results saved to '{results_file}'")
            if faces_dir:
                print(f"✓ Face images saved to '{faces_dir}/'")

    else:
        # Original single image detection
        user_input = args.input
        user_input = {"messages": [("user", user_input)]}
        output = await model.ainvoke(user_input)
        _, message = output["messages"][-1]
        print(message)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Detect objects in images with OwlViT2. Supports both regular and panoramic images."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_DETECTION_MODEL,
        choices=DETECTION_MODELS,
        help=f"The model to execute (default: {DEFAULT_DETECTION_MODEL})",
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
        help=(
            "Save the extracted cubemap faces and annotated images to disk (only for"
            " panoramic images)"
        ),
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
        help=("Show detailed output including object frequency (only for panoramic images)"),
    )
    args = parser.parse_args()

    asyncio.run(run_serverless_detection_model(args))
