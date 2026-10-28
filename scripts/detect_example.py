"""
CLI tool for detecting animal eyes and measuring visual metrology distances.
Wraps the standalone function detect_animal_eyes_from_image().

Supports both:
1. Single Image Mode:
   python scripts/detect_example.py path/to/image.jpg
   python scripts/detect_example.py path/to/image.jpg --output outputs/result.jpg

2. Batch Folder Mode (傳入資料夾):
   python scripts/detect_example.py data/coco_samples
   python scripts/detect_example.py data/coco_samples --output outputs/my_batch_results
   python scripts/detect_example.py data/coco_samples --limit 10
"""

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from app.services import detect_animal_eyes_from_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_single_image(input_path: Path, output_file: Path, scale: float = None):
    print(f"\nProcessing image: {input_path}")
    print(f"   Model Engine : ViTPoseEyeSegmenter (ViTPose AP-10K)")
    print(f"   Output Image : {output_file}\n")

    result = detect_animal_eyes_from_image(
        image_path=str(input_path),
        output_path=str(output_file),
        pixel_to_mm_scale=scale,
    )

    print("=" * 65)
    print(f"Image: {result['image_name']} | Total Animals Found: {result['total_animals_detected']}")
    print("=" * 65)

    if not result["animals"]:
        print("No animals detected in this image.")
        return result

    for animal in result["animals"]:
        aid = animal["animal_id"]
        sp = animal["species"]
        le = animal["left_eye"]
        re = animal["right_eye"]
        dist_px = animal["intra_eye_distance_px"]
        dist_mm = animal.get("intra_eye_distance_mm")
        
        mm_str = f" ({dist_mm:.2f} mm)" if dist_mm is not None else ""
        area_l = le.get("area", 0.0)
        area_r = re.get("area", 0.0)
        print(f"\nAnimal #{aid} [{sp.upper()}]:")
        print(f"   - Left Eye  Centroid : {le['centroid']} (Area: {area_l} px)")
        print(f"   - Right Eye Centroid : {re['centroid']} (Area: {area_r} px)")
        print(f"   - Intra-eye Distance : {dist_px:.2f} px{mm_str}")

    inter_dists = result.get("inter_animal_right_eye_distances", [])
    if inter_dists:
        print("\nInter-animal Right-Eye Distances:")
        for pair in inter_dists:
            p_name = pair["pair"]
            d_px = pair["distance_px"]
            d_mm = pair.get("distance_mm")
            mm_str = f" ({d_mm:.2f} mm)" if d_mm is not None else ""
            print(f"   - {p_name}: {d_px:.2f} px{mm_str}")

    print("\n" + "=" * 65)
    print(f"Finished! Annotated image saved to:\n   {output_file.resolve()}\n")
    return result


def process_folder(folder_path: Path, output_folder: Path, scale: float = None, limit: int = None, save_images: bool = True):
    image_files = sorted([
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print(f"Error: No valid image files found in folder: {folder_path}")
        return

    if limit is not None and limit > 0:
        image_files = image_files[:limit]

    output_folder.mkdir(parents=True, exist_ok=True)
    images_out_dir = output_folder / "annotated_images"
    if save_images:
        images_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"BATCH MODE: Processing folder '{folder_path}'")
    print(f"   Total Images Found  : {len(image_files)}")
    print(f"   Annotated Output Dir: {images_out_dir if save_images else '(Disabled)'}")
    print(f"   Model Engine        : ViTPoseEyeSegmenter (ViTPose AP-10K)")
    print("=" * 70 + "\n")

    all_records = []
    species_counter = Counter()
    total_animals = 0
    multi_animal_images = 0
    t0 = time.time()

    for idx, img_path in enumerate(image_files, start=1):
        out_img_path = (images_out_dir / f"result_{img_path.name}") if save_images else None
        
        try:
            res = detect_animal_eyes_from_image(
                image_path=str(img_path),
                output_path=str(out_img_path) if out_img_path else None,
                pixel_to_mm_scale=scale,
            )
            count = res.get("total_animals_detected", 0)
            total_animals += count
            if count >= 2:
                multi_animal_images += 1

            species_list = [a["species"].upper() for a in res.get("animals", [])]
            for sp in species_list:
                species_counter[sp] += 1

            intra_dists = [f"{a['intra_eye_distance_px']:.1f}px" for a in res.get("animals", [])]
            intra_str = ", ".join(intra_dists) if intra_dists else "N/A"

            inter_dists = res.get("inter_animal_right_eye_distances", [])
            inter_str = f"Inter-eye: {inter_dists[0]['distance_px']:.1f}px" if inter_dists else ""

            print(
                f"[{idx:3d}/{len(image_files)}] {img_path.name:24s} -> "
                f"{count} animals ({', '.join(species_list) or 'None'}) | "
                f"Intra: [{intra_str}] {inter_str}"
            )

            # Collect records for batch CSV
            for a in res.get("animals", []):
                le = a["left_eye"]["centroid"]
                re = a["right_eye"]["centroid"]
                all_records.append({
                    "image_file": img_path.name,
                    "image_width": res["image_shape"][1],
                    "image_height": res["image_shape"][0],
                    "animal_id": a["animal_id"],
                    "species": a["species"],
                    "confidence": a["confidence"],
                    "bbox_x1": a["bbox"][0],
                    "bbox_y1": a["bbox"][1],
                    "bbox_x2": a["bbox"][2],
                    "bbox_y2": a["bbox"][3],
                    "left_eye_x": le[0],
                    "left_eye_y": le[1],
                    "right_eye_x": re[0],
                    "right_eye_y": re[1],
                    "intra_eye_dist_px": a["intra_eye_distance_px"],
                    "intra_eye_dist_mm": a.get("intra_eye_distance_mm"),
                    "total_animals_in_img": count,
                })

        except Exception as e:
            print(f"[{idx:3d}/{len(image_files)}] Error on {img_path.name}: {e}")

    elapsed = time.time() - t0

    # Export Batch CSV Report
    csv_file = output_folder / "batch_metrology_results.csv"
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Time Elapsed   : {elapsed:.2f} s ({elapsed/max(1, len(image_files)):.2f} s/image)")
    print(f"Images Processed     : {len(image_files)}")
    print(f"Images with >=2 Animals: {multi_animal_images} ({multi_animal_images/max(1, len(image_files))*100:.1f}%)")
    print(f"Total Animals Found  : {total_animals}")
    print(f"Species Breakdown    : {dict(species_counter)}")
    if all_records:
        mean_intra = df["intra_eye_dist_px"].mean()
        print(f"Mean Intra-Eye Dist  : {mean_intra:.2f} px")
        print(f"Detailed CSV Export  : {csv_file.resolve()}")
    if save_images:
        print(f"Annotated Images Dir : {images_out_dir.resolve()}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Detect animal eyes and calculate visual metrology distances using ViTPoseEyeSegmenter. Supports single image or entire folder."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="data/coco_samples",
        help="Path to target image file OR a directory containing images (default: data/coco_samples)",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_path",
        default=None,
        help="Output file path (single image mode) OR output folder path (folder mode).",
    )
    parser.add_argument(
        "--scale", "-s",
        dest="scale",
        type=float,
        default=None,
        help="Optional physical scale in mm/px (e.g. 0.5 mm/px)",
    )
    parser.add_argument(
        "--limit", "-l",
        dest="limit",
        type=int,
        default=None,
        help="Optional limit on number of images to process when passing a folder",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="In batch folder mode, disable saving rendered images to disk to maximize speed",
    )

    args = parser.parse_args()

    input_path = Path(args.image_path)
    if not input_path.exists():
        print(f"❌ Error: Path not found: {input_path}")
        sys.exit(1)

    # 1. Folder Mode
    if input_path.is_dir():
        out_folder = (
            Path(args.output_path)
            if args.output_path
            else PROJECT_ROOT / "outputs" / f"batch_{input_path.stem}"
        )
        process_folder(
            folder_path=input_path,
            output_folder=out_folder,
            scale=args.scale,
            limit=args.limit,
            save_images=not args.no_save_images,
        )

    # 2. Single Image Mode
    else:
        if args.output_path is None:
            out_dir = PROJECT_ROOT / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / f"result_{input_path.stem}.jpg"
        else:
            output_file = Path(args.output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        process_single_image(
            input_path=input_path,
            output_file=output_file,
            scale=args.scale,
        )


if __name__ == "__main__":
    main()
