"""
Comparison demonstration comparing Traditional vs ViTPose eye segmentation methods:
1. TraditionalEyeSegmenter: Classical morphological filtering & symmetry matching
2. ViTPoseEyeSegmenter: ViTPose AP-10K deep learning keypoint detection + contour refinement
"""

from pathlib import Path
import sys
import time

import cv2

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.animal_metrology import AnimalVisualMetrology


def main():
    img_path = Path("data/coco_samples/coco_000000555705_two_cats_shoes.jpg")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" EYE SEGMENTATION METHOD COMPARISON: TRADITIONAL vs VITPOSE ")
    print("=" * 70)

    # 1. Evaluate Method 1: Traditional
    print("\n[Method 1] TraditionalEyeSegmenter (OpenCV Morphology + Symmetry)")
    engine_trad = AnimalVisualMetrology(eye_method="traditional")
    t0 = time.time()
    res_trad = engine_trad.process_image(img_path)
    time_trad = (time.time() - t0) * 1000

    print(f"  Execution Time: {time_trad:.1f} ms")
    print(f"  Animals Detected: {res_trad['total_animals_detected']}")
    for a in res_trad["animals"]:
        print(f"   * Animal #{a['animal_id']} ({a['species']}):")
        print(f"     - Left Eye  : {a['left_eye']['centroid']} (Conf: {a['left_eye']['confidence']}, Area: {a['left_eye']['area']})")
        print(f"     - Right Eye : {a['right_eye']['centroid']} (Conf: {a['right_eye']['confidence']}, Area: {a['right_eye']['area']})")
        print(f"     - Intra-eye Distance: {a['intra_eye_distance_px']} px")
    if res_trad["inter_animal_right_eye_distances"]:
        p = res_trad["inter_animal_right_eye_distances"][0]
        print(f"   * Inter Right-Eye Distance ({p['pair']}): {p['distance_px']} px")

    out_trad = output_dir / "comparison_traditional.jpg"
    vis_trad = engine_trad.visualize(img_path, res_trad, out_trad)

    # Save zoomed-in verification crops for Traditional method
    cv2.imwrite(
        str(output_dir / "zoom_cat1_traditional.jpg"),
        cv2.resize(vis_trad[115:155, 225:305], (320, 160), interpolation=cv2.INTER_NEAREST),
    )
    cv2.imwrite(
        str(output_dir / "zoom_cat2_traditional.jpg"),
        cv2.resize(vis_trad[80:125, 340:425], (340, 180), interpolation=cv2.INTER_NEAREST),
    )

    # 2. Evaluate Method 2: ViTPose
    print("\n[Method 2] ViTPoseEyeSegmenter (ViTPose AP-10K Keypoints + Refinement)")
    engine_pose = AnimalVisualMetrology(eye_method="vitpose")
    t0 = time.time()
    res_pose = engine_pose.process_image(img_path)
    time_pose = (time.time() - t0) * 1000

    print(f"  Execution Time: {time_pose:.1f} ms")
    print(f"  Animals Detected: {res_pose['total_animals_detected']}")
    for a in res_pose["animals"]:
        print(f"   * Animal #{a['animal_id']} ({a['species']}):")
        print(f"     - Left Eye  : {a['left_eye']['centroid']} (Conf: {a['left_eye']['confidence']}, Area: {a['left_eye']['area']})")
        print(f"     - Right Eye : {a['right_eye']['centroid']} (Conf: {a['right_eye']['confidence']}, Area: {a['right_eye']['area']})")
        print(f"     - Intra-eye Distance: {a['intra_eye_distance_px']} px")
    if res_pose["inter_animal_right_eye_distances"]:
        p = res_pose["inter_animal_right_eye_distances"][0]
        print(f"   * Inter Right-Eye Distance ({p['pair']}): {p['distance_px']} px")

    out_pose = output_dir / "comparison_vitpose.jpg"
    engine_pose.visualize(img_path, res_pose, out_pose)

    print("\n" + "=" * 70)
    print(" COMPARISON SUMMARY ")
    print("=" * 70)
    print(f"| Metric                      | Traditional Method | ViTPose Method     |")
    print(f"|-----------------------------|--------------------|--------------------|")
    print(f"| Latency (ms)                | {time_trad:14.1f} ms | {time_pose:14.1f} ms |")
    print(f"| Animal 1 Intra-dist (px)    | {res_trad['animals'][0]['intra_eye_distance_px']:15.2f} px | {res_pose['animals'][0]['intra_eye_distance_px']:15.2f} px |")
    print(f"| Animal 2 Intra-dist (px)    | {res_trad['animals'][1]['intra_eye_distance_px']:15.2f} px | {res_pose['animals'][1]['intra_eye_distance_px']:15.2f} px |")
    print(f"| Inter Right-Eye dist (px)   | {res_trad['inter_animal_right_eye_distances'][0]['distance_px']:15.2f} px | {res_pose['inter_animal_right_eye_distances'][0]['distance_px']:15.2f} px |")
    print("=" * 70)


if __name__ == "__main__":
    main()

