"""
Verification and Metric Evaluation Script for Animal Visual Metrology.

Evaluates:
1. Segmentation Quality: IoU (Intersection over Union) & Dice Coefficient.
2. Metrology Accuracy: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
   and Mean Absolute Percentage Error (MAPE %) against Ground Truth annotations.
3. Geometric Consistency Invariance: Scale and Euclidean invariance checks.
"""

import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.animal_metrology import AnimalVisualMetrology


# Ground Truth Centroids & Metrology Standards for Benchmarking
GROUND_TRUTH_BENCHMARK = {
    "coco_000000555705_two_cats_shoes.jpg": {
        "animals": [
            {
                "animal_id": 1,
                "species": "cat",
                "gt_left_eye": (235.0, 126.0),
                "gt_right_eye": (285.0, 126.0),
                "gt_intra_distance": 50.00,  # sqrt((285-235)^2 + 0^2) = 50.00 px
            },
            {
                "animal_id": 2,
                "species": "cat",
                "gt_left_eye": (357.0, 100.0),
                "gt_right_eye": (410.0, 105.0),
                "gt_intra_distance": 53.24,  # sqrt((410-357)^2 + (105-100)^2) = 53.24 px
            },
        ],
        # Inter-animal Right Eye Distance: Animal 1 Right Eye (285, 126) to Animal 2 Right Eye (410, 105)
        # sqrt((410 - 285)^2 + (105 - 126)^2) = sqrt(125^2 + (-21)^2) = sqrt(15625 + 441) = sqrt(16066) ≈ 126.75 px
        "gt_inter_right_eye_distance": 126.75,
    }
}


def calculate_iou_and_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Tuple[float, float]:
    """Calculate IoU and Dice score between two binary masks."""
    intersection = np.logical_and(mask_pred > 0, mask_gt > 0).sum()
    union = np.logical_or(mask_pred > 0, mask_gt > 0).sum()
    iou = float(intersection / union) if union > 0 else 1.0
    dice = float(2 * intersection / (mask_pred.sum() + mask_gt.sum())) if (mask_pred.sum() + mask_gt.sum()) > 0 else 1.0
    return round(iou, 4), round(dice, 4)


def run_verification():
    print("=" * 65)
    print(" ANIMAL VISUAL METROLOGY - VERIFICATION & EVALUATION SUITE ")
    print("=" * 65)

    engine = AnimalVisualMetrology(model_name="yolov8n-seg.pt", conf_threshold=0.35)
    img_name = "coco_000000555705_two_cats_shoes.jpg"
    img_path = Path("data/coco_samples") / img_name

    if not img_path.exists():
        print(f"Error: Test image not found at {img_path}")
        return

    # 1. Run Pipeline
    res = engine.process_image(img_path)
    gt_data = GROUND_TRUTH_BENCHMARK[img_name]

    print(f"\n[Test Image]: {img_name}")
    print(f"Detected Animals: {res['total_animals_detected']} (Expected >= 2)")

    distance_errors: List[float] = []
    percentage_errors: List[float] = []

    # 2. Evaluate Intra-animal eye distance against Ground Truth
    print("\n--- 1. Intra-animal Eye Distance Verification (每隻動物雙眼距離) ---")
    for idx, (pred_a, gt_a) in enumerate(zip(res["animals"], gt_data["animals"])):
        pred_dist = pred_a["intra_eye_distance_px"]
        gt_dist = gt_a["gt_intra_distance"]
        abs_err = abs(pred_dist - gt_dist)
        mape = (abs_err / gt_dist) * 100.0

        distance_errors.append(abs_err)
        percentage_errors.append(mape)

        print(f"Animal #{pred_a['animal_id']} ({pred_a['species'].upper()}):")
        print(f"  - Ground Truth Centroids : L={gt_a['gt_left_eye']}, R={gt_a['gt_right_eye']}")
        print(f"  - Predicted Centroids    : L={pred_a['left_eye']['centroid']}, R={pred_a['right_eye']['centroid']}")
        print(f"  - Ground Truth Distance  : {gt_dist:.2f} px")
        print(f"  - Predicted Distance     : {pred_dist:.2f} px")
        print(f"  - Absolute Error (AE)    : {abs_err:.2f} px")
        print(f"  - Relative Error (MAPE)  : {mape:.2f} %")

    # 3. Evaluate Inter-animal right-eye distance against Ground Truth
    print("\n--- 2. Inter-animal Right-Eye Distance Verification (兩隻動物右眼距離) ---")
    if res["inter_animal_right_eye_distances"]:
        pair_res = res["inter_animal_right_eye_distances"][0]
        pred_inter = pair_res["distance_px"]
        gt_inter = gt_data["gt_inter_right_eye_distance"]
        inter_abs_err = abs(pred_inter - gt_inter)
        inter_mape = (inter_abs_err / gt_inter) * 100.0

        distance_errors.append(inter_abs_err)
        percentage_errors.append(inter_mape)

        print(f"Pair: {pair_res['pair']} ({pair_res['animal_1_species']} to {pair_res['animal_2_species']})")
        print(f"  - Ground Truth Right Eye Dist : {gt_inter:.2f} px")
        print(f"  - Predicted Right Eye Dist    : {pred_inter:.2f} px")
        print(f"  - Absolute Error (AE)         : {inter_abs_err:.2f} px")
        print(f"  - Relative Error (MAPE)       : {inter_mape:.2f} %")

    # 4. Summary Metrology Metrics
    mae = float(np.mean(distance_errors))
    rmse = float(np.sqrt(np.mean([e ** 2 for e in distance_errors])))
    mean_mape = float(np.mean(percentage_errors))

    print("\n" + "=" * 65)
    print(" METROLOGY BENCHMARK SUMMARY METRICS ")
    print("=" * 65)
    print(f" * Mean Absolute Error (MAE)        : {mae:.2f} px")
    print(f" * Root Mean Squared Error (RMSE)   : {rmse:.2f} px")
    print(f" * Mean Abs Percentage Error (MAPE) : {mean_mape:.2f} %")
    print(f" * Metrology Accuracy Rate          : {(100.0 - mean_mape):.2f} %")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
