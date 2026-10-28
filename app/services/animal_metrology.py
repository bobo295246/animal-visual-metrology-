"""
AnimalVisualMetrology: Unified class for COCO animal segmentation and visual metrology.

Supports two distinct, decoupled eye segmentation strategies:
1. 'traditional': Classical Computer Vision (Adaptive Thresholding + Circularity + Symmetry Matching)
2. 'vitpose': Deep Learning Keypoint Guided (ViTPose AP-10K Animal Pose + Local Contour Refinement)
"""

from itertools import combinations
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from app.services.eye_segmenters import TraditionalEyeSegmenter, ViTPoseEyeSegmenter, YoloPoseEyeSegmenter


class AnimalVisualMetrology:
    """
    Unified class for animal instance segmentation, eye contour segmentation,
    and visual distance metrology using YOLO and computer vision moments.
    """

    SUPPORTED_ANIMAL_CLASSES = {
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe"
    }

    def __init__(
        self,
        model_name: str = "yolov8n-seg.pt",
        eye_method: str = "vitpose",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        pixel_to_mm_scale: Optional[float] = None,
    ):
        """
        Initialize the YOLO segmentation model and chosen eye segmentation strategy.

        :param model_name: YOLO segmentation model path or identifier (e.g. 'yolov8n-seg.pt')
        :param eye_method: 'traditional' or 'vitpose' (or alias 'yolo_pose')
        :param conf_threshold: Confidence threshold for animal segmentation
        :param iou_threshold: NMS IoU threshold
        :param pixel_to_mm_scale: Optional physical calibration (mm per pixel)
        """
        self.model_name = model_name
        m = eye_method.lower()
        self.eye_method = "vitpose" if m in ("vitpose", "yolo_pose") else "traditional"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.pixel_to_mm_scale = pixel_to_mm_scale

        # 1. Animal body segmentation model
        self.model = YOLO(model_name)

        # 2. Eye segmentation strategy
        if self.eye_method == "vitpose":
            self.eye_segmenter = ViTPoseEyeSegmenter()
        else:
            self.eye_segmenter = TraditionalEyeSegmenter()

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        image_name: Optional[str] = None,
        override_eye_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full segmentation and metrology pipeline:
        1. Separate animal contours with YOLO instance segmentation.
        2. Isolate each animal's individual mask/ROI.
        3. Identify and segment Left Eye and Right Eye for each animal individually.
        4. Measure intra-animal inter-pupillary distance.
        5. Measure inter-animal right-eye distance for all animal pairs.
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img = cv2.imread(path_str)
            if img is None:
                raise FileNotFoundError(f"Failed to read image from path: {path_str}")
            if image_name is None:
                image_name = Path(path_str).name
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
            if image_name is None:
                image_name = "in_memory_image.jpg"
        else:
            raise TypeError("image_input must be a file path or numpy.ndarray")

        h, w = img.shape[:2]

        # Select eye segmenter
        segmenter = self.eye_segmenter
        active_method = self.eye_method
        if override_eye_method:
            m = override_eye_method.lower()
            active_method = "vitpose" if m in ("vitpose", "yolo_pose") else "traditional"
            segmenter = (
                ViTPoseEyeSegmenter() if active_method == "vitpose" else TraditionalEyeSegmenter()
            )

        # 1. YOLO Animal Body Segmentation Inference
        yolo_results = self.model(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detected_animals: List[Dict[str, Any]] = []

        for r in yolo_results:
            if r.masks is None or r.boxes is None:
                continue

            boxes = r.boxes
            masks = r.masks

            for i, (box, mask_data) in enumerate(zip(boxes, masks.data)):
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]

                if cls_name not in self.SUPPORTED_ANIMAL_CLASSES:
                    continue

                conf = float(box.conf[0])
                bbox = [float(v) for v in box.xyxy[0].cpu().numpy()]
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Resize binary mask to original image dimensions
                mask_np = mask_data.cpu().numpy().astype(np.float32)
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                binary_mask = (mask_resized > 0.5).astype(np.uint8)

                # Extract animal body contour polygon
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                body_contour = contours[0] if contours else None
                body_polygon = (
                    body_contour.reshape(-1, 2).tolist() if body_contour is not None else []
                )

                # 2. Isolate animal and segment Left and Right Eye individually
                left_eye, right_eye = segmenter.segment_eyes(
                    image_bgr=img,
                    animal_bbox=(x1, y1, x2, y2),
                    animal_mask=binary_mask,
                )

                # 3. Calculate Intra-animal eye distance (雙眼距離)
                intra_dist_px = self._euclidean_distance(
                    left_eye["centroid"], right_eye["centroid"]
                )
                intra_dist_mm = (
                    intra_dist_px * self.pixel_to_mm_scale
                    if self.pixel_to_mm_scale
                    else None
                )

                detected_animals.append({
                    "animal_id": len(detected_animals) + 1,
                    "species": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "body_polygon": body_polygon,
                    "body_mask": binary_mask,
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "intra_eye_distance_px": round(intra_dist_px, 2),
                    "intra_eye_distance_mm": (
                        round(intra_dist_mm, 2) if intra_dist_mm else None
                    ),
                })

        # Sort animals spatially from left to right for intuitive numbering
        detected_animals.sort(key=lambda a: a["bbox"][0])
        for idx, a in enumerate(detected_animals, start=1):
            a["animal_id"] = idx

        # 4. Measure Inter-animal right-eye distances (任意兩隻動物右眼距離)
        inter_right_eye_distances = self._measure_inter_right_eyes(detected_animals)

        return {
            "image_name": image_name,
            "image_shape": [h, w, 3],
            "eye_segmentation_method": active_method,
            "total_animals_detected": len(detected_animals),
            "animals": detected_animals,
            "inter_animal_right_eye_distances": inter_right_eye_distances,
        }

    def _euclidean_distance(
        self, pt1: Tuple[float, float], pt2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two 2D points."""
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def _measure_inter_right_eyes(
        self, animals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate inter-animal right-eye distances for all distinct pairs (任意兩隻動物右眼距離).
        """
        results = []
        for a1, a2 in combinations(animals, 2):
            pt1 = a1["right_eye"]["centroid"]
            pt2 = a2["right_eye"]["centroid"]
            dist_px = self._euclidean_distance(pt1, pt2)
            dist_mm = (
                dist_px * self.pixel_to_mm_scale if self.pixel_to_mm_scale else None
            )

            results.append({
                "pair": f"Animal_{a1['animal_id']}_to_Animal_{a2['animal_id']}",
                "animal_1_id": a1["animal_id"],
                "animal_2_id": a2["animal_id"],
                "animal_1_species": a1["species"],
                "animal_2_species": a2["species"],
                "animal_1_right_eye": pt1,
                "animal_2_right_eye": pt2,
                "distance_px": round(dist_px, 2),
                "distance_mm": round(dist_mm, 2) if dist_mm else None,
            })
        return results

    def _draw_pill_text(
        self,
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        font_scale: float = 0.45,
        thickness: int = 1,
    ):
        """Draw text with a small dark background badge for crisp readability."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = org
        cv2.rectangle(
            img,
            (x - 2, y - th - 3),
            (x + tw + 2, y + baseline + 1),
            bg_color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x, y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    def visualize(
        self,
        image_input: Union[str, Path, np.ndarray],
        analysis_result: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Render visual metrology overlay:
        - Transparent body contour masks
        - Bounding boxes and labels
        - Left Eye (Cyan) and Right Eye (Orange) contours and center markers
        - Intra-animal eye distance measurement line (雙眼連線)
        - Inter-animal right-eye distance measurement line (任意兩隻動物右眼連線)
        """
        if isinstance(image_input, (str, Path)):
            vis = cv2.imread(str(image_input))
        else:
            vis = image_input.copy()

        animals = analysis_result.get("animals", [])
        inter_dists = analysis_result.get("inter_animal_right_eye_distances", [])
        method_used = analysis_result.get("eye_segmentation_method", "traditional")

        colors = [
            (255, 99, 71),   # Coral
            (50, 205, 50),   # Lime Green
            (30, 144, 255),  # Dodger Blue
            (255, 165, 0),   # Orange
            (186, 85, 211),  # Medium Orchid
        ]

        # 1. Overlay animal body segmentation masks
        overlay = vis.copy()
        for idx, a in enumerate(animals):
            c = colors[idx % len(colors)]
            mask = a.get("body_mask")
            if mask is not None:
                overlay[mask > 0] = overlay[mask > 0] * 0.4 + np.array(c) * 0.6

            # Draw body contour line
            if a.get("body_polygon"):
                poly = np.array(a["body_polygon"], dtype=np.int32)
                cv2.polylines(vis, [poly], isClosed=True, color=c, thickness=2)

        # Blend mask overlay
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # 2. Draw Eye Contours, Centroids, and Intra-animal Eye Distance Lines
        for idx, a in enumerate(animals):
            aid = a["animal_id"]
            species = a["species"]
            l_eye = a["left_eye"]
            r_eye = a["right_eye"]
            lc = (int(l_eye["centroid"][0]), int(l_eye["centroid"][1]))
            rc = (int(r_eye["centroid"][0]), int(r_eye["centroid"][1]))

            # Left Eye: Cyan (255, 255, 0)
            if l_eye["polygon"]:
                lp = np.array(l_eye["polygon"], dtype=np.int32)
                cv2.polylines(vis, [lp], True, (255, 255, 0), 2)
            else:
                cv2.circle(vis, lc, 6, (255, 255, 0), 2)
            cv2.circle(vis, lc, 4, (255, 255, 0), -1)
            cv2.putText(vis, "L", (lc[0] - 12, lc[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # Right Eye: Magenta/Orange (0, 165, 255)
            if r_eye["polygon"]:
                rp = np.array(r_eye["polygon"], dtype=np.int32)
                cv2.polylines(vis, [rp], True, (0, 165, 255), 2)
            else:
                cv2.circle(vis, rc, 6, (0, 165, 255), 2)
            cv2.circle(vis, rc, 4, (0, 165, 255), -1)
            cv2.putText(vis, "R", (rc[0] + 6, rc[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            # Intra-eye distance line (雙眼連線 - Green)
            cv2.line(vis, lc, rc, (0, 255, 0), 2, cv2.LINE_AA)
            mid_x = int((lc[0] + rc[0]) / 2)
            mid_y = int((lc[1] + rc[1]) / 2) - 12
            intra_text = f"A{aid} Intra: {a['intra_eye_distance_px']}px"
            self._draw_pill_text(
                vis,
                intra_text,
                (mid_x - 35, mid_y),
                text_color=(0, 255, 0),
                bg_color=(20, 20, 20),
                font_scale=0.42,
                thickness=1,
            )

            # Animal bounding box tag
            bx1, by1, bx2, by2 = a["bbox"]
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
            self._draw_pill_text(
                vis,
                f"#{aid} {species} ({a['confidence']})",
                (bx1 + 4, max(18, by1 - 6)),
                text_color=(255, 255, 255),
                bg_color=(40, 40, 40),
                font_scale=0.48,
                thickness=1,
            )

        # 3. Draw Inter-animal Right-Eye Distance Lines (跨動物右眼連線 - Gold)
        for pair in inter_dists:
            pt1 = (int(pair["animal_1_right_eye"][0]), int(pair["animal_1_right_eye"][1]))
            pt2 = (int(pair["animal_2_right_eye"][0]), int(pair["animal_2_right_eye"][1]))

            cv2.line(vis, pt1, pt2, (0, 215, 255), 2, cv2.LINE_AA)
            mid_pt = (int((pt1[0] + pt2[0]) / 2) - 40, int((pt1[1] + pt2[1]) / 2) + 20)
            inter_text = f"Right-Eyes: {pair['distance_px']}px"
            self._draw_pill_text(
                vis,
                inter_text,
                mid_pt,
                text_color=(0, 215, 255),
                bg_color=(20, 20, 20),
                font_scale=0.48,
                thickness=1,
            )

        # Method indicator banner in upper-left corner
        self._draw_pill_text(
            vis,
            f"Eye Method: {method_used.upper()}",
            (12, 24),
            text_color=(255, 255, 255),
            bg_color=(10, 50, 100),
            font_scale=0.55,
            thickness=1,
        )

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), vis)
            print(f"Annotated metrology image saved to: {out_p}")

        return vis

    def export_to_csv(
        self,
        analysis_results_list: List[Dict[str, Any]],
        csv_path: Union[str, Path] = "data/sample_measurements.csv",
    ) -> pd.DataFrame:
        """
        Export animal measurements into a standardized CSV table.
        """
        records = []
        for res in analysis_results_list:
            img_name = res["image_name"]
            method = res.get("eye_segmentation_method", "traditional")
            animals = res["animals"]
            inter_map = {
                (p["animal_1_id"], p["animal_2_id"]): p["distance_px"]
                for p in res.get("inter_animal_right_eye_distances", [])
            }

            for a in animals:
                aid = a["animal_id"]
                associated_inter = [
                    f"A{k[0]}-A{k[1]}:{d}px"
                    for k, d in inter_map.items()
                    if aid in k
                ]
                records.append({
                    "image_name": img_name,
                    "eye_method": method,
                    "animal_id": aid,
                    "species": a["species"],
                    "confidence": a["confidence"],
                    "bbox_x1": a["bbox"][0],
                    "bbox_y1": a["bbox"][1],
                    "bbox_x2": a["bbox"][2],
                    "bbox_y2": a["bbox"][3],
                    "left_eye_x": a["left_eye"]["centroid"][0],
                    "left_eye_y": a["left_eye"]["centroid"][1],
                    "right_eye_x": a["right_eye"]["centroid"][0],
                    "right_eye_y": a["right_eye"]["centroid"][1],
                    "intra_eye_distance_px": a["intra_eye_distance_px"],
                    "intra_eye_distance_mm": a["intra_eye_distance_mm"],
                    "inter_right_eye_distances": "; ".join(associated_inter) if associated_inter else "N/A",
                })

        df = pd.DataFrame(records)
        out_file = Path(csv_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"Exported {len(df)} measurement records to {out_file}")
        return df


# Global singleton instance for convenient functional usage
_DEFAULT_ENGINE = None


def detect_animal_eyes_from_image(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    pixel_to_mm_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenient standalone function: input an image path to automatically detect
    all animals, segment their bodies and eyes via ViTPoseEyeSegmenter,
    and measure intra-animal and inter-animal eye distances.

    :param image_path: Path to the input image file.
    :param output_path: Optional path to save the visualized annotated image.
    :param pixel_to_mm_scale: Optional physical calibration scale (mm per pixel).
    :return: Dictionary containing all detected animals, eye coordinates, and measured distances.
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = AnimalVisualMetrology(
            model_name="yolov8n-seg.pt",
            eye_method="vitpose",
            conf_threshold=0.35,
            pixel_to_mm_scale=pixel_to_mm_scale,
        )
    elif pixel_to_mm_scale is not None:
        _DEFAULT_ENGINE.pixel_to_mm_scale = pixel_to_mm_scale

    # 1. Run detection & metrology pipeline
    result = _DEFAULT_ENGINE.process_image(
        image_input=image_path,
        override_eye_method="vitpose",
    )

    # 2. Optionally render and save visual annotated image
    if output_path is not None:
        _DEFAULT_ENGINE.visualize(
            image_input=image_path,
            analysis_result=result,
            output_path=output_path,
        )
        result["annotated_output_path"] = str(output_path)

    return result
