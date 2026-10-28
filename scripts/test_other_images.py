"""
Run full automatic pipeline on multiple other multi-animal COCO test images:
1. Two Bears (coco_000000020247_two_bears.jpg)
2. Two Dogs (coco_000000236784_two_dogs_couch.jpg)
3. Three Cats (coco_000000387383_three_cats.jpg)
"""

from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from app.services.animal_metrology import AnimalVisualMetrology

def run_tests():
    engine = AnimalVisualMetrology(eye_method="vitpose", conf_threshold=0.35)
    
    test_cases = [
        ("data/coco_samples/coco_000000020247_two_bears.jpg", "outputs/result_two_bears.jpg"),
        ("data/coco_samples/coco_000000236784_two_dogs_couch.jpg", "outputs/result_two_dogs.jpg"),
        ("data/coco_samples/coco_000000387383_three_cats.jpg", "outputs/result_three_cats.jpg"),
    ]
    
    for in_path, out_path in test_cases:
        p = Path(in_path)
        if not p.exists():
            print(f"Skipping: {p} not found")
            continue
            
        print("\n" + "=" * 65)
        print(f" PROCESSING: {p.name}")
        print("=" * 65)
        
        res = engine.process_image(p)
        print(f"Total Animals Detected: {res['total_animals_detected']}")
        
        for a in res["animals"]:
            aid = a["animal_id"]
            sp = a["species"]
            le = a["left_eye"]["centroid"]
            re = a["right_eye"]["centroid"]
            intra = a["intra_eye_distance_px"]
            print(f"  * Animal #{aid} [{sp.upper()}] (Conf: {a['confidence']}):")
            print(f"    - Left Eye : {le}")
            print(f"    - Right Eye: {re}")
            print(f"    - Intra-eye Distance: {intra} px")
            
        print("  * Inter-animal Right-Eye Distances:")
        for pair in res["inter_animal_right_eye_distances"]:
            print(f"    - {pair['pair']} ({pair['animal_1_species']} to {pair['animal_2_species']}): {pair['distance_px']} px")
            
        # Visualize
        vis = engine.visualize(p, res, out_path)
        print(f"Saved full annotated metrology image to: {out_path}")

if __name__ == "__main__":
    run_tests()

