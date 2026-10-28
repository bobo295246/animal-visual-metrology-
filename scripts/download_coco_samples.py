"""
Download and prepare multi-animal sample images from the official MS COCO 2017 dataset.

Fetches genuine COCO val2017 images and verifies them via YOLOv8-seg to guarantee
each image contains >= 2 animals suitable for animal segmentation and visual metrology.
"""

from collections import defaultdict
import json
from pathlib import Path
import sys
import urllib.request

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

SUPPORTED_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

ANNOTATIONS_URL = (
    "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
)
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"


def get_annotations():
    cache_path = PROJECT_ROOT / "data" / ".cache_instances_val2017.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Loading cached annotations from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Downloading COCO val2017 annotations (~19MB)...")
    req = urllib.request.Request(ANNOTATIONS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
        data = json.loads(content)
        return data


def download_image(url: str, dest_path: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_coco_samples(target_count: int = 100):
    out_dir = PROJECT_ROOT / "data" / "coco_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_data = get_annotations()
    categories = {c["id"]: c["name"] for c in coco_data["categories"]}
    animal_cat_ids = {cid for cid, name in categories.items() if name in SUPPORTED_ANIMAL_CLASSES}

    id_to_img_meta = {img["id"]: img for img in coco_data["images"]}

    img_animals = defaultdict(list)
    for ann in coco_data["annotations"]:
        cid = ann["category_id"]
        if cid in animal_cat_ids and not ann.get("iscrowd", 0):
            img_animals[ann["image_id"]].append(categories[cid])

    candidate_ids = [img_id for img_id, cats in img_animals.items() if len(cats) >= 2]
    print(f"Found {len(candidate_ids)} candidate images in COCO with >= 2 animal annotations.")

    yolo_model = YOLO("yolov8n-seg.pt")
    existing_files = list(out_dir.glob("*.jpg"))
    verified_images = []

    print("\nVerifying existing images in coco_samples...")
    for f in existing_files:
        try:
            res = yolo_model(str(f), conf=0.30, verbose=False)[0]
            animals_found = [
                res.names[int(b.cls[0])]
                for b in res.boxes
                if res.names[int(b.cls[0])] in SUPPORTED_ANIMAL_CLASSES
            ]
            if len(animals_found) >= 2:
                verified_images.append({
                    "file_name": f.name,
                    "path": str(f),
                    "animals": animals_found,
                    "animal_count": len(animals_found),
                })
        except Exception:
            pass

    print(f"Already have {len(verified_images)} verified multi-animal images.")

    for img_id in candidate_ids:
        if len(verified_images) >= target_count:
            break

        meta = id_to_img_meta[img_id]
        raw_fname = meta["file_name"]
        file_path = out_dir / raw_fname

        if file_path.exists():
            continue

        img_url = IMAGE_BASE_URL + raw_fname
        if not download_image(img_url, file_path):
            continue

        try:
            res = yolo_model(str(file_path), conf=0.30, verbose=False)[0]
            animals_found = [
                res.names[int(b.cls[0])]
                for b in res.boxes
                if res.names[int(b.cls[0])] in SUPPORTED_ANIMAL_CLASSES
            ]
            if len(animals_found) >= 2:
                verified_images.append({
                    "file_name": raw_fname,
                    "path": str(file_path),
                    "animals": animals_found,
                    "animal_count": len(animals_found),
                })
                print(
                    f"[{len(verified_images):3d}/{target_count}] Verified: {raw_fname} "
                    f"-> Animals: {', '.join(animals_found)}"
                )
            else:
                file_path.unlink(missing_ok=True)
        except Exception:
            file_path.unlink(missing_ok=True)

    manifest_path = out_dir / "coco_samples_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(verified_images, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"Success! Saved {len(verified_images)} verified multi-animal images to:")
    print(f"   {out_dir}")
    print(f"   Manifest: {manifest_path}")
    print("=" * 65)


if __name__ == "__main__":
    download_coco_samples(target_count=100)
