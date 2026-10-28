"""FastAPI endpoints for authentication, metrology inference, CSV, and benchmarks."""

import base64
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
import numpy as np

from app.core.config import settings
from app.core.security import authenticate_user, create_access_token, get_current_user
from app.models.schemas import (
    BenchmarkVerificationResponse,
    LoginRequest,
    MetrologyAnalysisResponse,
    TokenResponse,
)
from app.services.animal_metrology import AnimalVisualMetrology

router = APIRouter()

# Instantiate single shared metrology engine instance
metrology_engine = AnimalVisualMetrology(
    model_name=settings.YOLO_SEG_MODEL,
    eye_method=settings.DEFAULT_EYE_METHOD,
    conf_threshold=settings.YOLO_CONF_THRESHOLD,
    iou_threshold=settings.YOLO_IOU_THRESHOLD,
    pixel_to_mm_scale=settings.DEFAULT_PIXEL_TO_MM_SCALE,
)


@router.post("/auth/login", response_model=TokenResponse, summary="Login to obtain Bearer JWT token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate using default test credentials:
    - Admin: `admin` / `AdminPassword123!`
    - Tester: `metrology_tester` / `TesterPassword123!`
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        role=user["role"],
        username=user["username"],
    )


@router.post("/auth/login-json", response_model=TokenResponse, summary="Login via JSON body")
async def login_json(body: LoginRequest):
    """Authenticate via JSON payload."""
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        role=user["role"],
        username=user["username"],
    )


@router.post("/metrology/analyze-image", response_model=MetrologyAnalysisResponse, summary="Analyze uploaded animal image")
async def analyze_image(
    file: UploadFile = File(...),
    eye_method: str = Form("vitpose", description="Eye segmentation method: 'vitpose' (or alias 'yolo_pose') or 'traditional'"),
    scale_mm_per_px: Optional[float] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload an image (>= 2 animals) to run:
    1. YOLO Instance Segmentation (Animal bodies)
    2. Left Eye & Right Eye Contour Segmentation (via YOLO-Pose or Traditional algorithm)
    3. Intra-animal eye distance measurement (雙眼距離)
    4. Inter-animal right-eye distance measurement (任意兩隻動物右眼距離)
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    if scale_mm_per_px:
        metrology_engine.pixel_to_mm_scale = scale_mm_per_px
    else:
        metrology_engine.pixel_to_mm_scale = settings.DEFAULT_PIXEL_TO_MM_SCALE

    # Execute segmentation & metrology
    result = metrology_engine.process_image(
        img,
        image_name=file.filename,
        override_eye_method=eye_method,
    )

    # Render annotated image to outputs/
    output_dir = Path(settings.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_filename = f"annotated_{Path(file.filename).stem}.jpg"
    annotated_path = output_dir / annotated_filename
    metrology_engine.visualize(img, result, annotated_path)

    # Prepare response
    result_dict = {
        "image_name": result["image_name"],
        "image_shape": result["image_shape"],
        "total_animals_detected": result["total_animals_detected"],
        "annotated_image_url": f"/outputs/{annotated_filename}",
        "animals": result["animals"],
        "inter_animal_right_eye_distances": result["inter_animal_right_eye_distances"],
    }
    return result_dict


@router.get("/metrology/sample-measurements-csv", summary="Download sample measurements CSV")
async def get_sample_csv():
    """Download the pre-generated sample CSV with multi-animal COCO measurements."""
    csv_path = Path(settings.CSV_EXPORT_PATH)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV report not generated yet")
    return FileResponse(
        str(csv_path),
        media_type="text/csv",
        filename="coco_animal_measurements.csv",
    )


@router.get("/metrology/benchmark-verification", response_model=BenchmarkVerificationResponse, summary="Get Ground Truth evaluation report")
async def get_benchmark_report():
    """
    Run and return the Ground Truth benchmark evaluation metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - Metrology Accuracy Rate
    """
    img_name = "coco_000000555705_two_cats_shoes.jpg"
    img_path = Path("data/coco_samples") / img_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark image not found")

    res = metrology_engine.process_image(img_path)

    # Ground Truth values
    gt_evals = [
        {
            "metric": "Cat 1 Intra-pupillary Distance",
            "ground_truth_px": 50.00,
            "predicted_px": res["animals"][0]["intra_eye_distance_px"],
            "abs_error_px": round(abs(res["animals"][0]["intra_eye_distance_px"] - 50.00), 2),
            "mape_pct": round(abs(res["animals"][0]["intra_eye_distance_px"] - 50.00) / 50.00 * 100, 2),
        },
        {
            "metric": "Cat 2 Intra-pupillary Distance",
            "ground_truth_px": 53.24,
            "predicted_px": res["animals"][1]["intra_eye_distance_px"],
            "abs_error_px": round(abs(res["animals"][1]["intra_eye_distance_px"] - 53.24), 2),
            "mape_pct": round(abs(res["animals"][1]["intra_eye_distance_px"] - 53.24) / 53.24 * 100, 2),
        },
        {
            "metric": "Inter-animal Right Eye Distance",
            "ground_truth_px": 126.75,
            "predicted_px": res["inter_animal_right_eye_distances"][0]["distance_px"],
            "abs_error_px": round(abs(res["inter_animal_right_eye_distances"][0]["distance_px"] - 126.75), 2),
            "mape_pct": round(abs(res["inter_animal_right_eye_distances"][0]["distance_px"] - 126.75) / 126.75 * 100, 2),
        },
    ]

    mae = float(np.mean([e["abs_error_px"] for e in gt_evals]))
    rmse = float(np.sqrt(np.mean([e["abs_error_px"] ** 2 for e in gt_evals])))
    mape = float(np.mean([e["mape_pct"] for e in gt_evals]))

    return BenchmarkVerificationResponse(
        test_image=img_name,
        total_animals_evaluated=len(res["animals"]),
        mean_absolute_error_px=round(mae, 2),
        root_mean_squared_error_px=round(rmse, 2),
        mean_absolute_percentage_error_pct=round(mape, 2),
        accuracy_rate_pct=round(100.0 - mape, 2),
        evaluations=gt_evals,
    )
