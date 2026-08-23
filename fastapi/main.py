import os
import sys
from contextlib import asynccontextmanager
from datetime import timezone, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    HearingFeatures,
    PredictionResponse,
)
from dummy_data_generator import generate_dummy_batch, generate_dummy_hearing

# Feature definitions aligned with trained CatBoost model
FEATURE_COLUMNS: list[str] = [
    "District",
    "norm_purpose_grouped",
    "prev_norm_purpose",
    "hearing_idx",
    "case_age_at_hearing",
    "prev_gap",
    "gap_trend",
]
CAT_FEATURES: list[str] = ["District", "norm_purpose_grouped", "prev_norm_purpose"]
NUM_FEATURES: list[str] = [
    "hearing_idx",
    "case_age_at_hearing",
    "prev_gap",
    "gap_trend",
]


def find_and_load_model() -> tuple[CatBoostClassifier, str]:
    """Resolves and loads the stagnation_model.cbm from various possible parent/sibling paths."""
    base_dir = Path(__file__).resolve().parent
    parent_dir = base_dir.parent

    candidate_paths = [
        # Explicit env var
        os.getenv("MODEL_PATH"),
        # Model folder in parent folder (e.g. JurisFlow/model/stagnation_model.cbm)
        parent_dir / "model" / "stagnation_model.cbm",
        # Model in backend folder
        parent_dir / "backend" / "stagnation_model.cbm",
        parent_dir / "backend" / "jurisflow" / "models" / "stagnation_model.cbm",
        # Local paths relative to execution cwd
        Path("model/stagnation_model.cbm"),
        Path("../model/stagnation_model.cbm"),
        base_dir / "model" / "stagnation_model.cbm",
    ]

    valid_candidates = [
        Path(p).resolve()
        for p in candidate_paths
        if p and str(p).strip() and Path(p).exists()
    ]

    if not valid_candidates:
        searched_paths = "\n".join(
            [str(p) for p in candidate_paths if p and str(p).strip()]
        )
        raise FileNotFoundError(
            f"Could not locate 'stagnation_model.cbm'. Searched locations:\n{searched_paths}"
        )

    target_path = valid_candidates[0]
    model = CatBoostClassifier()
    model.load_model(str(target_path))
    return model, str(target_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the machine learning model on application startup."""
    try:
        model, path = find_and_load_model()
        app.state.model = model
        app.state.model_path = path
        print(f" JurisFlow Model successfully loaded from: {path}")
        print(f" Model Feature Columns: {FEATURE_COLUMNS}")
    except Exception as e:
        app.state.model = None
        app.state.model_path = None
        print(f" Error loading CatBoost model: {e}", file=sys.stderr)
    yield
    # Clean up on shutdown
    app.state.model = None


# Initialize FastAPI application
app = FastAPI(
    title="JurisFlow Stagnation Prediction API",
    description="Production-grade inference service for predicting procedural stagnation in Indian NI-138 court cases.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Configure CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_risk_factors(item: HearingFeatures) -> list[str]:
    """Identifies human-interpretable risk contributors from case trajectory heuristics."""
    factors = []

    # 1. Purpose repetition check
    if (
        item.norm_purpose_grouped.lower().strip()
        == item.prev_norm_purpose.lower().strip()
    ):
        factors.append(
            f"Repeated Purpose Streak: Both current and previous hearing purpose are '{item.norm_purpose_grouped}'"
        )

    # 2. Prolonged gap check
    if item.prev_gap > 45.0:
        factors.append(
            f"Extended Hearing Interval: Gap since previous hearing ({item.prev_gap:.1f} days) exceeds 45-day threshold"
        )

    # 3. Gap trend acceleration
    if item.gap_trend > 10.0:
        factors.append(
            f"Worsening Scheduling Delay: Current gap is {item.gap_trend:.1f} days longer than historical case average"
        )

    # 4. Prolonged litigation age
    if item.case_age_at_hearing > 365.0:
        factors.append(
            f"Case Age: Case has been active for {item.case_age_at_hearing:.0f} days (> 1 year)"
        )

    # 5. High hearing iteration count
    if item.hearing_idx >= 6:
        factors.append(
            f"High Hearing Index: Hearing #{item.hearing_idx} without final resolution"
        )

    if not factors:
        factors.append(
            "Nominal Trajectory: Case hearing progression intervals and purpose shifts are within normal bounds"
        )

    return factors


def format_risk_assessment(stagnation_prob: float) -> tuple[str, str]:
    """Classifies risk level and generates executive risk summary."""
    if stagnation_prob >= 0.65:
        risk_level = "HIGH"
        summary = (
            f"High Risk of Procedural Stagnation ({stagnation_prob * 100:.1f}%). "
            "Case is strongly exhibiting stagnation patterns (repeated purpose or prolonged adjournment). "
            "Immediate administrative review or expedited listing recommended."
        )
    elif stagnation_prob >= 0.35:
        risk_level = "MEDIUM"
        summary = (
            f"Moderate Risk of Procedural Stagnation ({stagnation_prob * 100:.1f}%). "
            "Case shows early indicators of procedural friction. Monitor next hearing outcome."
        )
    else:
        risk_level = "LOW"
        summary = (
            f"Low Risk ({stagnation_prob * 100:.1f}%). "
            "Case trajectory reflects healthy procedural progression."
        )
    return risk_level, summary


def run_inference_single(
    item: HearingFeatures, model: CatBoostClassifier | None
) -> PredictionResponse:
    """Performs inference for a single hearing record."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please verify model file path and restart service.",
        )

    # Prepare DataFrame matching exact CatBoost feature format
    data_dict = {
        "District": [str(item.District).strip()],
        "norm_purpose_grouped": [str(item.norm_purpose_grouped).lower().strip()],
        "prev_norm_purpose": [str(item.prev_norm_purpose).lower().strip()],
        "hearing_idx": [int(item.hearing_idx)],
        "case_age_at_hearing": [float(item.case_age_at_hearing)],
        "prev_gap": [float(item.prev_gap)],
        "gap_trend": [float(item.gap_trend)],
    }
    df = pd.DataFrame(data_dict)[FEATURE_COLUMNS]

    # Predict probabilities: [progression_prob (0), stagnation_prob (1)]
    probabilities = model.predict_proba(df)[0]
    progression_prob = float(probabilities[0])
    stagnation_prob = float(probabilities[1])
    is_stagnant = bool(stagnation_prob >= 0.5)

    risk_level, risk_summary = format_risk_assessment(stagnation_prob)
    top_factors = extract_risk_factors(item)

    return PredictionResponse(
        is_stagnant_predicted=is_stagnant,
        stagnation_probability=round(stagnation_prob, 4),
        progression_probability=round(progression_prob, 4),
        risk_level=risk_level,
        risk_summary=risk_summary,
        top_risk_factors=top_factors,
        input_features={col: data_dict[col][0] for col in FEATURE_COLUMNS},
        cnr=item.cnr,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/", tags=["Info"])
def get_root(request: Request):
    """Root endpoint returning service status and documentation link."""
    model = getattr(request.app.state, "model", None)
    return {
        "service": "JurisFlow Stagnation Prediction API",
        "status": "online",
        "model_loaded": model is not None,
        "model_path": getattr(request.app.state, "model_path", None),
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "features": "/features",
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /predict/batch",
            "dummy_data": "GET /dummy-data",
            "dummy_data_batch": "GET /dummy-data/batch",
            "predict_random": "GET /predict/random",
        },
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check(request: Request):
    """Checks model loading health and configuration."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return HealthCheckResponse(
        status="healthy",
        model_loaded=True,
        model_path=getattr(request.app.state, "model_path", "Unknown"),
        model_features=FEATURE_COLUMNS,
        categorical_features=CAT_FEATURES,
    )


@app.get("/features", tags=["Info"])
def get_features_info():
    """Returns the input schema details, feature types, and examples for API consumers."""
    return {
        "categorical_features": {
            "District": "District name where case is registered (e.g., 'Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala')",
            "norm_purpose_grouped": "Normalized hearing purpose ('appearance', 'evidence', 'arguments', 'notice', 'summons', 'other')",
            "prev_norm_purpose": "Normalized purpose of preceding hearing ('appearance', 'evidence', 'notice', etc.)",
        },
        "numerical_features": {
            "hearing_idx": "Sequence number of hearing (1, 2, 3...)",
            "case_age_at_hearing": "Total days since case was filed/first recorded",
            "prev_gap": "Days between previous hearing and current hearing",
            "gap_trend": "Trend indicator: (prev_gap - average_previous_gap)",
        },
        "sample_payload": {
            "District": "Ludhiana",
            "norm_purpose_grouped": "appearance",
            "prev_norm_purpose": "appearance",
            "hearing_idx": 4,
            "case_age_at_hearing": 150.0,
            "prev_gap": 50.0,
            "gap_trend": 10.0,
            "cnr": "PB01001234562024",
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(record: HearingFeatures, request: Request):
    """
    Predict procedural stagnation probability for a single court hearing.

    Accepts hearing metadata and case trajectory indicators, runs CatBoost inference,
    and returns stagnation risk probability and contextual risk factor diagnostics.
    """
    model = getattr(request.app.state, "model", None)
    return run_inference_single(record, model)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchPredictionRequest, request: Request):
    """
    Runs batch inference on multiple hearing records.

    Returns aggregated metrics (high/medium/low risk counts, average probability)
    along with detailed prediction responses for each record.
    """
    model = getattr(request.app.state, "model", None)
    predictions: list[PredictionResponse] = []

    # Process in batches of 1000 to manage memory and load securely
    CHUNK_SIZE = 1000
    for i in range(0, len(payload.records), CHUNK_SIZE):
        chunk = payload.records[i : i + CHUNK_SIZE]
        chunk_preds = [run_inference_single(rec, model) for rec in chunk]
        predictions.extend(chunk_preds)

    high_risk = sum(1 for p in predictions if p.risk_level == "HIGH")
    med_risk = sum(1 for p in predictions if p.risk_level == "MEDIUM")
    low_risk = sum(1 for p in predictions if p.risk_level == "LOW")
    avg_prob = (
        float(np.mean([p.stagnation_probability for p in predictions]))
        if predictions
        else 0.0
    )

    return BatchPredictionResponse(
        total_records=len(predictions),
        high_risk_count=high_risk,
        medium_risk_count=med_risk,
        low_risk_count=low_risk,
        average_stagnation_probability=round(avg_prob, 4),
        predictions=predictions,
    )


@app.get("/dummy-data", response_model=HearingFeatures, tags=["Dummy Data"])
def get_dummy_data(scenario: str = "random"):
    """
    Generate a single dummy hearing record.
    Scenario can be: 'high_risk', 'normal_progress', 'moderate_risk', or 'random'.
    """
    valid_scenarios = ["high_risk", "normal_progress", "moderate_risk", "random"]
    if scenario not in valid_scenarios:
        raise HTTPException(
            status_code=400, detail=f"Invalid scenario. Must be one of {valid_scenarios}"
        )
    data = generate_dummy_hearing(scenario)
    return data


@app.get("/dummy-data/batch", response_model=list[HearingFeatures], tags=["Dummy Data"])
def get_dummy_batch_data(count: int = 10, scenario: str = "mixed"):
    """
    Generate a batch of dummy hearing records.
    Scenario can be: 'high_risk', 'normal_progress', 'moderate_risk', 'random', or 'mixed'.
    """
    valid_scenarios = [
        "high_risk",
        "normal_progress",
        "moderate_risk",
        "random",
        "mixed",
    ]
    if scenario not in valid_scenarios:
        raise HTTPException(
            status_code=400, detail=f"Invalid scenario. Must be one of {valid_scenarios}"
        )
    if count > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100")
    data = generate_dummy_batch(count, scenario)
    return data


@app.get("/predict/random", response_model=PredictionResponse, tags=["Inference"])
def predict_random(request: Request, scenario: str = "random"):
    """
    Generates a random dummy hearing record and predicts its stagnation probability.
    """
    valid_scenarios = ["high_risk", "normal_progress", "moderate_risk", "random"]
    if scenario not in valid_scenarios:
        raise HTTPException(
            status_code=400, detail=f"Invalid scenario. Must be one of {valid_scenarios}"
        )

    dummy_data = generate_dummy_hearing(scenario)
    record = HearingFeatures(**dummy_data)
    model = getattr(request.app.state, "model", None)
    return run_inference_single(record, model)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
