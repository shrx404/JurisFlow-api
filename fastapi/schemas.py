from typing import Any

from pydantic import BaseModel, Field


class HearingFeatures(BaseModel):
    District: str = Field(
        default="Ludhiana",
        description=(
            "District court where the NI-138 case is being heard "
            "(e.g., Ludhiana, Amritsar, Jalandhar, Patiala)"
        ),
    )
    norm_purpose_grouped: str = Field(
        default="appearance",
        description=(
            "Normalized and grouped purpose of current hearing "
            "(e.g., appearance, evidence, notice, arguments, summons, other)"
        ),
    )
    prev_norm_purpose: str = Field(
        default="appearance",
        description="Normalized purpose of the immediately preceding hearing",
    )
    hearing_idx: int = Field(
        default=3,
        ge=1,
        description="Sequential index of the hearing for this case (1-based integer)",
    )
    case_age_at_hearing: float = Field(
        default=120.0,
        ge=0.0,
        description="Case age in days at the time of this hearing",
    )
    prev_gap: float = Field(
        default=45.0, ge=0.0, description="Days elapsed since the previous hearing"
    )
    gap_trend: float = Field(
        default=5.0,
        description="Trend of hearing gap (prev_gap - average_previous_gap)",
    )
    cnr: str | None = Field(
        default=None,
        description="Optional Case Navigation Record identifier (for tracking/audit purposes)",
    )
    hearing_date: str | None = Field(
        default=None, description="Optional hearing date string (YYYY-MM-DD)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "District": "Ludhiana",
                "norm_purpose_grouped": "appearance",
                "prev_norm_purpose": "appearance",
                "hearing_idx": 4,
                "case_age_at_hearing": 180.0,
                "prev_gap": 55.0,
                "gap_trend": 12.5,
                "cnr": "PB01001234562024",
                "hearing_date": "2024-06-15",
            }
        }
    }


class PredictionResponse(BaseModel):
    is_stagnant_predicted: bool = Field(
        description="True if model predicts stagnation (class 1), False otherwise (class 0)"
    )
    stagnation_probability: float = Field(
        description="Probability of procedural stagnation at next hearing (0.0 to 1.0)"
    )
    progression_probability: float = Field(
        description="Probability of healthy case progression (0.0 to 1.0)"
    )
    risk_level: str = Field(description="Risk categorization: LOW, MEDIUM, or HIGH")
    risk_summary: str = Field(description="Human-readable assessment summary")
    top_risk_factors: list[str] = Field(
        default_factory=list,
        description="Identified risk contributors based on trajectory heuristics",
    )
    input_features: dict[str, Any] = Field(
        description="Cleaned input feature vector used for prediction"
    )
    cnr: str | None = Field(default=None, description="CNR number if provided")
    timestamp: str = Field(description="UTC timestamp of inference execution")


class BatchPredictionRequest(BaseModel):
    records: list[HearingFeatures] = Field(
        description="List of hearing records to run batch inference on", min_length=1
    )


class BatchPredictionResponse(BaseModel):
    total_records: int = Field(description="Total number of evaluated records")
    high_risk_count: int = Field(description="Count of cases classified as HIGH risk")
    medium_risk_count: int = Field(
        description="Count of cases classified as MEDIUM risk"
    )
    low_risk_count: int = Field(description="Count of cases classified as LOW risk")
    average_stagnation_probability: float = Field(
        description="Mean stagnation probability across the batch"
    )
    predictions: list[PredictionResponse] = Field(
        description="Individual prediction results for each record"
    )


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_features: list[str]
    categorical_features: list[str]
