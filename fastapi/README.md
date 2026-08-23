# JurisFlow FastAPI Inference Service

FastAPI inference service for predicting procedural stagnation in Indian NI-138 negotiable instrument court cases using a serialized **CatBoost Classifier** (`stagnation_model.cbm`).

---

## 📁 Directory Structure

```
JurisFlow/
├── model/
│   └── stagnation_model.cbm       # Serialized CatBoost ML Model
└── fastapi/
    ├── main.py                    # FastAPI application & inference logic
    ├── schemas.py                 # Pydantic request & response models
    ├── dummy_data_generator.py    # Synthetic test data generator & API test client
    ├── requirements.txt           # Python dependencies
    └── README.md                  # Documentation & examples
```

---

## 🚀 Getting Started

### 1. Install & Sync Dependencies

Make sure you are in the `fastapi` directory.

#### Using `uv` (Recommended)
```bash
cd fastapi
uv sync
```

#### Using `pip`
```bash
cd fastapi
pip install -r requirements.txt
```

### 2. Start the FastAPI Server

#### Using `uv` (Recommended)
```bash
uv run uvicorn main:app --reload --port 8000
```

> **Note:** Running `uvicorn` directly without `uv run` may attempt to use your global Python environment rather than the project's local `.venv`, which can cause `ModuleNotFoundError` for packages like `catboost`. Alternatively, activate `.venv` first (`.\.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS).

#### Using standard environment / activated venv
```bash
uvicorn main:app --reload --port 8000
```

- **API Docs (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative Docs (ReDoc):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)


---

## 📊 Endpoints

### 1. `POST /predict` (Single Hearing Prediction)
Evaluates a single court hearing trajectory.

**Request:**
```json
{
  "District": "Ludhiana",
  "norm_purpose_grouped": "appearance",
  "prev_norm_purpose": "appearance",
  "hearing_idx": 4,
  "case_age_at_hearing": 180.0,
  "prev_gap": 55.0,
  "gap_trend": 12.5,
  "cnr": "PBLU010123452024"
}
```

**Response:**
```json
{
  "is_stagnant_predicted": true,
  "stagnation_probability": 0.8124,
  "progression_probability": 0.1876,
  "risk_level": "HIGH",
  "risk_summary": "High Risk of Procedural Stagnation (81.2%). Case is strongly exhibiting stagnation patterns...",
  "top_risk_factors": [
    "Repeated Purpose Streak: Both current and previous hearing purpose are 'appearance'",
    "Extended Hearing Interval: Gap since previous hearing (55.0 days) exceeds 45-day threshold",
    "Worsening Scheduling Delay: Current gap is 12.5 days longer than historical case average"
  ],
  "input_features": {
    "District": "Ludhiana",
    "norm_purpose_grouped": "appearance",
    "prev_norm_purpose": "appearance",
    "hearing_idx": 4,
    "case_age_at_hearing": 180.0,
    "prev_gap": 55.0,
    "gap_trend": 12.5
  },
  "cnr": "PBLU010123452024",
  "timestamp": "2026-08-22T17:25:00.000Z"
}
```

---

### 2. `POST /predict/batch` (Batch Prediction)
Evaluates an array of hearing records and provides summary risk statistics.

**Request:**
```json
{
  "records": [
    {
      "District": "Ludhiana",
      "norm_purpose_grouped": "appearance",
      "prev_norm_purpose": "appearance",
      "hearing_idx": 4,
      "case_age_at_hearing": 180.0,
      "prev_gap": 55.0,
      "gap_trend": 12.5,
      "cnr": "PBLU010123452024"
    },
    {
      "District": "Amritsar",
      "norm_purpose_grouped": "evidence",
      "prev_norm_purpose": "appearance",
      "hearing_idx": 2,
      "case_age_at_hearing": 40.0,
      "prev_gap": 20.0,
      "gap_trend": -5.0,
      "cnr": "PBAM010678902024"
    }
  ]
}
```

---

## 🧪 Dummy Data Generator & Test Script

The `dummy_data_generator.py` script lets you generate realistic test data and test the API directly from the command line:

### Generate Dummy Data:
```bash
# Print 5 realistic mock case records:
python dummy_data_generator.py --count 5

# Generate specific scenarios ("high_risk", "normal_progress", "moderate_risk", "mixed"):
python dummy_data_generator.py --count 10 --scenario high_risk

# Save synthetic dataset to JSON or CSV:
python dummy_data_generator.py --count 20 --output-json test_cases.json --output-csv test_cases.csv
```

### Test Running Server:
```bash
# Run automated inference tests against the local server:
python dummy_data_generator.py --test-api
```
