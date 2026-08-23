import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import BatchPredictionRequest
from main import run_inference_single
import catboost

def test_batch():
    # Load JSON
    with open("real_data_batch.json", "r") as f:
        data = json.load(f)
        
    # Validate against Schema
    try:
        req = BatchPredictionRequest(**data)
        print("Schema Validation: PASSED")
    except Exception as e:
        print("Schema Validation: FAILED")
        print(e)
        return
        
    # Try Inference
    try:
        # Load model
        model = catboost.CatBoostClassifier()
        model.load_model("../model/stagnation_model.cbm")
        print("Model Loaded: PASSED")
        
        for rec in req.records:
            res = run_inference_single(rec, model)
        print("Inference on all records: PASSED")
        print("Everything looks good! The JSON works perfectly.")
    except Exception as e:
        print("Inference: FAILED")
        print(e)

if __name__ == "__main__":
    test_batch()
