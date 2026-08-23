import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import app
from fastapi.testclient import TestClient
import warnings
warnings.filterwarnings('ignore')

def run_test():
    # Using context manager ensures startup events (model loading) fire correctly
    with TestClient(app) as client:
        # Load JSON payload
        with open("real_data_batch.json", "r") as f:
            payload = json.load(f)
            
        print("Sending POST request to /predict/batch...")
        response = client.post("/predict/batch", json=payload)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("\n--- API RESPONSE SUMMARY ---")
            print(f"Total Records Evaluated: {data['total_records']}")
            print(f"High Risk Cases:         {data['high_risk_count']}")
            print(f"Medium Risk Cases:       {data['medium_risk_count']}")
            print(f"Low Risk Cases:          {data['low_risk_count']}")
            print(f"Avg Stagnation Prob:     {data['average_stagnation_probability'] * 100:.2f}%")
            
            print("\n--- FIRST 2 PREDICTIONS ---")
            print(json.dumps(data['predictions'][:2], indent=2))
            
            print("\nTest completely successful!")
        else:
            print("Error:")
            print(response.text)

if __name__ == "__main__":
    run_test()
