import time
from fastapi.testclient import TestClient
from main import app

# Ensure lifespan is triggered for TestClient
with TestClient(app) as client:
    def create_payload(num_records):
        record = {
            "District": "Ludhiana",
            "norm_purpose_grouped": "appearance",
            "prev_norm_purpose": "appearance",
            "hearing_idx": 4,
            "case_age_at_hearing": 150.0,
            "prev_gap": 50.0,
            "gap_trend": 10.0,
            "cnr": "PB01001234562024"
        }
        return {"records": [record for _ in range(num_records)]}

    batch_sizes = [1000, 5000, 10000, 20000, 50000]

    print("Benchmarking Batch Predictions...")
    print("-" * 60)
    for size in batch_sizes:
        payload = create_payload(size)
        
        try:
            start = time.time()
            response = client.post("/predict/batch", json=payload)
            end = time.time()
            
            duration = end - start
            status = response.status_code
            
            if status == 200:
                print(f"Batch Size: {size:6d} | Status: {status} | Time: {duration:6.2f}s | Per Record: {(duration/size)*1000:6.2f}ms")
            else:
                print(f"Batch Size: {size:6d} | Status: {status} | Error: {response.text[:100]}")
        except Exception as e:
            print(f"Batch Size: {size:6d} | Error: {str(e)}")
            break
