"""
JurisFlow - Dummy Data Generator & Inference Test Client
========================================================
Generates synthetic, realistic court hearing trajectory records for Indian NI-138 cases,
and provides utility functions to test the FastAPI stagnation prediction endpoints.

Usage Examples:
---------------
1. Generate and print placeholder records to stdout:
   python dummy_data_generator.py --count 3

2. Generate records and save to JSON and CSV:
   python dummy_data_generator.py --count 10 --output-json sample_cases.json --output-csv sample_cases.csv

3. Test live inference against the running FastAPI service:
   python dummy_data_generator.py --test-api --api-url http://127.0.0.1:8000
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import requests
except ImportError:
    requests = None


# Realistic categorical values derived from Punjab NI-138 judicial datasets
PUNJAB_DISTRICTS = [
    "Ludhiana",
    "Amritsar",
    "Jalandhar",
    "Patiala",
    "Bathinda",
    "Hoshiarpur",
    "SAS Nagar (Mohali)",
    "Gurdaspur",
    "Sangrur",
    "Ferozepur",
    "Kapurthala",
    "Pathankot",
]

PURPOSE_GROUPS = [
    "appearance",
    "notice",
    "summons",
    "evidence",
    "arguments",
    "other",
]


def generate_cnr(district: str, year: int = 2024) -> str:
    """Generates a realistic 16-character Case Navigation Record (CNR) identifier."""
    dist_code = district[:2].upper()
    random_num = random.randint(100000, 999999)
    return f"PB{dist_code}0{random_num}{year}"


def generate_dummy_hearing(scenario: str = "random", cnr_prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a single hearing record for inference.
    
    Parameters:
    -----------
    scenario: "high_risk" | "normal_progress" | "moderate_risk" | "random"
    """
    district = random.choice(PUNJAB_DISTRICTS)
    cnr = cnr_prefix or generate_cnr(district)
    hearing_date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")

    if scenario == "high_risk":
        # Characteristic of stagnation: repeated same purpose, long delays, accelerating gaps
        repeated_purpose = random.choice(["appearance", "evidence", "notice", "summons"])
        hearing_idx = random.randint(4, 9)
        prev_gap = round(random.uniform(48.0, 95.0), 1)  # > 45 days threshold
        gap_trend = round(random.uniform(10.0, 35.0), 1)  # widening delay trend
        case_age_at_hearing = round(hearing_idx * 55.0 + random.uniform(20.0, 100.0), 1)

        return {
            "District": district,
            "norm_purpose_grouped": repeated_purpose,
            "prev_norm_purpose": repeated_purpose,  # repeated streak
            "hearing_idx": hearing_idx,
            "case_age_at_hearing": case_age_at_hearing,
            "prev_gap": prev_gap,
            "gap_trend": gap_trend,
            "cnr": cnr,
            "hearing_date": hearing_date,
        }

    elif scenario == "normal_progress":
        # Characteristic of healthy progression: purpose advancing, short controlled gap
        purpose_transitions = [
            ("summons", "appearance"),
            ("appearance", "evidence"),
            ("evidence", "arguments"),
            ("arguments", "other"),
        ]
        prev_purpose, curr_purpose = random.choice(purpose_transitions)
        hearing_idx = random.randint(1, 3)
        prev_gap = round(random.uniform(12.0, 28.0), 1)  # fast turnaround
        gap_trend = round(random.uniform(-15.0, 2.0), 1)  # negative or flat trend
        case_age_at_hearing = round(hearing_idx * 22.0 + random.uniform(5.0, 20.0), 1)

        return {
            "District": district,
            "norm_purpose_grouped": curr_purpose,
            "prev_norm_purpose": prev_purpose,
            "hearing_idx": hearing_idx,
            "case_age_at_hearing": case_age_at_hearing,
            "prev_gap": prev_gap,
            "gap_trend": gap_trend,
            "cnr": cnr,
            "hearing_date": hearing_date,
        }

    elif scenario == "moderate_risk":
        # Middle-of-the-road case
        curr_purpose = random.choice(["evidence", "arguments"])
        prev_purpose = random.choice(["appearance", curr_purpose])
        hearing_idx = random.randint(3, 5)
        prev_gap = round(random.uniform(32.0, 44.0), 1)
        gap_trend = round(random.uniform(2.0, 8.0), 1)
        case_age_at_hearing = round(hearing_idx * 38.0 + random.uniform(10.0, 40.0), 1)

        return {
            "District": district,
            "norm_purpose_grouped": curr_purpose,
            "prev_norm_purpose": prev_purpose,
            "hearing_idx": hearing_idx,
            "case_age_at_hearing": case_age_at_hearing,
            "prev_gap": prev_gap,
            "gap_trend": gap_trend,
            "cnr": cnr,
            "hearing_date": hearing_date,
        }

    else:  # "random"
        selected_scenario = random.choices(
            ["high_risk", "moderate_risk", "normal_progress"],
            weights=[0.35, 0.30, 0.35],
            k=1
        )[0]
        return generate_dummy_hearing(scenario=selected_scenario, cnr_prefix=cnr)


def generate_dummy_batch(count: int = 10, scenario: str = "mixed") -> List[Dict[str, Any]]:
    """Generates a batch of synthetic hearing records."""
    records = []
    for i in range(count):
        if scenario == "mixed":
            # Rotate across scenarios for variety
            chosen_scenario = ["high_risk", "moderate_risk", "normal_progress"][i % 3]
        else:
            chosen_scenario = scenario
        records.append(generate_dummy_hearing(scenario=chosen_scenario))
    return records


def save_to_json(records: List[Dict[str, Any]], filepath: str) -> None:
    """Exports records to a formatted JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f" Successfully exported {len(records)} records to JSON: {filepath}")


def save_to_csv(records: List[Dict[str, Any]], filepath: str) -> None:
    """Exports records to a CSV file."""
    if pd is None:
        print(" pandas is required to export to CSV. Please install pandas or use JSON output.")
        return
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f" Successfully exported {len(records)} records to CSV: {filepath}")


def test_api_inference(base_url: str = "http://127.0.0.1:8000") -> None:
    """Sends test inference requests to the running FastAPI server and prints formatted results."""
    if requests is None:
        print(" requests library is required for live API testing. Install it via 'pip install requests'.")
        return

    print(f"\n=======================================================")
    print(f" Testing JurisFlow FastAPI Endpoints at {base_url}")
    print(f"=======================================================\n")

    # 1. Health check
    health_url = f"{base_url}/health"
    try:
        res = requests.get(health_url, timeout=5)
        if res.status_code == 200:
            print(f"[OK] GET /health: 200 OK")
            print(f"    Status: {res.json().get('status')}")
            print(f"    Model Path: {res.json().get('model_path')}")
        else:
            print(f"[FAIL] GET /health failed with status {res.status_code}: {res.text}")
            return
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Could not connect to {base_url}. Make sure FastAPI is running (`uvicorn main:app --reload`).")
        return

    # 2. Single high-risk prediction test
    predict_url = f"{base_url}/predict"
    print(f"\n--- Testing Single Prediction (High-Risk Case Scenario) ---")
    high_risk_sample = generate_dummy_hearing("high_risk")
    print(f"Payload:\n{json.dumps(high_risk_sample, indent=2)}")

    res = requests.post(predict_url, json=high_risk_sample, timeout=5)
    if res.status_code == 200:
        data = res.json()
        print(f"\nResult:")
        print(f"  • Predicted Stagnant: {data['is_stagnant_predicted']}")
        print(f"  • Stagnation Probability: {data['stagnation_probability'] * 100:.2f}%")
        print(f"  • Risk Tier: [{data['risk_level']}]")
        print(f"  • Summary: {data['risk_summary']}")
        print(f"  • Top Factors: {data['top_risk_factors']}")
    else:
        print(f"[FAIL] POST /predict failed ({res.status_code}): {res.text}")

    # 3. Single healthy progression prediction test
    print(f"\n--- Testing Single Prediction (Normal Progression Scenario) ---")
    normal_sample = generate_dummy_hearing("normal_progress")
    print(f"Payload:\n{json.dumps(normal_sample, indent=2)}")

    res = requests.post(predict_url, json=normal_sample, timeout=5)
    if res.status_code == 200:
        data = res.json()
        print(f"\nResult:")
        print(f"  • Predicted Stagnant: {data['is_stagnant_predicted']}")
        print(f"  • Stagnation Probability: {data['stagnation_probability'] * 100:.2f}%")
        print(f"  • Risk Tier: [{data['risk_level']}]")
        print(f"  • Summary: {data['risk_summary']}")
    else:
        print(f"[FAIL] POST /predict failed ({res.status_code}): {res.text}")

    # 4. Batch prediction test
    batch_url = f"{base_url}/predict/batch"
    print(f"\n--- Testing Batch Prediction (10 Case Batch) ---")
    batch_samples = generate_dummy_batch(10, scenario="mixed")
    batch_payload = {"records": batch_samples}

    res = requests.post(batch_url, json=batch_payload, timeout=8)
    if res.status_code == 200:
        batch_data = res.json()
        print(f"Batch Summary:")
        print(f"  • Total Cases Evaluated: {batch_data['total_records']}")
        print(f"  • High Risk Count: {batch_data['high_risk_count']}")
        print(f"  • Medium Risk Count: {batch_data['medium_risk_count']}")
        print(f"  • Low Risk Count: {batch_data['low_risk_count']}")
        print(f"  • Average Stagnation Probability: {batch_data['average_stagnation_probability'] * 100:.2f}%")
        print(f"\nBreakdown:")
        for idx, item in enumerate(batch_data['predictions'], 1):
            cnr = item.get('cnr', f'Case #{idx}')
            prob = item['stagnation_probability'] * 100
            print(f"  [{idx:02d}] {cnr} | Risk: {item['risk_level']:<6} | Stagnation Prob: {prob:5.1f}%")
    else:
        print(f"[FAIL] POST /predict/batch failed ({res.status_code}): {res.text}")

    print(f"\n=======================================================")
    print(f" Test Completed Successfully!")
    print(f"=======================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="JurisFlow - Generate dummy court case hearing records and test inference API."
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="Number of synthetic hearing records to generate (default: 5)"
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        choices=["mixed", "high_risk", "normal_progress", "moderate_risk", "random"],
        default="mixed",
        help="Scenario profile to simulate (default: mixed)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional file path to save output records as JSON"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional file path to save output records as CSV"
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Execute live inference tests against the FastAPI server"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL for the FastAPI service (default: http://127.0.0.1:8000)"
    )

    args = parser.parse_args()

    # If --test-api is selected, run API integration test suite
    if args.test_api:
        test_api_inference(base_url=args.api_url)
        return

    # Generate dummy records
    records = generate_dummy_batch(count=args.count, scenario=args.scenario)

    # Save to file if specified
    if args.output_json:
        save_to_json(records, args.output_json)
    if args.output_csv:
        save_to_csv(records, args.output_csv)

    # If no output file specified, print formatted JSON to console
    if not args.output_json and not args.output_csv:
        print(f"\nGenerated {len(records)} Dummy Hearing Records (Scenario: '{args.scenario}'):\n")
        print(json.dumps(records, indent=2))
        print(f"\nTip: Run with `--test-api` to send these directly to the FastAPI server.")


if __name__ == "__main__":
    main()
