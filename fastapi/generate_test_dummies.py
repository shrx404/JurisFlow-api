import json
from dummy_data_generator import generate_dummy_batch, generate_dummy_hearing

# Generate a dummy dataset for batch prediction
batch_data = {'records': generate_dummy_batch(10)}

with open('dummy_for_batch.json', 'w', encoding='utf-8') as f:
    json.dump(batch_data, f, indent=2)

# Generate a dummy dataset for single prediction
single_data = generate_dummy_hearing()

with open('dummy_single.json', 'w', encoding='utf-8') as f:
    json.dump(single_data, f, indent=2)

print("Generated dummy_for_batch.json and dummy_single.json successfully.")
