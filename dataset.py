import json

# Load dataset from JSON file
with open("threats.json", "r") as f:
    threat_dataset = json.load(f)