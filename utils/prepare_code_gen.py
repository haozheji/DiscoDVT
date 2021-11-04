import sys
import json
import os 

DATA_NAME = sys.argv[1]
data = json.load(open(f"data/{DATA_NAME}.json", "r"))
with open(f"data/{DATA_NAME}_code.json", "w") as f:
    json.dump({"test": {"src": data["test"]["src"]}}, f, indent=4)