import json
import os 
import subprocess
import sys

DATA_NAME = sys.argv[1]
MODEL_DIR = sys.argv[2]

model_dir = MODEL_DIR
save_dir = f"data/{DATA_NAME}_code.json"

if os.path.exists(save_dir + ".bpe"):
    subprocess.call(["rm", save_dir + ".bpe"])
    print("Removing old data cache...")

def read_code(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append([int(x) for x in line.strip().split()])
    return data

train_code = read_code(os.path.join(model_dir, "code_train.txt"))
valid_code = read_code(os.path.join(model_dir, "code_valid.txt"))

data = json.load(open(f"data/{DATA_NAME}.json", "r"))
res = {}
assert(len(train_code) == len(data["train"]["src"]))
assert(len(valid_code) == len(data["valid"]["src"]))
res["train"] = {"src": data["train"]["src"], "tgt": train_code}
res["valid"] = {"src": data["valid"]["src"], "tgt": valid_code}
res["test"] = {"src": data["test"]["src"]}

with open(save_dir, "w") as f:
    json.dump(res, f, indent=4)