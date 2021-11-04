import torch
import sys

DATA_NAME = sys.argv[1]
GEN_CODE_DIR = sys.argv[2]


data_cache_dir = f"data/{DATA_NAME}.json.bpe"

print("Loading data cache...")
data = torch.load(data_cache_dir)
print("Loading latent codes...")
test_code = [[int(x) for x in y.strip().split()] for y in open(GEN_CODE_DIR, "r").readlines()]
data["test"]["code"] = test_code
print("Saving data cache...")
torch.save(data, data_cache_dir)