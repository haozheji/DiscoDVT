import random
import os
import json

random.seed(42)

def write(data, filename):
    with open(filename, "w") as f:
        for line in data:
            f.write(line + '\n')

titles = [x.strip() for x in open("../data/titles", "r").readlines()]
plots = []
with open("../data/plots", "r") as f:
    plot = []
    for line in f.readlines():
        if line.strip() == "<EOS>":
            plot_line = " ".join(plot)
            plots.append(plot_line)
            plot = []
        else:
            plot.append(line.strip())


train_src, train_tgt = [], []
valid_src, valid_tgt = [], []
test_src, test_tgt = [], []

for src, tgt in zip(titles, plots):
    prob = random.random()
    if prob <= 0.9:
        train_src.append(src)
        train_tgt.append(tgt)
    elif prob <= 0.95:
        valid_src.append(src)
        valid_tgt.append(tgt)
    else:
        test_src.append(src)
        test_tgt.append(tgt)

print("# Train: {}".format(len(train_src)))
print("# Valid: {}".format(len(valid_src)))
print("# Test: {}".format(len(test_src)))

res = {}

res["train"] = {"src": train_src, "tgt": train_tgt}
res["valid"] = {"src": valid_src, "tgt": valid_tgt}
res["test"] = {"src": test_src, "tgt": test_tgt}

with open("../data/wikiplots.json", "w") as f:
    json.dump(res, f, indent=4)



