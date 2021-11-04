import json
from transformers import BartTokenizer
import torch
from tqdm import tqdm
import sys

all_labels = ["none", "after12", "after21", "also12", "also21", 
        "although12", "although21", "and12", "and21", "as12", "as21",
        "before12", "before21", "so12", "so21", "still12", "still21", 
        "because12", "because21", "then12"]
        
print("# relations:", len(all_labels))
data_name = sys.argv[1]

def read_json(filename):
    return json.load(open(filename, "r"))

def encode_disc(tokenizer, data_disc):
    tgt, _labels, pos = [], [], []
    for line in tqdm(data_disc):
        edus = line["edus"]
        labels = []
        for x in line["labels"]:
            if x in all_labels:
                labels.append(all_labels.index(x))
            else:
                labels.append(0)
        text, pos_ids = [], []
        for i, edu in enumerate(edus):
            edu_tok = tokenizer.encode(" " + edu)
            if i == 0:
                edu_tok = edu_tok[:-1]
            elif i == len(edus) - 1:
                edu_tok = edu_tok[1:]
            else:
                edu_tok = edu_tok[1:-1]

            if i == 0:
                pos_ids.append(len(text) + 1)
            else:
                pos_ids.append(len(text))
            text.extend(edu_tok)
        
        tgt.append(text)
        _labels.append(labels)
        pos.append(pos_ids)

    return {"tgt": tgt, "labels": _labels, "pos": pos}


def proc(tokenizer, data, data_disc):
    res = {}
    print("encode train...")
    assert(len(data["train"]["src"]) == len(data_disc["train"]))
    src = []
    for line in data["train"]["src"]:
        src.append(tokenizer.encode(" " + line))
    
    res["train"] = {"src": src}
    res_train = encode_disc(tokenizer, data_disc["train"])
    res["train"].update(res_train)

    print("encode valid...")
    assert(len(data["valid"]["src"]) == len(data_disc["valid"]))
    src = []
    for line in data["valid"]["src"]:
        src.append(tokenizer.encode(" " + line))
    
    res["valid"] = {"src": src}
    res_valid = encode_disc(tokenizer, data_disc["valid"])
    res["valid"].update(res_valid)

    print("encode test...")
    assert(len(data["test"]["src"]) == len(data_disc["test"]))
    src = []
    for line in data["test"]["src"]:
        src.append(tokenizer.encode(" " + line))
    
    res["test"] = {"src": src}
    res_test = encode_disc(tokenizer, data_disc["test"])
    res["test"].update(res_test)

    return res

def main():
    train_disc = read_json("../" + "data/" + data_name + "_train_disc.json")
    valid_disc = read_json("../" + "data/" + data_name + "_valid_disc.json")
    test_disc = read_json("../" + "data/" + data_name + "_test_disc.json")
    data_disc = {"train": train_disc, "valid": valid_disc, "test": test_disc}

    data = read_json("../" + "data/" + data_name + ".json")

    tokenizer = BartTokenizer.from_pretrained("../models/bart-base")
    to_save = proc(tokenizer, data, data_disc)

    print("Save to cache...")
    torch.save(to_save, "../data/" + data_name + ".json.bpe")

if __name__ == "__main__":
    main()