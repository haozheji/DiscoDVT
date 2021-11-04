from nltk.tokenize import word_tokenize
from nltk import ngrams
import nltk
import sys
import torch
from collections import Counter

GEN_FILE = sys.argv[1]

CONTEXT_LENS = [8, 16]

def read(filename):
    with open(filename, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data


def build_dict(texts):
    t2i = {}
    for text in texts:
        for tok in text:
            if tok not in t2i:
                t2i[tok] = len(t2i)
    print("Build vocab size: {}".format(len(t2i)))
    return t2i
 
def tok_repeat_l(hypo_toks, context_len=16):
    hypo = torch.tensor(hypo_toks).long()
    T = hypo.size(0)
    
    # prev_hypo[t, :] = [y_1, y_2, ... , y_t-1, -1 ,-1, ... , -1]
    prev_hypo = hypo.expand(T, T).masked_fill(torch.ones(T, T).triu().bool(), -1)

    # prev_hypo[t, :] = [-1, ... , -1, y_t-k-1, ..., y_t-1, -1 ,-1, ... , -1]
    prev_hypo = prev_hypo.masked_fill(torch.ones(T, T).tril(-context_len).bool(), -1)

    repeat = (hypo[:, None] == prev_hypo)
    has_repeat = repeat.sum(1).gt(0)
    total_repeat = has_repeat.sum()

    return total_repeat * 1.0 / T 


hyps = read(GEN_FILE)

hyps = [word_tokenize(x) for x in hyps]

metrics = {}
for c_len in CONTEXT_LENS:
    metrics.update({f"tok_repeat_{c_len}": 0.0})

dictionary = build_dict(hyps)
hyp_ids = []
for hyp in hyps:
    hyp_id = []
    for tok in hyp:
        hyp_id.append(dictionary[tok])
    
    for c_len in CONTEXT_LENS:
        metrics[f"tok_repeat_{c_len}"] += tok_repeat_l(hyp_id, context_len=c_len)


for k, v in metrics.items():
    metrics[k] = v * 1.0 / len(hyps)

print(metrics)

    











