from typing import List, Dict
import sys
import collections
import math
import csv
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk


def read(filename):
    with open(filename, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data


def distinct_ngrams(inputs, n, vocabs=None):
    output = {}
    for input in inputs:
        for i in range(len(input)-n+1):
            g = ' '.join(input[i:i+n])
            valid = True
            if vocabs is not None:
                for tok in g.split():
                    if tok not in vocabs:
                        valid = False
                        break
            if valid:
                output.setdefault(g, 0)
                output[g] += 1

    if sum(output.values())==0:
        ratio = 0
    else:
        ratio = float(len(output.keys()))/ sum(output.values())

    return ratio


def sentence_bleu(refs, hypos, order=None):
    bleu = 0.0
    for r,h in zip(refs, hypos):
        if order == None:
            weight = [0.25, 0.25, 0.25, 0.25]
        elif order == 1:
            weight = [1.0, 0.0, 0.0, 0.0]
        elif order == 2:
            weight = [0.5, 0.5, 0.0, 0.0]
        elif order == 3:
            weight = [0.33, 0.33, 0.33, 0.0]
        bleu += nltk.translate.bleu_score.sentence_bleu([r], h, weights=weight)
    return bleu / len(hypos)
  
	
def evaluate(GEN_FILE):
    preds = read(GEN_FILE)
    refs = read(REF_FILE)
    res = {"B1": 0.0, "B2": 0.0, "rB1": 0.0, "rB2": 0.0, "D4":0.0, "D5":0.0}
    tok_preds = [word_tokenize(x) for x in preds]
    tok_refs = [word_tokenize(x) for x in refs]

    vocabs = []
    for ref in tok_refs:
        vocabs.extend([x.lower() for x in ref])
    
    vocabs = set(vocabs)

    res["B1"] = sentence_bleu(tok_refs, tok_preds, order=1)
    res["B2"] = sentence_bleu(tok_refs, tok_preds, order=2)

    res["rB1"] = sentence_bleu(tok_preds, tok_refs, order=1)
    res["rB2"] = sentence_bleu(tok_preds, tok_refs, order=2)

    tok_preds_uncased = [[y.lower() for y in x] for x in tok_preds]

    res["D4"] += distinct_ngrams(tok_preds_uncased, 4, vocabs)
    res["D5"] += distinct_ngrams(tok_preds_uncased, 5, vocabs)

    print(res)
    
GEN_FILE = sys.argv[1]
REF_FILE = sys.argv[2]
evaluate(GEN_FILE)

