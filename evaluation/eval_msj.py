from multiset_distances import MultisetDistances
from nltk.tokenize import word_tokenize
import sys

GEN_FILE = sys.argv[1]
REF_FILE = sys.argv[2]


def read(filename):
    with open(filename, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data

refs = read(REF_FILE)
hyps = read(GEN_FILE)

refs = [word_tokenize(x) for x in refs]
hyps = [word_tokenize(x) for x in hyps]

ref_avg_len = 0
hyp_avg_len = 0
for line in refs:
    ref_avg_len += len(line)
ref_avg_len /= len(refs)
for line in hyps:
    hyp_avg_len += len(line)
hyp_avg_len /= len(hyps)

print("Reference avg length: {}".format(ref_avg_len))
print("Hypothesis avg length: {}".format(hyp_avg_len))

msd = MultisetDistances(references=refs, min_n=2, max_n=3)
msj_distance = msd.get_jaccard_score(sentences=hyps)

print("MSJ distance: {}".format(msj_distance))
