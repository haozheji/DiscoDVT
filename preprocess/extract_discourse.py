import requests
import json
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from urllib.parse import quote
import sys


PORT = 12345
BOUND_RELAX = 2
MAX_TOK = 512

en_dependency_patterns = {
  # S2 ~ S2 head (full S head) ---> connective
  "after": [
    {"S1": "advcl", "S2": "mark"},
  ],
  "also": [
    {"S1": "advcl", "S2": "advmod"},
  ],
  "although": [
    {"S1": "advcl", "S2": "mark"},
  ],
  "and": [
    {"S1": "conj", "S2": "cc"},
  ],
  "as": [
    {"S1": "advcl", "S2": "mark"},
  ],
  "before": [
    {"S1": "advcl", "S2": "mark"},
  ],
  "so": [
    # {"S1": "parataxis", "S2": "dep", "POS": "IN", "flip": True},
    {"S1": "advcl", "S2": "mark"},
  ],
  "still": [
    #{"S1": "parataxis", "S2": "advmod", "POS": "RB", "acceptable_order": "S1 S2"},
    {"S1": "dep", "S2": "advmod"},
  ],
  "then": [
    {"S1": "parataxis", "S2": "advmod"},
  ],
  "because": [
    {"S1": "advcl", "S2": "mark"},
  ]
}

def corenlp_serve(sentences, utf8=True):
    url = "http://localhost:" + str(PORT) + "?properties={annotators:'tokenize,ssplit,pos,depparse'}"
    sentences = " ".join(sentences.split()[:MAX_TOK])
    if utf8:
        sentences = sentences.encode("utf-8")
    
    parse_string = requests.post(url, data=quote(sentences)).text
    parse_string = parse_string.replace('\r\n', '')
    parse_string = parse_string.replace('\x19', '')
    #parse_string = parse_string.strip("'<>() ").replace('\'', '\"')
    try:
        parsed_output = json.loads(parse_string, strict=False)
    except:
        print("Json parsing error. Skip example ", sentences)
        return None
        #raise RuntimeError("Json decode error", sentences)
    
    return parsed_output["sentences"]


class DiscourseLabeler():
    def __init__(self, sentences, parsed_sents):
        self.ori_sentences = sentences
        self.parsed_sents = parsed_sents
    
    def tokenize(self, toks):
        return [tok["word"] for tok in toks]

    def proc(self):
        label_res = []
        prev_seg_start = 0
        for parsed_sent in self.parsed_sents:
            label_res_intra = []
            label_res_inter = []
            deps = parsed_sent["basicDependencies"]
            toks = parsed_sent["tokens"]
            mark_span = self.proc_single(deps, toks)
            no_label = True
            if len(mark_span) > 0:

                for mark, span_cands in mark_span.items():
                    prev_seg_start_temp = toks[0]["characterOffsetBegin"]
                    for span in span_cands:
                        if type(span) is tuple:
                            tokens = self.tokenize(toks)
                            arg1_idx, arg2_idx, _mark_idx = span
                            arg1_span = self.get_subtree_span(arg1_idx + 1, deps)
                            arg2_span = self.get_subtree_span(arg2_idx + 1, deps)
                            
                            #print("sent: ", tokens)
                            #print("arg1: ", tokens[arg1_span[0]-1:arg1_span[1]])
                            #print("arg2: ", tokens[arg2_span[0]-1:arg2_span[1]])
                            #print()

                            label = ""


                            if (arg1_span[0] == 1 and arg1_span[1] == len(toks)):
                                # Corner case: nearly complete
                                if abs(arg2_span[1] - arg2_span[0] + 1 - len(toks)) <= BOUND_RELAX:
                                    label_res_inter.append((mark+"12", (prev_seg_start, toks[0]["characterOffsetBegin"], toks[-1]["characterOffsetEnd"])))
                                    no_label = False
                                    continue

                                if abs(arg2_span[0] - 1) <= BOUND_RELAX:
                                    label = mark + "21"
                                    try:
                                        mark_idx = toks[arg2_span[1]]["characterOffsetBegin"]
                                    except:
                                        raise RuntimeError("Index out of bound", (self.tokenize(toks), arg2_span[1]))
                                    mark_tok_idx = arg2_span[1]
                                elif abs(arg2_span[1] - len(toks)) <= BOUND_RELAX:
                                    label = mark + "12"
                                    mark_idx = toks[arg2_span[0] - 1]["characterOffsetBegin"]
                                    mark_tok_idx = arg2_span[0] - 1
                                else:
                                    continue

                                balancing_coeff = abs(1 - mark_tok_idx / (len(toks) - mark_tok_idx))

                                # (segment start index, mark start index, segment end index)
                                label_res_intra.append((label, (toks[0]["characterOffsetBegin"], 
                                                        mark_idx,
                                                        toks[-1]["characterOffsetEnd"]),
                                                        balancing_coeff))
                                no_label = False

                            elif (arg2_span[0] == 1 and arg2_span[1] == len(toks)):
                                if abs(arg1_span[1] - arg1_span[0] + 1 - len(toks)) <= BOUND_RELAX:
                                    label_res_inter.append((mark+"12", (prev_seg_start, toks[0]["characterOffsetBegin"], toks[-1]["characterOffsetEnd"])))
                                    no_label = False
                                    continue

                                # cc case
                                if abs(arg1_span[0] - 1) <= BOUND_RELAX:
                                    label = mark + "12"
                                    mark_idx = toks[arg1_span[1]]["characterOffsetBegin"]
                                    mark_tok_idx = arg1_span[1]
                                elif abs(arg1_span[1] - len(toks)) <= BOUND_RELAX:
                                    label = mark + "21"
                                    mark_idx = toks[_mark_idx-1]["characterOffsetBegin"]
                                    mark_tok_idx = _mark_idx-1
                                else:
                                    continue
                                
                                balancing_coeff = abs(1 - mark_tok_idx / (len(toks) - mark_tok_idx))

                                # (segment start index, mark start index, segment end index, balancing_coeff)
                                # balancing_coeff = abs(1 - arg1_len / arg2_len)
                                label_res_intra.append((label, (toks[0]["characterOffsetBegin"], 
                                                        mark_idx,
                                                        toks[-1]["characterOffsetEnd"]),
                                                        balancing_coeff))
                                
                                no_label = False
                            
                        else:
                            label_res_inter.append((mark+"12", (prev_seg_start, toks[0]["characterOffsetBegin"], toks[-1]["characterOffsetEnd"])))
                            no_label = False
                        
                    
            if label_res_inter == []:
                # if no label
                label_res_inter.append(("none", (prev_seg_start, toks[0]["characterOffsetBegin"], toks[-1]["characterOffsetEnd"])))
            
            
            # There should be only one valid
            label_res.append(label_res_inter[0])

            if label_res_intra != []:
                # select more balanced segmentation
                prefer_label_intra = label_res_intra[0]
                for label_intra in label_res_intra:
                    if label_intra[-1] < prefer_label_intra[-1]:
                        prefer_label_intra = label_intra
                
                
                label_res.append(prefer_label_intra[:2])

            if no_label:    
                prev_seg_start = toks[0]["characterOffsetBegin"]
            elif label_res_intra != []:
                prev_seg_start = prefer_label_intra[1][1]

        return label_res

    def proc_single(self, deps, toks):
        detected_markers = {}
        #prev_sent = False
        
        for marker, patterns in en_dependency_patterns.items():
            
            arg2_res = self.get_arg2_dep(marker, deps)
            if arg2_res == []:
                # No discourse markers found!
                continue
            
            #print(arg2_res)
            for (arg2_dep, arg2_idx) in arg2_res:
            
                arg1_res = self.get_arg1_dep(arg2_dep["governor"], marker, deps)
            
                if arg1_res == None:
                    # No discourse markers found!
                    continue

                elif arg1_res == True:
                    detected_markers[marker] = [True]

                else:
                    arg1_dep, arg1_idx = arg1_res


                    if marker not in detected_markers:
                        detected_markers[marker] = [(arg1_idx, arg2_idx, arg2_dep["dependent"])]
                    else:
                        detected_markers[marker].append((arg1_idx, arg2_idx, arg2_dep["dependent"]))

        
        return detected_markers

    
    def get_arg2_dep(self, marker, deps):
        res = []
        for dep in deps:
            # match marker type
            if dep["dependentGloss"].lower() == marker:
                
                # match dependency type
                for e in en_dependency_patterns[marker]:
                    if e["S2"] == dep["dep"]:
                        #print(e)
                        res.append((dep, dep["governor"] - 1))
    
        return res

    def get_arg1_dep(self, arg2_dep_idx, marker, deps):
        for dep in deps:
            # match arg2 idx (child)
            # exclude cc
            if dep["dependent"] == arg2_dep_idx and marker not in ["and", "but"]:
                
                # match dependency type
                for e in en_dependency_patterns[marker]:
                    if e["S1"] == dep["dep"]:
                        
                        return dep, dep["governor"] - 1
                
                # Corner case: arg1's father is ROOT
                # Probably previous sentence is arg1
                if dep["dep"] == "ROOT":
                    return True
                    
            # match arg2 idx (father)
            # cc case
            elif dep["governor"] == arg2_dep_idx:
                
                # match dependency type
                for e in en_dependency_patterns[marker]:
                    if e["S1"] == dep["dep"]:
                        
                        return dep, dep["dependent"] - 1   
            
        return None

    def get_subtree_span(self, root, deps):
        children = []
        level_start, level_end = root, root
        for dep in deps:
            if dep["governor"] == root:
                start, end = self.get_subtree_span(dep["dependent"], deps)
                level_start = min(start, level_start)
                level_end = max(end, level_end)
        
        return level_start, level_end


def pack_up(res, sentences):
    if res == []:
        return {"edus": [sentences.strip()], "labels": []}
    prev_mark = 0
    segs = []
    labels = []
    for (label, spans) in res[1:]:
        s,m,e = spans
        segs.append(sentences[prev_mark:m].strip())
        prev_mark = m
        labels.append(label)
    
    segs.append(sentences[prev_mark:])

    assert(len(segs) == len(labels) + 1)

    return {"edus": segs, "labels": labels}


def extract_single(paragraph):
    parsed_res = corenlp_serve(paragraph)
    if parsed_res is None:
        return pack_up([], paragraph)
    disc = DiscourseLabeler(paragraph, parsed_res)
    res = disc.proc()
    pack_res = pack_up(res, paragraph)
    return pack_res

def mproc(parser, args, num_proc=5):
    pool = Pool(processes=num_proc)

    jobs = [pool.apply_async(func=parser, args=(arg,)) for arg in args]
    pool.close()

    res_list = []
    for job in tqdm(jobs):
        res_list.append(job.get())
    
    return res_list

def read_json(filename):
    return json.load(open(filename, "r"))

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

DATA_NAME = sys.argv[1]
NUM_PROC = sys.argv[2] if len(sys.argv) > 2 else 10
SPLITS = ["train", "valid", "test"]
#SPLIT = "train"

DATA_DIR = f"../data/{DATA_NAME}.json"
#SAVE_DIR = f"../data/{DATA_NAME}_{SPLIT}_disc.json"

def main(split_name):
    data = read_json(DATA_DIR)
    paras = data[split_name]["tgt"]

    parsed_paras = mproc(extract_single, paras, num_proc=20)
    
    save_json(parsed_paras, f"../data/{DATA_NAME}_{split_name}_disc.json")
    


if __name__ == "__main__":
    for split_name in SPLITS:
        main(split_name)