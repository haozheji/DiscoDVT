import logging
import socket
import json
import torch
import os
from datetime import datetime
from torch.utils.data import Dataset


logger = logging.getLogger(__file__)

def make_logdir(save_dir, model_name):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(save_dir, model_name)
    return logdir



def get_dataset(tokenizer, dataset_path, cache_type="bpe"):
    '''
    Examples: BartTokenizer (Need prepend space)
    >> bart_tokenizer.encode("Hello world!")
    >> [0, 31414, 232, 328, 2] 
    # Already wrap <s> and </s> around the input

    >> bart_tokenizer.convert_tokens_to_ids(bart_tokenizer.tokenize("Hello world!"))
    >> [31414, 232, 328]
    # No space prepend 'Hello'
    # No <s> or </s> wrapped

    data format:
    - valid
        - src
            - [sent (str)]
            ...
        - tgt
            - [sent1 (str), sent2 (str)] # possible multiple references for valid and test.
            ...
    - train
        ...
    - test
        ...
    '''
    dataset_cache = dataset_path + '.' + cache_type
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.encode(' ' + obj)
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            if isinstance(obj, int):
                return obj
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


def trunc(text, max_length, trunc_where):
    if len(text) <= max_length:
        return text
    bos, text, eos = text[0], text[1:-1], text[-1]
    if trunc_where == "back":
        return [bos] + text[:max_length - 2 - len(text)] + [eos]
    if trunc_where == "front":
        return [bos] + text[len(text) - max_length + 2:] + [eos]

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, latent_vocab_size, max_src_len=64, max_tgt_len=256, trunc_where="back", soft_target=False):
        self.tokenizer = tokenizer
        self.data = data
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.trunc_where = trunc_where
        self.latent_vocab_size = latent_vocab_size
        self.soft_target = soft_target
    
    def __len__(self):
        return len(self.data["src"])
    
    def __getitem__(self, idx):
        pad_idx = self.tokenizer.encoder["<pad>"]
        latent_bos_idx = self.latent_vocab_size
        latent_eos_idx = self.latent_vocab_size + 1
        latent_pad_idx = self.latent_vocab_size + 2

        src = trunc(self.data["src"][idx], self.max_src_len, self.trunc_where)
        input_ids = src + [pad_idx] * (self.max_src_len - len(src))
        attention_mask = [1] * len(src) + [0] * (self.max_src_len - len(src))

        if self.data.get("tgt", False):
            tgt = [latent_bos_idx] + self.data["tgt"][idx] + [latent_eos_idx]
            tgt = trunc(tgt, self.max_tgt_len + 2, self.trunc_where)

            decoder_input_ids = tgt[:-1] + [latent_pad_idx] * (self.max_tgt_len + 1 - len(tgt[:-1]))
            labels = tgt[1:] + [latent_pad_idx] * (self.max_tgt_len + 1 - len(tgt[1:]))

            return (torch.tensor(input_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(decoder_input_ids),
                    torch.tensor(labels))

        return (torch.tensor(input_ids),
                torch.tensor(attention_mask))

class PairDataset(Dataset):
    def __init__(self, data, latent_vocab_size, tokenizer, max_src_len=64, max_tgt_len=32, max_disc_len=64, trunc_where="back", generate_with_code=False, no_none_loss=False):
        self.tokenizer = tokenizer
        self.data = data
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_disc_len = max_disc_len
        self.trunc_where = trunc_where # back or front
        self.generate_with_code = generate_with_code
        self.latent_vocab_size = latent_vocab_size
        self.no_none_loss = no_none_loss
    
    def __len__(self):
        return len(self.data["src"])

    def trunc_disc(self, pos):
        trunc_pos = []
        for p in pos:
            if p < self.max_tgt_len:
                trunc_pos.append(p)
            else:
                break
        
        # prevent bug: 
        # manually truncate length to max_disc_len
        return trunc_pos[:self.max_disc_len + 1]

    def __getitem__(self, idx):
        '''
        input_ids, attention_mask, labels
        '''
        pad_idx = self.tokenizer.encoder["<pad>"]

        src = trunc(self.data["src"][idx], self.max_src_len, self.trunc_where)
        input_ids = src + [pad_idx] * (self.max_src_len - len(src))
        # input_ids: <s> x1 x2 x3 </s>
        # decoder_input_ids: <s> y1 y2 y3 y4 
        # labels: y1 y2 y3 y4 </s>
        attention_mask = [1] * len(src) + [0] * (self.max_src_len - len(src))

        if self.generate_with_code:
            code = self.data["code"][idx]
            code = code[:self.max_tgt_len]
            code = code + [self.latent_vocab_size] * (self.max_tgt_len - len(code))
        else:
            tgt = trunc(self.data["tgt"][idx], self.max_tgt_len + 1, self.trunc_where) # Only one special token in the Decoder.
            decoder_input_ids = tgt[:-1] + [pad_idx] * (self.max_tgt_len - len(tgt[:-1]))
            labels = tgt[1:] + [pad_idx] * (self.max_tgt_len - len(tgt[1:]))
            decoder_attention_mask = [1] * len(tgt[:-1]) + [0] * (self.max_tgt_len - len(tgt[:-1]))
            
            if self.data.get("pos", False):
                discourse_pos = self.trunc_disc(self.data["pos"][idx] + [len(tgt) - 1]) 
                discourse_labels = self.data["labels"][idx][:len(discourse_pos) - 2]

                discourse_pos_scatter = [0]
                prev_p = discourse_pos[0]
                for i, p in enumerate(discourse_pos[1:]):
                    discourse_pos_scatter += [i] * (p - prev_p)
                    prev_p = p
                
                discourse_pos_scatter += [self.max_disc_len - 1] * (self.max_tgt_len - len(discourse_pos_scatter))
                #discourse_pos = discourse_pos + [0] * (self.max_disc_len - len(discourse_pos))
                if self.no_none_loss:
                    discourse_labels = [-1 if x == 0 else x for x in discourse_labels]
                discourse_labels = discourse_labels + [-1] * (self.max_disc_len - 1 - len(discourse_labels))
            else:
                return (torch.tensor(input_ids),
                        torch.tensor(attention_mask),
                        torch.tensor(decoder_input_ids),
                        torch.tensor(decoder_attention_mask),
                        torch.tensor(labels))

        # Only need attention mask at encoder side
        # Decoder attention mask is implemented by the BartModel
        # Example:
        # input = [This is not padded <pad> <pad>]
        # mask = [1 1 1 1 0 0]
        #print(input_ids)
        if self.generate_with_code:
            return (torch.tensor(input_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(code))

        return (torch.tensor(input_ids),
                torch.tensor(attention_mask),
                torch.tensor(decoder_input_ids),
                torch.tensor(decoder_attention_mask),
                torch.tensor(discourse_pos_scatter),
                torch.tensor(discourse_labels),
                torch.tensor(labels))


class LMDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64, trunc_where="back"):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.trunc_where = trunc_where # back or front
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pad_idx = self.tokenizer.convert_tokens_to_ids(["<pad>"])[0]
        bos_idx = self.tokenizer.encoder["<|endoftext|>"]
        eos_idx = self.tokenizer.encoder["<|endoftext|>"]

        input_ids = [bos_idx] + self.data[idx][0] + [eos_idx]

        input_ids = trunc(input_ids, self.max_len, self.trunc_where)

        input_ids = input_ids + [pad_idx] * (self.max_len - len(input_ids))
        labels = input_ids + [-100] * (self.max_len - len(input_ids))
        attention_mask = [1] * (len(input_ids)) + [0] * (self.max_len - len(input_ids))
        

        return (torch.tensor(input_ids),
                torch.tensor(attention_mask),
                torch.tensor(labels))

    