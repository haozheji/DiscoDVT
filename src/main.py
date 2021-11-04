# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import json
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import nltk
from nltk.tokenize import word_tokenize
from tensorboardX import SummaryWriter

import re
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
#from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
#                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from transformers import (AdamW, BartTokenizer, BartConfig, WEIGHTS_NAME, CONFIG_NAME)

from modeling_bart import BartForConditionalGeneration, DiscoDVT

from data import get_dataset, PairDataset, make_logdir


GUMBEL_ANNEAL_RATE = 5e-5
MIN_TEMP = 0.1
START_TEMP = 0.9

ALPHA_ANNEAL_RATE = 5e-4

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def set_seed(args):
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    #if args.local_rank > 0:
    torch.cuda.manual_seed_all(args.seed)



def nltk_bleu(refs, hypos, order=None):
    refs = [word_tokenize(x.strip()) for x in refs]
    hypos = [word_tokenize(x.strip()) for x in hypos]
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


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    datasets = get_dataset(tokenizer, args.dataset_path)

    print("Build inputs and labels")

    train_dataset = PairDataset(datasets["train"], args.latent_vocab_size, tokenizer, args.max_src_len, args.max_tgt_len, no_none_loss=args.no_none_loss)
    if args.valid_sample_N != -1:
        datasets["valid"] = dict([(k, v[:args.valid_sample_N]) for k,v in datasets["valid"].items()])
    valid_dataset = PairDataset(datasets["valid"], args.latent_vocab_size, tokenizer, args.max_src_len, args.max_tgt_len, no_none_loss=args.no_none_loss)

    if args.test_sample_N != -1:
        if args.generate_with_code:# or args.decode_code:
            assert(len(datasets["test"]["code"]) >= args.test_sample_N), "#Test samples {} should not be larger than the #latent code samples {}".format(args.test_sample_N, len(datasets["test"]["code"]))
        datasets["test"] = dict([(k, v[:args.test_sample_N]) for k,v in datasets["test"].items()])
    

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    #valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.workers, shuffle=(not args.distributed and not args.decode_code), pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.workers, shuffle=False, pin_memory=False)
    
    if args.do_generate:# or args.decode_code:
        test_dataset = PairDataset(datasets["test"], args.latent_vocab_size, tokenizer, args.max_src_len, args.max_tgt_len, generate_with_code=args.generate_with_code, no_none_loss=args.no_none_loss)
        test_loader = DataLoader(test_dataset, batch_size=args.valid_batch_size, num_workers=args.workers, shuffle=False, pin_memory=False)
    else:
        test_dataset = None
        test_loader = None

    print("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset[0][0].shape))
    print("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset[0][0].shape))
    if args.do_generate:
        print("Test dataset (Batch, Candidates, Seq length): {}".format(test_dataset[0][0].shape))
    return (train_loader, train_dataset), (valid_loader, valid_dataset), (test_loader, test_dataset)


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--tb_log_dir", type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--relation_num", type=int, default=20, help="number of discourse relation")
    parser.add_argument("--no_none_loss", action="store_true", help="whether compute loss on none relation pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_src_len", type=int, default=64, help="Length of source text")
    parser.add_argument("--max_tgt_len", type=int, default=32, help="Length of target text")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--valid_metric", type=str, default="loss", help="Supported metrics: loss, bleu")
    parser.add_argument("--save_every_eval", action='store_true')
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.25, help="Coefficient of avg entropy regularization")
    parser.add_argument("--alpha_anneal", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.0, help="Coefficient of discourse preserving loss")
    parser.add_argument("--decode_code", type=str, default="", help="Decode code from train or valid")
    parser.add_argument("--latent_vocab_size", type=int, default=512)
    parser.add_argument("--gumbel_trick", action="store_true")
    parser.add_argument("--gumbel_anneal", action="store_true")
    parser.add_argument("--gumbel_anneal_rate", type=float, default=GUMBEL_ANNEAL_RATE)
    parser.add_argument("--start_temp", type=float, default=START_TEMP)
    parser.add_argument("--end_temp", type=float, default=MIN_TEMP)
    parser.add_argument("--do_generate", action='store_true')
    parser.add_argument("--min_length", type=int, default=100)
    parser.add_argument("--rep_penalty", type=float, default=1.0)
    parser.add_argument("--generate_with_code", action='store_true')
    parser.add_argument("--topp", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_cnn_layers", type=int, default=2)
    parser.add_argument("--only_train_bottleneck", action="store_true")
    parser.add_argument("--load_from_full", action="store_true")
    parser.add_argument("--constant_lr", action="store_true")
    parser.add_argument("--code_dropout", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers for batching")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--valid_iterations", type=int, default=-1, help="Perform validation every N iterations (-1 defaulting to epoch end)")
    parser.add_argument("--valid_sample_N", type=int, default=-1, help="Number of samples used for validation (-1 defaulting to full)")
    parser.add_argument("--test_sample_N", type=int, default=-1, help="Number of samples used for test (-1 defaulting to full)")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. print => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    print("Arguments: %s", pformat(args))

    # creating directories is not save for multiprocessing
    log_dir = os.path.join(args.save_dir, args.model_name)
    args.distributed = (args.local_rank != -1)
    if not args.do_generate and not args.decode_code:
        tb_logger = SummaryWriter(os.path.join(args.tb_log_dir, args.model_name))
        
    # set seed
    set_seed(args)

    # Initialize distributed training if needed
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    print("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = BartTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)

    config = BartConfig.from_pretrained(args.model_checkpoint)

    model = DiscoDVT(config, latent_num=args.latent_vocab_size, straight_through=(not args.gumbel_trick), num_cnn_layers=args.num_cnn_layers, code_dropout=args.code_dropout, relation_num=args.relation_num)
    if args.do_generate or args.decode_code or args.load_from_full:
        if args.decode_code != "":
            assert(args.decode_code == "train" or args.decode_code == "valid" or args.decode_code == "test")
            print("Decoding latent code of {} set".format(args.decode_code))
        state_dict = torch.load(os.path.join(args.model_checkpoint, "pytorch_model.bin"), map_location=args.device)
        
        # allow initialized parameters
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        # resize code book embedding size for extra padding
        if args.do_generate and args.generate_with_code:
            model.resize_code_book_for_generation()
            print("Resize codebook for generation !")
    else:
        model.from_pretrained(args.model_checkpoint)#model_class.from_pretrained(args.model_checkpoint)
    
    model.to(args.device)
    
    if args.only_train_bottleneck:
        print("Only update params of the bottleneck!")
        for p in model.decoder.parameters():
            p.requires_grad = False
        for p in model.encoder.parameters():
            p.requires_grad = False
        free_params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        free_params = model.parameters()
    optimizer = AdamW(free_params, lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    print("Prepare datasets")
    (train_loader, train_dataset), (valid_loader, valid_dataset), (test_loader, test_dataset) = get_data_loaders(args, tokenizer)

    def avg_entropy(logits, attention_mask):
        prob = nn.Softmax(-1)(logits)
        # size adaptive
        bsz, length, _ = prob.size()
        attention_mask = attention_mask.view(bsz, length, -1).max(-1)[0]
        attention_mask = (attention_mask == 0)
        masked_prob = prob.masked_fill(attention_mask.unsqueeze(-1).expand_as(prob), 0.0)
        
        masked_prob_avg = prob.mean(1) # bsz, code_size
        return - (masked_prob_avg * masked_prob_avg.log()).sum(-1).mean()
        '''
        masked_prob_avg = masked_prob.view(-1, masked_prob.size(-1)).mean(0)
        return - (masked_prob_avg * masked_prob_avg.log()).sum(-1)
        '''
        
    
    def total_variation(one_hot, attention_mask):
        bsz, length, K = one_hot.size()
        v = torch.arange(0, K).float().to(one_hot.device)
        signal = one_hot @ v

        attention_mask = attention_mask.view(bsz, length, -1).max(-1)[0]
        attention_mask = (attention_mask == 0)
        masked_signal = signal.masked_fill(attention_mask, 0)
        return (masked_signal[:,1:] - masked_signal[:,:-1]).abs().mean()
        

    # Training function and trainer
    def update(engine, batch):
        iteration = engine.state.iteration

        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        if len(batch) == 7:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, discourse_pos, discourse_labels, labels = batch
        else:
            # for warm-start
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch
            discourse_pos, discourse_labels = None, None

        gumbel_temp = None
        if args.gumbel_anneal:
            gumbel_temp = max(args.end_temp, args.start_temp * math.exp(- args.gumbel_anneal_rate * iteration))
        else:
            gumbel_temp = args.start_temp

        outputs = model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, 
            decoder_attention_mask=decoder_attention_mask, discourse_pos=discourse_pos, discourse_labels=discourse_labels,
            labels=labels, temperature=gumbel_temp
        )
        lm_loss = outputs[0] / args.gradient_accumulation_steps
        logits = outputs[1] 
        one_hot = outputs[2]
        if outputs[3] != None:
            disc_loss = outputs[3] / args.gradient_accumulation_steps
        else:
            disc_loss = 0.0 

        entropy = avg_entropy(logits, decoder_attention_mask) / args.gradient_accumulation_steps
        if args.alpha_anneal:
            alpha = args.alpha * math.exp(- ALPHA_ANNEAL_RATE * iteration)
        else:
            alpha = args.alpha
        loss = lm_loss - alpha * entropy + args.gamma * disc_loss
            
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        tb_logger.add_scalar("train_loss", lm_loss, iteration)
        tb_logger.add_scalar("entropy", entropy, iteration)
        tb_logger.add_scalar("train_disc_loss", disc_loss, iteration)
        
        if args.gumbel_anneal:
            tb_logger.add_scalar("gumbel_temperature", gumbel_temp, iteration)
        
        tb_logger.add_scalar("alpha", alpha, iteration)

        return loss.item()

        

    trainer = Engine(update)
    def generate(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            if args.generate_with_code:
                input_ids, attention_mask, code = batch
                decoder_attention_mask = None
            else:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, discourse_pos, discourse_labels, labels = batch
                code = model.get_codebook_indices(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
            #bsz, latent_len = code.size()
            #decoder_attention_mask = decoder_attention_mask.view(bsz, latent_len, -1).max(-1)[0]
            #decoder_attention_mask = (decoder_attention_mask == 0)
            #code = code.masked_fill(decoder_attention_mask, args.latent_vocab_size).long()
            
            if hasattr(model, 'module'):
                outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, 
                    latent_codes=code, do_sample=True, num_beams=1, use_cache=True, min_length=args.min_length, repetition_penalty=args.rep_penalty,
                    max_length=args.max_tgt_len, top_p=args.topp, temperature=args.temperature, decoder_start_token_id=0)#, early_stopping=True) # <s>: 0
            else:
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, 
                    latent_codes=code, do_sample=True, num_beams=1, use_cache=True, min_length=args.min_length, repetition_penalty=args.rep_penalty,
                    max_length=args.max_tgt_len, top_p=args.topp, temperature=args.temperature, decoder_start_token_id=0)#, early_stopping=True) # <s>: 0
            
            hypos = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            #print(hypos)
            clean_hypos = []
            for hypo, inp in zip(hypos, inputs):
                #clean_hypo = finalize(hypo).strip()
                clean_hypos.append(hypo.strip())
                print("[S]: {}".format(inp))
                print("[H]: {}".format(hypo))
            
            return clean_hypos
    generator = Engine(generate)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            if len(batch) == 7:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, discourse_pos, discourse_labels, labels = batch
            else:
                # for warm-start
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch
                discourse_pos, discourse_labels = None, None
            
            outputs = model(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask, discourse_pos=discourse_pos, 
                use_cache=False
            )
            lm_logits = outputs[0]

            lm_logits_flat_shifted = lm_logits.view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = labels.view(-1)
            
            return lm_logits_flat_shifted, lm_labels_flat_shifted
        
    def generate_code(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, discourse_pos, discourse_labels, labels = batch
            code_ids = model.get_codebook_indices(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)

            clean_code_ids = []
            for i, code in enumerate(code_ids):
                length = code.size(0)
                mask = decoder_attention_mask[i].view(length, -1).max(-1)[0].bool()
                code = code.masked_select(mask).tolist()
                clean_code_ids.append(code)

            return clean_code_ids
    code_generator = Engine(generate_code)


    if args.valid_metric == "loss":
        evaluator = Engine(inference)
    elif args.valid_metric == "bleu":
        evaluator = Engine(generate)


    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    if args.decode_code:
        pbar = ProgressBar(persist=True)
        pbar.attach(code_generator)

        generated_codes = []
        code_generator.add_event_handler(Events.ITERATION_COMPLETED, lambda x: generated_codes.extend(x.state.output))
        if args.decode_code == "train":
            code_generator.run(train_loader)
        elif args.decode_code == "valid":
            code_generator.run(valid_loader)
        elif args.decode_code == "test":
            code_generator.run(test_loader)



        with open(os.path.join(log_dir, f"code_{args.decode_code}.txt"), "w") as f:
            for line in generated_codes:
                f.write(" ".join([str(x) for x in line]) + "\n")

        return 

    
    if args.do_generate:
        pbar = ProgressBar(persist=True)
        pbar.attach(generator)

        test_refs = json.loads(open(args.dataset_path, "r").read())["test"]["tgt"]
        if args.test_sample_N != -1:
            test_refs = test_refs[:args.test_sample_N]
        test_hypos = []
        generator.add_event_handler(Events.ITERATION_COMPLETED, lambda x: test_hypos.extend(x.state.output))
        generator.run(test_loader)
        bleu1 = nltk_bleu(test_refs, test_hypos, order=1)
        
        print("Test BLEU1: {:.4f} ".format(bleu1))
        
        if args.generate_with_code:
            res_path = f"result.txt"
        else:
            res_path = f"result-pos.txt"
        with open(os.path.join(log_dir, res_path), "w") as f:
            for line in test_hypos:
                f.write(line+'\n')
        return 
        


    #trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))
    #if args.n_epochs < 1:
    #    trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(valid_loader))
    #if args.eval_before_start:
    #    trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(valid_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    #if args.distributed:
    #    trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
    #    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))
    
    # Linearly decrease the learning rate from lr to zero
    if args.constant_lr:
        end_lr = args.lr
    else:
        end_lr = 0.0
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), end_lr)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    
    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    #metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=1), output_transform=lambda x: (x[0], x[1]))}
    #metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    #metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    #for name, metric in metrics.items():
    #    metric.attach(evaluator, name)

    best_valid = {"loss": 10000.0, "bleu": 0.0}
    
    valid_refs = json.loads(open(args.dataset_path, "r").read())["valid"]["tgt"]
    if args.valid_sample_N != -1:
        valid_refs = valid_refs[:args.valid_sample_N]

    

    
    def save_best_valid(engine):
        epoch = engine.state.epoch
        iteration = engine.state.iteration

        do_save = False
        valid_result = 0
        valid_hypos = []
        if args.valid_metric == "bleu":
            evaluator.add_event_handler(Events.ITERATION_COMPLETED, lambda x: valid_hypos.extend(x.state.output))
            evaluator.run(valid_loader)
            bleu = nltk_bleu(valid_refs, valid_hypos)
            print(f"\nEpoch: {epoch} Iteration: {iteration} - bleu: {bleu}\n")
            if bleu > best_valid["bleu"]:
                do_save = True
                valid_result = bleu
            
            with open(os.path.join(log_dir, "result_val:ep{}iter{}.txt".format(epoch, iteration)), "w") as f:
                for line in valid_hypos:
                    f.write(line+'\n')

            tb_logger.add_scalar("bleu", bleu, iteration)


        elif args.valid_metric == "loss":
            loss_metric = Loss(torch.nn.CrossEntropyLoss(ignore_index=1), output_transform=lambda x: (x[0], x[1]))
            loss_metric.attach(evaluator, "loss")

            evaluator.run(valid_loader)
            valid_loss = evaluator.state.metrics["loss"]
            print(f"\nEpoch: {epoch} Iteration: {iteration} - loss: {valid_loss} \n")
            if valid_loss < best_valid["loss"]:
                do_save = True
                valid_result = valid_loss
            
            tb_logger.add_scalar("valid_loss", valid_loss, iteration)
            
        save_step = None
        if args.save_every_eval:
            do_save = True
            save_step = iteration
        if args.save_last:
            do_save = True
        if do_save:
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(log_dir, save_step=save_step)
            #orch.save(args, os.path.join(log_dir, 'training_args.bin'))
            #torch.save(scheduler.state_dict(), os.path.join(log_dir, "scheduler.bin"))
            #torch.save(optimizer.state_dict(), os.path.join(log_dir, "optimizer.bin"))
            model_to_save.config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
            tokenizer.save_pretrained(log_dir)
            print("Saving model checkpoint to %s", log_dir)
            best_valid[args.valid_metric] = valid_result

        

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        #evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        

        #tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        #tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        #tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())))
        #tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        #checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=1)
        #trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation
        if args.valid_iterations == -1:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, save_best_valid)
        else:
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=args.valid_iterations), save_best_valid)
    
        
        
    
    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    #if args.local_rank in [-1, 0] and args.n_epochs > 0:
    #    os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
    #    tb_logger.close()

if __name__ == "__main__":
    train()
