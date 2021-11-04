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

from modeling_bart import BartForConditionalGeneration, VQBART, PriorTransformer

from data import get_dataset, PairDataset, CodeDataset, make_logdir


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






def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    datasets = get_dataset(tokenizer, args.dataset_path)

    print("Build inputs and labels")
    if not args.do_generate:
        train_dataset = CodeDataset(datasets["train"], tokenizer, args.latent_vocab_size, args.max_src_len, args.max_tgt_len)
        if args.valid_sample_N != -1:
            datasets["valid"] = dict([(k, v[:args.valid_sample_N]) for k,v in datasets["valid"].items()])
        
        valid_dataset = CodeDataset(datasets["valid"], tokenizer, args.latent_vocab_size, args.max_src_len, args.max_tgt_len)
        

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.workers, shuffle=(not args.distributed), pin_memory=False)
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.workers, shuffle=False, pin_memory=False)
        

        print("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset[0][0].shape))
        print("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset[0][0].shape))
        test_loader, test_dataset = None, None

    else:
        if args.test_sample_N != -1:
            datasets["test"] = dict([(k, v[:args.test_sample_N]) for k, v in datasets["test"].items()])
        test_dataset = CodeDataset(datasets["test"], tokenizer, args.latent_vocab_size, args.max_src_len, args.max_tgt_len)
        test_loader = DataLoader(test_dataset, batch_size=args.valid_batch_size, num_workers=args.workers, shuffle=False, pin_memory=False)
        print("Test dataset (Batch, Candidates, Seq length): {}".format(test_dataset[0][0].shape))
        train_loader, train_dataset, valid_loader, valid_dataset = None, None, None, None
        

    
    return (train_loader, train_dataset), (valid_loader, valid_dataset), (test_loader, test_dataset)


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--pretrained_posterior_path", type=str, default="", help="Path to the pretrained vae model")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--tb_log_dir", type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_src_len", type=int, default=64, help="Length of source text")
    parser.add_argument("--max_tgt_len", type=int, default=32, help="Length of target text")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--valid_metric", type=str, default="loss", help="Supported metrics: loss")
    parser.add_argument("--alpha", type=float, default=0.25, help="Coefficient of avg entropy regularization")
    parser.add_argument("--latent_vocab_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--save_every_n", type=int, default=-1)
    parser.add_argument("--load_code_book", action="store_true")
    parser.add_argument("--soft_target", action="store_true")
    parser.add_argument("--load_full", action="store_true")
    parser.add_argument("--load_decoder", action="store_true")
    parser.add_argument("--do_generate", action='store_true')
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--topp", type=float, default=0.9)
    parser.add_argument("--min_length_ratio", type=float, default=0.7)
    parser.add_argument("--rep_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
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

    torch.autograd.set_detect_anomaly(True)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. print => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    print("Arguments: %s", pformat(args))

    # creating directories is not save for multiprocessing
    #log_dir = make_logdir(args.save_dir, args.model_name)
    log_dir = os.path.join(args.save_dir, args.model_name)
    args.distributed = (args.local_rank != -1)
    if not args.do_generate:
        tb_logger = SummaryWriter(os.path.join(args.tb_log_dir, args.model_name))
        hparams = {"lr": args.lr, "bsz": args.train_batch_size, "grad_accum": args.gradient_accumulation_steps, "epochs": args.n_epochs, "distributed": args.distributed, "src_len": args.max_src_len, "tgt_len": args.max_tgt_len} 
        metrics_dict = {"hparam/loss": 0}
        tb_logger.add_hparams(hparams, metrics_dict)

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
    config.dropout = args.dropout

    model = PriorTransformer(config, latent_num=args.latent_vocab_size)
    if args.do_generate:
        state_dict = torch.load(os.path.join(args.model_checkpoint, "pytorch_model.bin"), map_location=args.device)
        model.load_state_dict(state_dict)
    else:
        if not args.load_full:
            model.from_pretrained(args.model_checkpoint, args.pretrained_posterior_path, load_code_book=args.load_code_book, load_decoder=args.load_decoder)#model_class.from_pretrained(args.model_checkpoint)
        else:
            state_dict = torch.load(args.model_checkpoint, map_location="cpu")
            model.load_state_dict(state_dict)
    
    model.to(args.device)
    
    # Add special tokens if they are not already added
    #add_special_tokens_(model, tokenizer)

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True, weight_decay=args.weight_decay)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    print("Prepare datasets")
    (train_loader, train_dataset), (valid_loader, valid_dataset), (test_loader, test_dataset) = get_data_loaders(args, tokenizer)

    def avg_entropy(logits, attention_mask):
        prob = nn.Softmax(-1)(logits)
        attention_mask = (attention_mask == 0)
        masked_prob = prob.masked_fill(attention_mask.unsqueeze(-1).expand_as(prob), 0)
        masked_prob_avg = masked_prob.view(-1, masked_prob.size(-1)).mean(0)
        return - (masked_prob_avg * masked_prob_avg.log()).sum(-1)
        

    # Training function and trainer
    def update(engine, batch):
        iteration = engine.state.iteration

        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, attention_mask, decoder_input_ids, labels = batch

        outputs = model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels
        )
        loss = outputs.loss
        loss /= args.gradient_accumulation_steps
            
        if False:#args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        tb_logger.add_scalar("train_loss", loss, iteration)

        return loss.item()
    trainer = Engine(update)

    def finalize(tokens, bos=str(args.latent_vocab_size), eos=str(args.latent_vocab_size+1), pad=str(args.latent_vocab_size+2)):
        text = " ".join(tokens)
        text = re.sub(eos, "", text)
        text = re.sub(pad, "", text)
        text = re.sub(bos, "", text)

        return text.split()

                
    def generate(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, attention_mask = batch
            max_tgt_len = args.max_tgt_len
            
            if hasattr(model, 'module'):
                outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, min_length=int(max_tgt_len * args.min_length_ratio), repetition_penalty=args.rep_penalty,
                    max_length=max_tgt_len, do_sample=True, num_beams=args.num_beams, temperature=args.temperature, decoder_start_token_id=args.latent_vocab_size,
                    bos_token_id=args.latent_vocab_size, eos_token_id=args.latent_vocab_size+1, pad_token_id=args.latent_vocab_size+2)
            else:
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, min_length=int(max_tgt_len * args.min_length_ratio), repetition_penalty=args.rep_penalty,
                    max_length=max_tgt_len, do_sample=True, num_beams=args.num_beams, temperature=args.temperature, decoder_start_token_id=args.latent_vocab_size,
                    bos_token_id=args.latent_vocab_size, eos_token_id=args.latent_vocab_size+1, pad_token_id=args.latent_vocab_size+2)
            
            hypos = [finalize([str(y) for y in x]) for x in outputs.tolist()]
            
            return hypos
    generator = Engine(generate)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, attention_mask, decoder_input_ids, labels = batch
            #print(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            outputs = model(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels
            )
            lm_logits = outputs.logits
            lm_logits_flat_shifted = lm_logits.view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = labels.view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted
        
    if args.valid_metric == "loss":
        evaluator = Engine(inference)


    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch

    
    if args.do_generate:
        pbar = ProgressBar(persist=True)
        pbar.attach(generator)

        test_hypos = []
        generator.add_event_handler(Events.ITERATION_COMPLETED, lambda x: test_hypos.extend(x.state.output))
        generator.run(test_loader)
        
        with open(os.path.join(log_dir, "codes.txt"), "w") as f:
            for line in test_hypos:
                if type(line) == list:
                    line = " ".join(line)
                f.write(line+'\n')
        return 


    warmup_steps = int(args.warmup_ratio * args.n_epochs * len(train_loader))
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, 0.0), (warmup_steps, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    

    best_valid = {"loss": 10000.0}
    
    valid_refs = json.loads(open(args.dataset_path, "r").read())["valid"]["tgt"]
    if args.valid_sample_N != -1:
        valid_refs = valid_refs[:args.valid_sample_N]

    

    
    def save_best_valid(engine):
        epoch = engine.state.epoch
        iteration = engine.state.iteration

        do_save = False
        valid_result = 0


        if args.valid_metric == "loss":
            loss_metric = Loss(torch.nn.CrossEntropyLoss(ignore_index=args.latent_vocab_size + 2), output_transform=lambda x: (x[0], x[1]))
            loss_metric.attach(evaluator, "loss")
            evaluator.run(valid_loader)
            valid_loss = evaluator.state.metrics["loss"]
            print(f"\nEpoch: {epoch} Iteration: {iteration} - loss: {valid_loss}\n")
            if valid_loss < best_valid["loss"]:
                #do_save = True
                valid_result = valid_loss
            
            tb_logger.add_scalar("valid_loss", valid_loss, iteration)
                
        if args.save_last:
            do_save = True
            save_ep = None
        
        if args.save_every_n != -1:
            if epoch % args.save_every_n == 0 and epoch != 0:
                do_save = True
                save_ep = epoch

        if do_save:
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(log_dir, save_ep=save_ep)
            model_to_save.config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
            tokenizer.save_pretrained(log_dir)
            print("Saving model checkpoint to %s", log_dir)
            best_valid[args.valid_metric] = valid_result

        

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        #evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        

        if args.valid_iterations == -1:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, save_best_valid)
        else:
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=args.valid_iterations), save_best_valid)
    
        
        
    
    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)


if __name__ == "__main__":
    train()
