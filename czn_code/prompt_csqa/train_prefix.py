import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import json
import wandb

from model import Prefix_GPT

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
parser.add_argument('--model_dir', default='./model/prefix/')
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--seq_len",type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epoch",type=int, default=100)
parser.add_argument("--lr",type=float, default=1e-4)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--flat", action="store_true")
parser.add_argument("--inter_dim", type=int, default=768)

args = parser.parse_args()

if args.wandb:
    wandb.login()
    wandb.init(project="prefix", entity="happygu")


decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token


pos_dataset = {'sent':[],'type':[]}
neg_dataset = {'sent':[],'type':[]}
with open('data/SST/pos.txt', 'r') as f:
    for line in f.readlines():
        pos_dataset['sent'].append(line.strip())
        pos_dataset['type'].append(1)

with open('data/SST/neg.txt', 'r') as f:
    for line in f.readlines():
        neg_dataset['sent'].append(line.strip())
        neg_dataset['type'].append(0)

pos_train_dataset = Dataset.from_dict(pos_dataset)
neg_train_dataset = Dataset.from_dict(neg_dataset)

pos_train_dataset = pos_train_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
neg_train_dataset = neg_train_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)


pos_train_dataset.set_format(type='torch', columns=['type', 'input_ids', 'attention_mask'])
neg_train_dataset.set_format(type='torch', columns=['type', 'input_ids', 'attention_mask'])

training_args = TrainingArguments(
    output_dir=args.model_dir+'pos/',
    learning_rate=args.lr,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=False,
    no_cuda=args.no_cuda,
    save_strategy="steps",
    save_steps=200,
    fp16=args.fp16,
    report_to='wandb' if args.wandb else 'none'
)

model = Prefix_GPT(decoder=decoder, args=args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pos_train_dataset
)
train_out = trainer.train()



training_args = TrainingArguments(
    output_dir=args.model_dir+'neg/',
    learning_rate=args.lr,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=False,
    no_cuda=args.no_cuda,
    save_strategy="steps",
    save_steps=200,
    fp16=args.fp16,
    report_to='wandb' if args.wandb else 'none'
)

model = Prefix_GPT(decoder=decoder, args=args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=neg_train_dataset
)
train_out = trainer.train()
