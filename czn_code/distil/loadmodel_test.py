
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import jsonlines
from torch.nn import DataParallel
import wandb
from tqdm import tqdm
import datetime
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import numpy as np
from transformers import  AdamW,get_linear_schedule_with_warmup
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    T5ForConditionalGeneration, AutoTokenizer,
    BartForConditionalGeneration, BartTokenizer,
    Trainer, TrainingArguments
)
from transformers import  AdamW

from dataloader import DistlCSQADatasetRA,DistlCSQADataset,DistlCSQADatasetR

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-medium': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-large': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-xl': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    't5-large': (T5ForConditionalGeneration, AutoTokenizer),
    'facebook/bart-large': (BartForConditionalGeneration, BartTokenizer),
}
Dataloader_classes = {
    'RA':DistlCSQADatasetRA,
    'WOCOT':DistlCSQADataset,
    'R':DistlCSQADatasetR,
    'WOCOT-base':DistlCSQADataset
}
def loadmodel(model_path,model_name):
    # 加载预训练的模型和分词器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    model = model_class.from_pretrained(model_path)
    model = model.to(device)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model.eval()
    # while(1):
    input_context = ['Question: What do animals do when an enemy is approaching? Options: A) feel pleasure, B) procreate, C) pass water, D) listen to each other, E) sing,',
    'Question: What island country is ferret popular? Options: A) own home, B) north carolina, C) great britain, D) hutch, E) outdoors,']
    max_length = 128
    input_ids = tokenizer(input_context,  padding="max_length" if max_length else "longest",return_tensors="pt").input_ids.squeeze()
    outputs = model.generate(input_ids.to(device), max_length=200, num_return_sequences=1, early_stopping=True)
    for output in outputs:
        answer = tokenizer.decode(output, skip_special_tokens=True)
        print('[out]',answer)
    return

if __name__=='__main__':
    model_path = '/users5/znchen/distil/tmp/generate_model/facebookbartlarge/R/facebookbartlarge_6'
    model_name = 'facebook/bart-large'
    loadmodel(model_path, model_name)
    