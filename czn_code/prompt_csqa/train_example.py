import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import jsonlines
import wandb

# from model import AE


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
parser.add_argument('--model_dir', default='./model/')
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--latent_size", type=int, default=768)
parser.add_argument("--latent_num",type=int, default=1)
parser.add_argument("--seq_len_per_latent",type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epoch",type=int, default=600)
parser.add_argument("--lr",type=float, default=1e-4)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--no_fix", action="store_true")
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--contrasitive_loss", type=float, default=None)
parser.add_argument("--sparse_loss", type=float, default=None)
parser.add_argument("--latent_dis_loss", type=float, default=None)
parser.add_argument("--logit_dis_loss", type=float, default=None)

args = parser.parse_args()

if args.wandb:
    wandb.login('happygu')
    wandb.init(project="my-test-project")

# encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
# encoder = BertModel.from_pretrained(args.pretrained_encoder)
# decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
# decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
# decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')



# if not args.no_fix:
#     model.fix_decoder()


# if 'contrasitive_loss' in loss_list:
#     dataset = {'sent':[], 'type':[], 'adv_sent':[]}
# else:
#     dataset = {'sent':[],'type':[]}
dataset = {'question':[],'fact':[]}
with jsonlines.open('/users5/znchen/Question2Knowledge/SearchQasP/obqa_Additional/train_complete.jsonl', 'r') as f:
    for line in f:
        # print(line)
        # exit()
        question = line['question']['stem']
        fact = line['fact1']
        dataset['question'].append(question)
        dataset['fact'].append(fact)

# with open('data/SST/neg.txt', 'r') as f:
#     for line in f.readlines():
#         dataset['sent'].append(line.strip())
#         dataset['type'].append(0)

train_dataset = Dataset.from_dict(dataset)

train_dataset = train_dataset.map(lambda e: tokenizer(e['question'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
# train_dataset = train_dataset.rename_columns({'input_ids':'encoder_input_ids', 'attention_mask':'encoder_attention_mask', 'token_type_ids':'encoder_token_type_ids'})
# train_dataset = train_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
# train_dataset = train_dataset.rename_columns({'input_ids':'decoder_input_ids', 'attention_mask':'decoder_attention_mask'})
# if 'contrasitive_loss' in loss_list:
#     train_dataset = train_dataset.map(lambda e: encoder_tokenizer(e['adv_sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
#     train_dataset = train_dataset.rename_columns({'input_ids':'adv_input_ids', 'attention_mask':'adv_attention_mask', 'token_type_ids':'adv_token_type_ids'})
#     train_dataset.set_format(type='torch', columns=['encoder_input_ids', 'encoder_attention_mask', 'encoder_token_type_ids', 'type', 'decoder_input_ids', 'decoder_attention_mask',\
#         'adv_input_ids', 'adv_attention_mask', 'adv_token_type_ids'])
# else:
    # train_dataset.set_format(type='torch', columns=['encoder_input_ids', 'encoder_attention_mask', 'encoder_token_type_ids', 'type', 'decoder_input_ids', 'decoder_attention_mask'])
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'fact'])


training_args = TrainingArguments(
    output_dir=args.model_dir,
    learning_rate=args.lr,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=False,
    no_cuda=args.no_cuda,
    save_strategy="steps",
    save_steps=500,
    fp16=args.fp16,
    report_to='wandb' if args.wandb else 'none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
train_out = trainer.train()