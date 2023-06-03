from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import torch
import transformers
# from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import jsonlines
#model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
# model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


trained_path = '/users5/znchen/Question2Knowledge/SearchQasP/ft/gpt/tmp/gpt_checkpoint/checkpoint-1800'
# # trained_path = './model/checkpoint-2500/'
# model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained(trained_path)
# tokenizer = GPT2Tokenizer.from_pretrained(trained_path)

tokenizer.pad_token = tokenizer.eos_token
import os
def data(dir):
    dataset = {'question_fact':[]}
    with jsonlines.open(dir, 'r') as f:
        for line in f:
            # print(line)
            # exit()
            question = line['question']['stem']
            fact = line['fact1']
            dataset['question_fact'].append(question+" " +fact)
    return dataset

def generate(sentence,prompt=False):
    input_text=sentence
    input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).input_ids
    length = 0
    if prompt==False:
        length = 50
    else:
        length = 600
    # output = model.generate(input_ids=' ', max_length = 600 ,do_sample=True, top_k=10, top_p=0.9, repetition_penalty=1.2, return_dict=True, use_cache=True)
    output = model.generate(input_ids=input_ids, max_length = length ,do_sample=True, top_k=10, top_p=0.9, repetition_penalty=1.2, return_dict=True, use_cache=True)
    # print(tokenizer.batch_decode(output))
    res = tokenizer.batch_decode(output)
    # print("res : ",res)    
    return res

def train():
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/ft/gpt/data/train.txt'
    dataset = data(data_dir)
    train_dataset = Dataset.from_dict(dataset)
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    
    train_dataset = train_dataset.map(lambda e: tokenizer(e['question_fact'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    # train_dataset = train_dataset.map(lambda e: tokenizer(e['question_fact'][1:-1], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    # print(train_dataset)
    train_dataset = train_dataset.map(lambda e: {"labels": e["input_ids"]})

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])


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
        report_to='wandb' if args.wandb else 'none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    train_out = trainer.train()

def generate_f_with_question(q_dir,datasetname,ft=True):
    # if ft==False:
    #     model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    # elif ft==True:
    #     model = GPT2LMHeadModel.from_pretrained(trained_path)
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    knowledges = []
    question_knowledge = []
    print("generating using question")
    with jsonlines.open(q_dir,'r') as f:
        samplenum = 500
        num = 0
        for line in tqdm(f):
            num+=1
            
            if num>samplenum:
                break
            question=line["question"]['stem']

            # using question to generate 
            k = generate(question,prompt=False)
            question_knowledge.append({'data':line,'knowledge':k})

    
    # with jsonlines.open('./outputfile/q_only_knowledges.txt','w') as f:
    print("begin writing")
    if ft==True:
        writeFilename = './outputfile/gen/'+datasetname + '_ftmodel_'+ 'question.json'
    else:
        writeFilename =  './outputfile/gen/'+datasetname + '_gpt2_'+ 'question.json'

    with jsonlines.open(writeFilename,'w') as f:
        for item in question_knowledge:
            f.write(item)
 
        
    return question_knowledge

def generate_f_with_fixed_p(q_dir , datasetname,ft=True):
    # if ft==False:
    #     model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    # elif ft==True:
    #     model = GPT2LMHeadModel.from_pretrained(trained_path)
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    knowledges = []
    question_knowledge = []
    print("generating using prompt")
    with jsonlines.open(q_dir,'r') as f:
        samplenum = 500
        num = 0
        for line in tqdm(f):
            num+=1
            
            if num>samplenum:
                break
            question=line["question"]['stem']

            sentence="Generate some knowledge about the concepts in the input. Examples:\n\
                    Input: Google Maps and other highway and street GPS services have replaced what?\n \
                    Knowledge: Electronic maps are the modern version of paper atlas.\n\
                    Input: The fox walked from the city into the forest, what was it looking for?\n\
                    Knowledge: Natural habitats are usually away from cities.\n\
                    Input: You can share files with someone if you have a connection to a what?\n\
                    Knowledge: Files can be shared over the Internet.\n\
                    Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
                    Knowledge: Some people raise snakes as pets.\n\
                    Input: The body guard was good at his duties, he made the person who hired him what?\n\
                    Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
                    Input: {}\n\
                    Knowledge:".format(question)
            
            # using question to generate 
            # k = generate(question)
            # question_knowledge.append({'question':question,'knowledge':k})


            # # if use prompt , you need code below to cut the prompt. remenber set model
            k = generate(sentence,prompt=True)
            l_s = len(sentence)
            # print("k   sentence",len(k[0]),len(sentence))
            only_k = k[0][l_s:]
            # print(only_k)
            for i in range(len(only_k)):
                if only_k[i]=='\n':
                    only_k = only_k[0:i+1]
                    break

            question_knowledge.append({'question':question,'knowledge':only_k})

    print("begin writing")
    if ft==True:
        writeFilename = './outputfile/gen/' + datasetname + '_ftmodel_'+ 'fixedprompt.json'
    else:
        writeFilename = './outputfile/gen/' + datasetname + '_gpt2_'+ 'fixedprompt.json'

    with jsonlines.open(writeFilename,'w') as f:
        for item in question_knowledge:
            f.write(item)
        
    return question_knowledge

# def generate_f_with_ftmodel(q_dir):
#     k_list = []
#     with jsonlines.open(q_dir,'r') as f:
#         for line in tqdm(f):
#             question = line['question']['stem']
#             #用question+fact训练的模型 用question作为输入 生成知识
#             k = generate(question)
            
            
#             k = k[len(question):]
#             k_list.append(k)
#     print("begin writing")
#     with open(datasetname+"_gened_k_by_ftmodel.json",'w') as f:
#         for item in tqdm(k_list):
#             f.write(item)
#         f.close()



if __name__=='__main__':
    
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--model", type=str, default="gpt2-medium")
    # parser.add_argument("--model", type=str, default="gpt2")
    # parser.add_argument('--model_dir', default='./model/')
    # parser.add_argument("--no_cuda", action="store_true")
    # parser.add_argument("--latent_size", type=int, default=768)
    # parser.add_argument("--latent_num",type=int, default=1)
    # parser.add_argument("--seq_len_per_latent",type=int, default=50)
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--epoch",type=int, default=600)
    # parser.add_argument("--lr",type=float, default=1e-4)

    # parser.add_argument("--no_fix", action="store_true")
    # parser.add_argument("--max_length", type=int, default=50)
  
    # parser.add_argument("--wandb", action="store_true")



    # # os.environ["WANDB_DISABLED"] = "true"
    # args = parser.parse_args()
    # if args.wandb:
    #     wandb.init(project="q2k", entity="znchen")
    # train()

    #文件命名规则：   数据集名称_生成模型_prompt/question.json

    # datasetname = 'obqa_test'
    # q_dir = '/users5/znchen/Question2Knowledge/SearchQasP/obqa_Additional/test_complete.jsonl'
    # generate_f_with_question(q_dir,datasetname,ft=True)   #微调gpt 用问题生成fact
    # # generate_f_with_question(q_dir,datasetname,ft=False)    #直接用gpt 用问题生成fact
    # # generate_f_with_fixed_p(q_dir,datasetname,ft=False)   #直接用gpt  用promp生成fact
    # #



    #generate for csqa
    datasetname = 'csqa_test'
    q_dir = '/users5/znchen/Question2Knowledge/SearchQasP/train_rand_split.jsonl'
    generate_f_with_question(q_dir,datasetname,ft=True)
    # generate_f_with_question(q_dir,datasetname,ft=False)
    # generate_f_with_fixed_p(q_dir,datasetname,ft=False)



    