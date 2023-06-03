from cProfile import label
from random import choice
from transformers import RobertaTokenizer, RobertaForMultipleChoice,RobertaModel
import transformers
# print(transformers.__path__)
# exit()
import torch
import jsonlines
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# from torchsummary import summary
import math
from utils import readCommongen,readObqa,read_k
from predict import predict,getacc

def dataloader(data_dir):
    data = []
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            data.append(line)
    return data

def load_k(k_dir):
    k = []
    with jsonlines.open(k_dir,'r') as f:
        for line in f:
            k.append(line)
    return k

def main(data_dir,k_dir):
    data = dataloader(data_dir)
    k = load_k(k_dir)
    pre,labels = [],[]
    for i in tqdm(range(len(data))):
        batch = []
        labels.append(['A','B','C','D','E'].index(data[i]['answerKey']))

        question = data[i]['question']['stem']
        for j in range(5):
            item = data[i]['question']['choices'][j]
            try:
                context = k[5*i+j][0].strip('</s') +' Question: ' + question + ' Answer: ' + item['text']+'.'
                # context = ' Question: ' + question + ' Answer: ' + item['text']+'.'
            except:
                context = 'Question: ' + question + ' Answer: ' + item['text']+'.'
            print(context)
            batch.append(context)
        num_choice,choice_len  = 5,0
        prediction = predict(batch,num_choice,choice_len)
        # print(prediction)
        pre.append(prediction)
    print(len(pre),len(labels))
    acc = getacc(pre,labels)
    print('acc:',acc)
    return

if __name__=='__main__':
    data_dir = "/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl"
    # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/cg/k_ft_gpt_cg.j_transfered.jsonl'
    
    k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/cg/k_ft_t5_cg.j_transfered_test.jsonl'
    main(data_dir, k_dir)
    # 我觉得是知识没对上