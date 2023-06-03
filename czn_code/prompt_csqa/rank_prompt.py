#%%
import jsonlines
from cmd import PROMPT
import jsonlines
from transformers import GPT2Tokenizer, GPT2Model,GPT2LMHeadModel
# from pytorch_transformers import GPT2Tokenizer
# from pytorch_transformers import GPT2LMHeadModel
import torch
import numpy as np
#直接从prompt文件中读取
# import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import logging
import argparse	
from lm_text_generator import LMTextGenerator
from prompt import getprompt
logging.basicConfig(level=logging.INFO)
import time
from utils import readCommongen,readObqa
from transfer import dataTransferTok
from predict import predict,getacc

def getprefix(question):
    p = "Generate some knowledge about the sentence in the input. Examples:\n"
    example=["Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n",
    "Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n",
    'Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n',
    'Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n',
    'Input: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n' 
    ]
    # question = 'QUESTION'+'\n'
    prompts = []
    for i in range(5):
        prompt = p+example[i]+example[(i+1)%5]+example[(i+2)%5]+example[(i+3)%5]+example[(i+4)%5]+'Input: '+question+'\nKnowledge:'
        # print(prompt)
        prompts.append(prompt)
    return prompts

def generate(data_dir,res_dir):
    res=[]
    samplenum=20
    num = 0
    sentences = []
    write_context = []
    with jsonlines.open(data_dir,'r') as f:
        for line in tqdm(f):
            num+=1
            # if num>samplenum:
            #     break
            question=line["question"]['stem']
            write_context.append(line)
            cur_prompts = getprefix(question)

            for cur_prompt in cur_prompts:
                # print(cur_prompt)
                sentence=cur_prompt.format(question)
                sentences.append(sentence)
            # exit()
        # print(sentences)

    device = 0
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu") 
    print('device: ',device)
    # modelname = "voidful/metaICL_audio_hr_to_lr"
    # modelname = "gpt2-large"
    modelname = 'mrm8488/flan-t5-large-common_gen'
    # mdoelname = 'xlnet-large-cased'
    print(modelname)
    generator = LMTextGenerator(modelname,device)
    # generator = LMTextGenerator("gpt2-large",device)
    
    p = 0.2
    k = 0.0
    length = 32
    temperature = 1.0
    num_samples = 10
    # print(sentences[0])
    # exit()
    ks = generator.generate(sentences, p,k, temperature, length,num_samples, stop_token='\n')
    
    # print('res:   ',ks)
    # exit()
            # if this_k:
            #     res.append(this_k)
            # else:
            #     res.append(" ")
    fixed_prompt_k_dir=res_dir
    print(len(ks))
    with jsonlines.open(fixed_prompt_k_dir,'w') as f:
        # 写文件的时候写 jsonl  {question:{},knowledge:{}}
        for i in range(len(write_context)):
            for j in range(5):
                f.write({'data:':write_context[i],'knowledge:':ks[5*i+j]})
            print(ks[i])
            # if i > 10:
            #     break
    return res

def get_sequence(data_dir,k_dir,k_num = 0):
    k = []
    data = []

    k_flag = True  
    s_for_each_flag = True   # 每个陈述句使用其单独的知识 使用的时候需要对应知识dir
    # s_for_each_flag = False   # 每个陈述句使用其单独的知识 使用的时候需要对应知识dir
    print("K flag: ",k_flag) 
    with jsonlines.open(k_dir,'r') as f:
        for line in tqdm(f):
            k.append(line)
            # print(k)
            # break
    with jsonlines.open(data_dir,'r') as f:
        for line in tqdm(f):
            data.append(line)
            # print(data)
            # break
    labels = []        
    k_num = k_num
    num_choice = 5*k_num
    choice_len = 0

    kqc = []
    for i in tqdm(range(len(data))):
        question = data[i]['question']['stem']
        statements = data[i]["statements"]
        choices = data[i]['question']['choices']
        answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])

        labels.append(answer_key)
        
        # 这两个循环有问题
        for j in range(5):
            kqc_diffrentc = []
            for item in choices:
                num = 0
                if k_num==0:
                    kqc_diffrentc.append('Question: ' + question + ' Answer: ' + item['text']+'.')
                for k_item in k[i*5+j]:
                    if num>=k_num:
                        break
                    num += 1
                    kqc_tmp = k_item + ' Question: ' + question + ' Answer: ' + item['text']+'.'
                    kqc_diffrentc.append(kqc_tmp)
            kqc.append(kqc_diffrentc)
    # print(kqc)
    return kqc,labels

def predict_for_rankprompt(data_dir,k_dir,sentence_type = 'q',k_num = 0):
    kqc_batch = []
    kqc_pre = []
    ks_pre = []
    rank1y,rank2y,rank3y,rank4y,rank5y=[],[],[],[],[]

    kqc,labels = get_sequence(data_dir,k_dir,k_num)
    num_choice = 5
    choice_len = 0 #这个参数暂时没什么用
    # kqc_y = predict(kqc,num_choice,choice_len)

    ensamblelabels = []
    acc,ranky = [],[]
    num = 0
    for i in tqdm(range(len(kqc)//5)):
        # j = 0
        # if num>10:break
        num+=1
        a = []
        thisranky = []
        for j in range(5):
            kqc_y = predict(kqc[i*5+j],num_choice,choice_len)
            if k_num!=0:thisranky.append(int(kqc_y/k_num))
            else:thisranky.append(int(kqc_y))
        ranky.append(thisranky)
        
        ensamblelabels = [max(a,key=a.count) for a in ranky]
    # print(ranky)
    # print(ranky[:][0])
    ranky = np.array(ranky)
    # exit()  
    # 这里的形状还是有点问题
    print('1')
    acc.append(getacc(ranky[:,0],labels))
    print('2')
    acc.append(getacc(ranky[:,1],labels))
    print('3')  
    acc.append(getacc(ranky[:,2],labels))
    print('4')
    acc.append(getacc(ranky[:,3],labels))
    print('5')
    acc.append(getacc(ranky[:,4],labels))
    print('6')
    ensambleacc = getacc(ensamblelabels,labels)
    for i in range(5):
        print('acc{}:{}'.format(i,acc))
    print('ensamble:',ensambleacc)


    #kqc_pre 长度是5*data_len



# %%

if __name__=='__main__':
    q_dir='/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    res_dir='/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model/k_rank5.t5.jsonl'
    # 交换顺序生成的k 
    k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_rank_qk_dev_transfered.jsonl'
    # generate(q_dir,res_dir)
    # dataTransferTok(res_dir)
    knum = 1 #5类prompt 
    predict_for_rankprompt(q_dir,k_dir,'q',knum )