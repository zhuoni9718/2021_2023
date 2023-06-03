#%%
# from transformers import pipeline, set_seed
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
from predict import pre_use_all_k

# 载入预训练模型的分词器
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model_dir='/users5/znchen/Question2Knowledge/model/GPT2'
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.eval()

def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index

def generate_for_one(sentence):
    # print(sentence)
    
    # print(inputs_tensor.shape)
    # print(inputs)
    # print(type(inputs)) #list
    # print(len(inputs))  #句子长度
    # exit()
    gen_len = 50
    gen_num = 5
    # this_k=''
    res=[]
    for _ in range(gen_num):
        inputs = tokenizer.encode(sentence)
        inputs_tensor = torch.tensor([inputs])
        total_predicted_text = ""
        for _ in range(gen_len):
            with torch.no_grad():
                
                outputs = model(inputs_tensor)
                predictions = outputs[0]
            # print(predictions)
            # print(type(predictions))
            # print("prediction size: ",predictions.size())    #torch.Size([1, 162, 50257])
            # exit()
            predicted_index = select_top_k(predictions,k=10)

            predicted_text = tokenizer.decode(inputs+[predicted_index])
            total_predicted_text += tokenizer.decode(predicted_index)

            if '<|endoftext|>' in total_predicted_text:
            # 如果出现文本结束标志，就结束文本生成
                break

            inputs += [predicted_index]
            
            inputs_tensor = torch.tensor([inputs])
        res.append(total_predicted_text)
        # print("res: ",res)
    # exit()
    # print("generated knowledge is: ",total_predicted_text)
    return res



def generate_for_file(dir):
    # generator = pipeline('text-generation', model='gpt2')
    # set_seed(42)
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    res=[]
    len=512
    samplenum=100
    num=0
    with jsonlines.open(dir,'r') as f:
        for line in tqdm(f, mininterval=180):
            num+=1
            if num>samplenum:
                break
            sentence=line[0]
            this_k=generate_for_one(sentence)
            # exit()
            if this_k:
                res.append(this_k)
            else:
                res.append(sentence)
    return res


def generate_for_file_and_write(dir):
    #使用抽取出的相似问题构造K
    res=[]
    len=512
    num=0
    samplenum=1000   
    with jsonlines.open(dir,'r') as f:
        for line in f:
            num+=1
            if num>samplenum:
                break
            sentence=line[0]
            this_k=generate_for_one(sentence)
            # exit()
            res.append(this_k)
    k_dir='./k_only.txt'
    with jsonlines.open(k_dir,'w') as f:
        for item in res:
            print(item)
            f.write(item)
    
def generate_for_file_with_fixed_prompt(q_dir,res_dir,promptname,modelname):
    #使用固定的prompt生成K
    #q_dir是读取问题的文件路径,res_dir是写保存生成结果的文件的路径
    cur_prompt = getprompt(promptname)
    res=[]
    samplenum=5
    num = 0
    sentences = []
    write_context = []
    with jsonlines.open(q_dir,'r') as f:
        for line in tqdm(f, mininterval=180):
            num+=1
            # if num>samplenum:
            #     break
            question=line["question"]['stem']
            write_context.append(line)
            # 先使用第一条statement作为prompt
            answer_key = ['A','B','C','D','E'].index(line["answerKey"])
            statement = line["statements"][answer_key]["statement"]
            statements = line["statements"]

            if promptname=='SK':
                statement_for_each = False
                if statement_for_each == False:
                    sentence = cur_prompt.format(statement)
                    # print(sentence)
                    # exit()
                    sentences.append(sentence)
                else:
                    for s in statements:
                        sentence = cur_prompt.format(s['statement'])
                        sentences.append(sentence)
            else:
                sentence=cur_prompt.format(question)
                # print(sentence)
                # exit()
                sentences.append(sentence)
            # print(sentences)
            # exit()

            # 自己写的generator
            # this_k=generate_for_one(sentence)
            # self-talk的generator
    device = 0
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu") 
    print('device: ',device)
    print(modelname)
    generator = LMTextGenerator(modelname,device)
    # generator = LMTextGenerator("gpt2-large",device)
    
    p = 0.2
    k = 0
    length = 50
    temperature = 1.0
    num_samples = 10

    # print(sentences[0])
    # exit()
    stop_token = '<'
    ks = generator.generate(sentences, p,k, temperature, length,num_samples, stop_token=stop_token)
    
    # print('res:',ks)
    # exit()
    #         # if this_k:
            #     res.append(this_k)
            # else:
            #     res.append(" ")
    fixed_prompt_k_dir=res_dir
    with jsonlines.open(fixed_prompt_k_dir,'w') as f:
        # 写文件的时候写 jsonl  {question:{},knowledge:{}}
        for i in range(len(ks)):
            f.write({'data:':write_context[i],'knowledge:':ks[i]})
    return res
           




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed",  default=['fixed'], choices=[ 'fixed', 'unfixed'])
    args = parser.parse_args() 
    return args

def generate(sentence,prompt=False):
    input_text=sentence
    input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).input_ids
    # print("len sentence : ",len(sentence))
    length = 0
    if prompt==False:
        length = 50
    else:
        length = 600
    # output = model.generate(input_ids=' ', max_length = 600 ,do_sample=True, top_k=10, top_p=0.9, repetition_penalty=1.2, return_dict=True, use_cache=True)
    output = model.generate(input_ids=input_ids, max_length = length ,do_sample=True, top_k=10, top_p=0.9, repetition_penalty=1.2, return_dict=True, use_cache=True)

    res = tokenizer.batch_decode(output)
    # res[0] = res[0][len(sentence):-2]
    # print(res)
    
    return res

def gen_k_with_entityprompt(dir):
    gened =[]
    with jsonlines.open(dir,'r') as f:
        for line in f:
            sentence = line[0]
            res = generate(sentence,prompt=True)
            gened.append(res)
            # break
    return gened



def generate_api(data,k_dir,promptname):
    promptWithData = []
    promptTemplate =  getprompt(promptname)
    count = 0
    device = 0
    for item in data:
        # if count >10:
        #     exit()

        # print(promptTemplate.format(item['question']))
        promptWithData.append(promptTemplate.format(item['question']))
        count +=1
    print(promptWithData[:10])
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu") 
    print('device: ',device)
    modelname = "gpt2-large"
    print(modelname)
    generator = LMTextGenerator(modelname,device)
    p = 0.2
    k = 0.0
    length = 128
    temperature = 1.0
    num_samples = 5
    # print(sentences[0])
    # exit()
    ks = generator.generate(promptWithData, p,k, temperature, length,num_samples, stop_token='\n')

    with jsonlines.open(k_dir,'w') as f:
        for i in range(len(ks)):
            f.write(ks[i])
    return ks


def generate_commongen(q_dir,concept_dir,res_dir,modelname):
    #commongen 增强的模型生成知识
    #q_dir是读取问题的文件路径,res_dir是写保存生成结果的文件的路径
    res=[]
    samplenum=5
    num = 0
    sentences = []
    write_context = []
    with jsonlines.open(q_dir,'r') as f:
        for line in f:
            write_context.append(line)
    question_concepts = []
    with open(concept_dir,'r') as f:
        for line in tqdm(f.readlines()):
            num+=1
            # if num>samplenum:
            #     break
            question_concepts.append(f'Generate a sentence with: {line}.')

    device = 0
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu") 
    print('device: ',device)
    print(modelname)
    generator = LMTextGenerator(modelname,device)
    # generator = LMTextGenerator("gpt2-large",device)
    
    p = 0.2
    k = 0
    length = 50
    temperature = 1.0
    num_samples = 10

    # print(sentences[0])
    # exit()
    stop_token = '</s>'
    ks = generator.generate(question_concepts, p,k, temperature, length,num_samples, stop_token=stop_token)
    
    # print(ks)
    
    with jsonlines.open(res_dir,'w') as f:
        # 写文件的时候写 jsonl  {question:{},knowledge:{}}
        for i in range(len(write_context)):
            for j in range(5):  
                ks_for_thisq = ks[5*i+j]
                f.write({'data:':write_context[i],'knowledge:':ks_for_thisq})
        print(f'write to {res_dir}')
    return res
    
from transformers import BartTokenizer, BartForConditionalGeneration
def mygenerate_commongen(q_dir,concept_dir,res_dir,model_name):
    #commongen 增强的模型生成知识
    #q_dir是读取问题的文件路径,res_dir是写保存生成结果的文件的路径
    res=[]
    samplenum=5
    num = 0
    sentences = []
    write_context = []
    with jsonlines.open(q_dir,'r') as f:
        for line in f:
            write_context.append(line)
    question_concepts = []
    with open(concept_dir,'r') as f:
        for line in tqdm(f.readlines()):
            num+=1
            # if num>samplenum:
            #     break
            question_concepts.append(f'Generate a sentence with: {line}.')

    device = 0
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu") 
    print('device: ',device)
    print(model_name)
    
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    p = 0.2
    k = 0
    length = 50
    temperature = 1.0
    num_samples = 10

    # print(sentences[0])
    # exit()
    stop_token = '</s>'
    input_ids = tokenizer.encode(question_concepts, return_tensors="pt")
    out_ids = model.generate(input_ids.to(device), max_length=50, num_return_sequences=3)
    generated_texts = [tokenizer.decode(out_ids, skip_special_tokens=True) for output_seq in output]
    print(generated_text)
    
    with jsonlines.open(res_dir,'w') as f:
        # 写文件的时候写 jsonl  {question:{},knowledge:{}}
        for i in range(len(write_context)):
            for j in range(5):  
                ks_for_thisq = ks[5*i+j]
                f.write({'data:':write_context[i],'knowledge:':ks_for_thisq})
        print(f'write to {res_dir}')
    return res
def cgmain():
    # commongen 增强的模型
    # model_name = 'liujqian/gpt2-xl-finetuned-commongen'
    res_dir_lst=[
        # './outputfile/dif_model1/cg/k_ft_gpt_cg.train.jsonl',
        # './outputfile/dif_model1/cg/k_ft_t5_cg.jsonl',
        './outputfile/dif_model1/cg/k_ft_Bart_cg.jsonl'
    ]
    model_name_lst=[
        # 'mrm8488/GPT-2-finetuned-common_gen',
        # 'mrm8488/flan-t5-large-common_gen',
        'sibyl/BART-large-commongen'
    ]
    q_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    concept_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/qac/csqa.dev.qac.src'
    # q_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/train.statement.jsonl'
    # concept_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/qac/csqa.train.qac.src'
    writecontext = []
    for res_dir, model_name in zip(res_dir_lst,model_name_lst):
        print(f'{res_dir}\n {model_name}\nqdir:{q_dir}\nconcept_Dir:{concept_dir}')
        # generate_commongen(q_dir,concept_dir,res_dir,model_name)
        mygenerate_commongen(q_dir,concept_dir,res_dir,model_name)
        k_dir = dataTransferTok(res_dir)
        knum = 1
        try:
            _,acc = pre_use_all_k(q_dir,k_dir,'q',knum)
            writecontext.append({"kdir":k_dir,'acc':acc})
        except:
            continue
    print(writecontext)


def fix_prompt_main():
        
    # 用固定prompt 生成知识 
    writecontext = []
    q_dir='/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    
    # # q_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/test.statement.jsonl'
    # # q_dir='/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/train.statement.jsonl'
    # modelname = 'gpt2-large'
    # res_dir_lst=[
    #     './outputfile/dif_model1/gpt/k_what_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_why_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_where_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q1k_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q2k_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q3k_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q4k_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_qk_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q6k_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q7k_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q8k_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q9k_xlnet.jsonl',
    #     './outputfile/dif_model1/gpt/k_q10k_xlnet.jsonl'
    #     './outputfile/dif_model1/gpt/k_sk_gpt.jsonl',
    #     './outputfile/dif_model1/gpt/k_zeroshot_gpt.jsonl',
    #     './outputfile/dif_model1/gpt/k_wrongk_gpt.jsonl'
    # ]
    # prompt_name_lst=[
    #     'what',
    #     'why',
    #     'where'
    #     'Q1K',
    #     'Q2K',
    #     'Q3K',
    #     'Q4K',
    #     'QK',
    #     'Q6K',
    #     'Q7K',
    #     'Q8K',
    #     'Q9K',
    #     'Q10K',
    #     'zeroshot',
    #     'SK',
    #     'Q_wrong_K'
    # ]

    # for res_dir, prompt_name in zip(res_dir_lst,prompt_name_lst):
    #     print(res_dir, prompt_name)
    #     generate_for_file_with_fixed_prompt(q_dir,res_dir,prompt_name,modelname)
    #     k_dir = dataTransferTok(res_dir)
    #     knum = 3
    #     try:
    #         _,acc = pre_use_all_k(q_dir,k_dir,'q',knum)
    #         writecontext.append({"kdir":k_dir,'acc':acc})
    #     except:
    #         continue
    # print(writecontext)
    # modelname = 'xlnet-large-cased'
    # # modelname = 'facebook/bart-large'
    # res_dir_lst=[
    #     # './outputfile/dif_model1/xlnet/k_what_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_why_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_where_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_q3k_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_q4k_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_q6k_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_q8k_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_q9k_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_Sk_xlnet.jsonl',
    #     # './outputfile/dif_model1/xlnet/k_zeroshot_xlnet.jsonl',
    #     './outputfile/dif_model1/xlnet/k_wrongk_xlnet.jsonl'
    # ]
    # prompt_name_lst=[
    #     # 'what',
    #     # 'why',
    #     # 'where'
    #     # 'Q3K',
    #     # 'Q4K',
    #     # 'Q6K',
    #     # 'Q8K',
    #     # 'Q9K',
    #     # 'SK'
    #     # 'zeroshot',
    #     'Q_wrong_K'
    # ]
    # for res_dir, prompt_name in zip(res_dir_lst,prompt_name_lst):
    #     print(res_dir, prompt_name)

    #     generate_for_file_with_fixed_prompt(q_dir,res_dir,prompt_name,modelname)
    #     k_dir = dataTransferTok(res_dir)
    #     knum = 3
    #     try:
    #         _,acc = pre_use_all_k(q_dir,k_dir,'q',knum)
    #         print("kdir:{},acc:{}".format(k_dir,acc))
    #         writecontext.append({"kdir":k_dir,'acc':acc})
    #     except:
    #         continue
    # print(writecontext)

    modelname = 'mrm8488/flan-t5-large-common_gen'
    res_dir_lst=[
        # './outputfile/dif_model1/t5/k_what_xlnet.jsonl',
        # './outputfile/dif_model1/t5/k_why_xlnet.jsonl',
        # './outputfile/dif_model1/t5/k_where_xlnet.jsonl',
        # './outputfile/dif_model1/t5/k_q3k_xlnet.jsonl',
        # './outputfile/dif_model1/t5/k_q4k_xlnet.jsonl',
        # './outputfile/dif_model1/t5/k_q6k_xlnet.jsonl',
        # './outputfile/dif_model1/t5/k_q8k_xlnet.jsonl',
        # './outputfile/dif_model1/t5/k_q9k_xlnet.jsonl',
        # './outputfile/dif_model1/t5/k_Sk_T5.jsonl'
        # './outputfile/dif_model1/t5/k_zeroshot_T5.jsonl',
        # './outputfile/dif_model1/t5/k_wrongk_T5.jsonl'
        './outputfile/dif_model1/t5/k_10num_T5.jsonl'
    ]
    prompt_name_lst=[
    #     # 'what',
    #     # 'why',
    #     # 'where'
        # 'Q3K',
        # 'Q4K',
        # 'Q6K',
        # 'Q8K',
        # 'Q9K',
        # 'SK'
        # 'zeroshot',
        # 'Q_wrong_K'
        'QK'
    ]
    for res_dir, prompt_name in zip(res_dir_lst,prompt_name_lst):
        print(res_dir, prompt_name)

        generate_for_file_with_fixed_prompt(q_dir,res_dir,prompt_name,modelname)
        k_dir = dataTransferTok(res_dir)
        knum = 1
        try:
            _,acc = pre_use_all_k(q_dir,k_dir,'q',knum)
            writecontext.append({"kdir":k_dir,'acc':acc})
        except:
            continue
    print(writecontext)



if __name__=='__main__':

    # cgmain()
    fix_prompt_main()

    
    # prompt_dir='/users5/znchen/Question2Knowledge/SearchQasP/outputfile/obqa_prompt.jsonl'
    #用非固定prompt生成K
    # k=generate_for_file_and_write(prompt_dir)
    
    #固定prompt
    # q_dir='/users5/znchen/Question2Knowledge/SearchQasP/train_rand_split.jsonl'
    # res_dir='./outputfile/k_gen_by_fixed_promt0324.txt'
    # generate_for_file_with_fixed_prompt(q_dir,res_dir)
    
    # args = parse_args()
    
    # if args.fixed=='unfixed':
    #     prompt_dir='/users5/znchen/Question2Knowledge/SearchQasP/outputfile/obqa_prompt.jsonl'
    #     #用非固定prompt生成K
    #     k=generate_for_file_and_write(prompt_dir)
    # else:
    #     #用固定的prompt生成K
    #     q_dir='/users5/znchen/Question2Knowledge/SearchQasP/train_rand_split.jsonl'
    #     res_dir='./outputfile/k_gen_by_fixed_promt0324.txt'
    #     generate_for_file_with_fixed_prompt(q_dir,res_dir)

    #使用entitie prompt生成知识
    # model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    # tokenizer.pad_token = tokenizer.eos_token
    
    # entities_prompt_dir = 'entities.txt'
    # gened = gen_k_with_entityprompt(entities_prompt_dir)
    # with jsonlines.open('./outputfile/gened_by_key_word_prompt','w') as f:
    #     for item in gened:
    #         f.write(item)

    # res_dir='./outputfile/dif_model1/k_q1k_t5.jsonl'
    # modelname = "voidful/metaICL_audio_hr_to_lr"
    # modelname = "gpt2-large"
    # modelname = 'mrm8488/t5-base-finetuned-common_gen'
    # modelname = 'mrm8488/flan-t5-large-common_gen'
    # modelname = 't5-large'
    # prompt_name = 'Q1K'
    # generate_for_file_with_fixed_prompt(q_dir,res_dir,prompt_name,modelname)
    # dataTransferTok(res_dir)

    # 用微调的模型生成知识
    # q_dir='/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/train.statement.jsonl'
    # q_dirs = {'dev':'/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl',\
    #     'test':'/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/train.statement.jsonl'}
    # q_dirs = {'train':'/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/train.statement.jsonl'}
    # res_dir='./outputfile/k_gen_by_qkk_1024.jsonl'
    # promt_type = "QKK"
    # # promptnames = ["logic_abs","logic_con","logic_ana","logic_ind","logic_cau",'logic_comprehensive']
    # promptnames = ["QKK","QK","logic_con","logic_ana","logic_ind","logic_cau",'logic_comprehensive']
    # # # promptnames = ['logic_comprehensive']
    # for item in promptnames:
    #     # for key,q_dir in q_dirs.items():
    #         time1 = time.time()
    #         promptname = item
    #         # res_dir = './outputfile/k_gen_by_'+promptname+'_512.'+key+'.jsonl'
    #         res_dir = './outputfile/k_gen_by_'+promptname+'_512.'+'.jsonl'
    #         generate_for_file_with_fixed_prompt(q_dir,res_dir,promptname)
    #         time2 = time.time()
    #         print(promptname,time2-time1)
    #         print('=======================================================')



    # # obqa 
    # data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/obqa/obqa_Additional/dev_complete.jsonl'
    # k_dir = './outputfile/k_gened_by_prompt_obqa.dev.jsonl'
    # data = readObqa(data_dir)
    # promptname = "logic_abs"
    # generate_api(data,k_dir,promptname)

    # # commongen 生成
    # data_dir = '/users5/znchen/commongen/commongen_data/commongen.dev.jsonl'
    # k_dir = './outputfile/k_gened_by_prompt_commongen10.dev.jsonl'
    # data = readCommongen(data_dir)
    # promptname = 'commongen2'
    # generate_api(data,k_dir,promptname)



# %%
