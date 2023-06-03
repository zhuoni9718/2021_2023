#%%
import faq
import os
import json
import jsonlines
import torch
from tqdm import tqdm
import logging

from predict import pre_use_all_k

logger = logging.getLogger("MC")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# run engine
engine = faq.Engine(logger)
from lm_text_generator import LMTextGenerator

from transfer import dataTransferTok

#%%

def buildPrompt(dir,save_dir):
    data_dir=dir
    prompts=[]
    i=0
    none_search_num = 0
    with open(data_dir,'r') as f:
        for line in tqdm(f):
            # i=i+1
            # if i==10:
            #     break
            sample = json.loads(line)
            question=sample['question']['stem']
            res = engine.search_dst(question, limit=5)
            # print((res))
            if res is None:
                #prompt设置为默认prompt
                p="Generate some knowledge about the sentence in the input. Examples:\n\
Question: Google Maps and other highway and street GPS services have replaced what?\n \
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Question: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Question: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Question: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n\
Question: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Question: {}\n\
Knowledge:".format(question)
                none_search_num+=1
#                 p="Instructions: For the following knowledge and question, generate the answer to the question. Examples:\n\
# Question: Google Maps and other highway and street GPS services have replaced what?\n \
# Answer: atlas\n\
# Question: The fox walked from the city into the forest, what was it looking for?\n\
# Answer: natural habitat\n\
# Question: You can share files with someone if you have a connection to a what?\n\
# Answer: computer network\n\
# Question: Too many people want exotic snakes. The demand is driving what to carry them?\n\
# Answer: pet shops\n\
# Question: {}\n\
# Answer:".format(question)
                prompts.append(p)
            else:
                res_qas=res['qas']
                # print(res_qas)
                # exit()
                p='Generate some knowledge about the sentence in the input. Examples:\n'
                
                for key in res_qas:
                    # print(key)
                    # exit()
                    p_question = key['question'].replace("_","")
                    p=p+'Input: '+p_question+'\n'
                    p=p+'Knowledge: '+key['fact']+'\n'
                    # p=p+'Answer: '+key['answer']+'\n'
                p=p+'Input: '+question+'\n'+'Knowledge:'
                # p=p+'Input: '+question+'\n'+'Answer:'
                prompts.append([p])
                # break
            # print(p)
            # exit()
            #开始写prompt
    # json_str = json.dumps(prompts)
    data_save_dir=save_dir
    with jsonlines.open(data_save_dir,'w') as f:
        # f.write(json_str)
        for key in prompts:
            # print(type(key))
            f.write(key)
        # f.write(json_str)
        f.close()
    print("build prompt, none search num:",none_search_num)
    return prompts


def generate(modelname,q_dir,res_dir,prompts):
    res=[]
    samplenum=2  
    num = 0
    sentences = []
    write_context = []
    with jsonlines.open(q_dir,'r') as f:
        for line in tqdm(f):
            num+=1
            # if num>samplenum:
            #     break
            question=line["question"]['stem']
            write_context.append(line)
            # 先使用第一条statement作为prompt
            statement = line["statements"][0]["statement"]
            statements = line["statements"]
            cur_prompt = prompts[num-1]
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
    # modelname = "voidful/metaICL_audio_hr_to_lr"
    # modelname = "gpt2-large"
    # modelname = 'mrm8488/t5-base-finetuned-common_gen'
    # modelname = 'mrm8488/flan-t5-large-common_gen'
    # modelname = 'xlnet-large-cased'
    print('model:',modelname)
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
    
    # print('res:   ',ks)
    # exit()
            # if this_k:
            #     res.append(this_k)
            # else:
            #     res.append(" ")
    fixed_prompt_k_dir=res_dir
    with jsonlines.open(fixed_prompt_k_dir,'w') as f:
        # 写文件的时候写 jsonl  {question:{},knowledge:{}}
        for i in range(len(ks)):
            f.write({'data:':write_context[i],'knowledge:':ks[i]})
    print("write successful")
    return res
           
def readprompts(prompt_dir):
    prompts = [] 
    with jsonlines.open(prompt_dir,'r') as f:
        for line in f:
            prompts.append(line[0])
    return prompts


if __name__=='__main__':
    engine.build_dst_idx()
    q_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    save_dir = './outputfile/dif_model/dev.faqprompt3k.json'

    res_dirs = {
    'xlnet-large-cased':'./outputfile/dif_model/k_faqprompt3k_xlnet0421.dev.jsonl',
    'gpt2-large':'./outputfile/dif_model/k_faqprompt3k_gpt.dev.jsonl',
    'mrm8488/flan-t5-large-common_gen':'./outputfile/dif_model/k_faqprompt3k_t5.dev.jsonl'
    }
    prompts = buildPrompt(q_dir,save_dir)
    prompts = readprompts(save_dir)
    # print(prompts)
    # exit()
    for key,res_dir in res_dirs.items():
        generate(key,q_dir,res_dir,prompts)
        k_dir = dataTransferTok(res_dir)
        knum = 1
        _,acc = pre_use_all_k(q_dir,k_dir,'q',knum)
        print('acc:',acc)

# %%
