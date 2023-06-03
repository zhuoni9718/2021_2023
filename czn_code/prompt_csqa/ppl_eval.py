from cmath import exp
from socket import SO_ERROR
from evaluate import load
from transformers import GPT2LMHeadModel,AutoTokenizer
import jsonlines
import torch
import time
from tqdm import tqdm
from lm_text_generator import init_model
perplexity = load("perplexity", module_type="metric")

device = 'gpu'
#device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")
print(torch.cuda.is_available())
def ppl_evaluate(k_dir):
    sum=0
    inputs = []
    with jsonlines.open(k_dir, "r") as f:
        for item in f:
            # print(item[:3])
            # exit()
            # input=item[0]#取1
           
            for i in range(len(item)):
                input = item[i]
                inputs.append(input)
            # input=item[:3]#取前3
    print(len(inputs))
    # exit()
    print('begin cal')
    time1 = time.time()
    results = perplexity.compute(model_id='gpt2',
                                device='gpu',
                                add_start_token=False,
                                predictions=inputs)
    print('cost time: ',time.time()-time1)
    mean_perplexity = round(results["mean_perplexity"], 2)#ppl_mean
    # 取出ppl较大的k 可以用一个hash表存 idx 对ppl分数排序
    eachperplexities = results['perplexities']
    eachperplexities_id = {}
    for i in range(len(eachperplexities)):
        eachperplexities_id[i] = eachperplexities[i]  #idx:ppl
    sortedp = sorted(list(eachperplexities_id.items()),key=lambda x:x[1])
    # print(list(eachperplexities_id.items()))
    # exit()
    # list(eachperplexities_id).sort(key=lambda x:x[1])
    # 输出ppl排序从高到低+问题
    # write_context = []
    # for i in range(len(sortedp)):
    #     write_context.append('Q : '++'k : '+sortedp[i])

    # 输出五十条
    ppllaegerk = [] 
    for item in sortedp[-51:]:
        ppllaegerk.append(inputs[item[0]]) 
    pplsmallerk = []
    for item in sortedp[:51]:
        pplsmallerk.append(inputs[item[0]])
    sorted_k = []
    for item in sortedp:
        sorted_k.append(inputs[item[0]])
    


    print('mean_perplexity:',mean_perplexity)  
    print('--------------------')

    return sorted_k,pplsmallerk,ppllaegerk

def ppl_evaluate_using_lm(k_dir):
    # 如果是loss就应该越小越好
    sum=0
    ppl_dict = {}
    with jsonlines.open(k_dir, "r") as f:
        count  = 0
        inputs = []
        
        for line in tqdm(f):
            # print(item[:3])
            # exit()
            input=line[:1]#取1
            inputs.append(input)
            # input=item[:3]#取前3
            # input_ids = tokenizer(input)
            # print(input)
        input_ids = tokenizer(inputs, return_tensors="pt", padding=True)["input_ids"].to(device) 
            # print(input_ids)
        # input_ids.append(input_id)
        loss = model(input_ids, labels=input_ids, return_dict=True).loss
            # 这个loss是nll
        print(loss)
        result = torch.exp(loss).no_grad().cpu()
        print(result)
        # exit()
        res_sum = sum(result)  #ppl_mean
        for i in range(len(result)):
            ppl_dict[i] = result
        
    
    # print(sum)
    print(sum/1221)
    print('--------------------')
    pplsmallerk,ppllargerk = [],[]
    sorted_res = sorted(list(ppl_dict.items()),key = lambda x:x[1])
    for i in range(1,52):
        pplsmallerk.append(sorted_res[i-1])
        ppllargerk.append(sorted_res[-i])
    return pplsmallerk,ppllargerk


def writeK(res_dir,pplsmallerk,ppllaegerk,sorted_k ):

    with jsonlines.open(res_dir,'w') as f:
        for item in pplsmallerk:
            f.write(item)
        f.write('==========smaller above===================')
        for item in ppllaegerk:
            f.write(item)
        f.write('==========all===================')
        for item in sorted_k:
            f.write(item)
    print('write succ')
    return 

device = 0
device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")
print(device)
# model, tokenizer = init_model('gpt2-xl', device)
model_name = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

if __name__== '__main__':
    # promptnames = ["logic_abs"]
    # promptnames = ["logic_abs","logic_con","logic_ana","logic_ind","logic_cau",'logic_comprehensive']
    # res_dir = './ppl.jsonl'
    # for item in promptnames:
    #     k_dir = './outputfile/k_gen_by_'+item+'_512_transfered.jsonl'
    #     print(item)
    #     sorted_k,pplsmallerk,ppllaegerk = ppl_evaluate(k_dir)
    #     # pplsmallerk,ppllaegerk = ppl_evaluate_using_lm(k_dir)
    #     writeK(res_dir,pplsmallerk,ppllaegerk,sorted_k )

    # k_dir = '/users5/znchen/jyzhang/baseline/qkk/k_gen_by_qkk_qs_len512.jsonl'
    # sorted_k,pplsmallerk,ppllaegerk = ppl_evaluate(k_dir)
    # # pplsmallerk,ppllaegerk = ppl_evaluate_using_lm(k_dir)
    # writeK(res_dir,pplsmallerk,ppllaegerk,sorted_k )

    # obqa
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/obqa/obqa_Additional/dev_complete.jsonl'
    k_dir = './outputfile/k_gened_by_prompt_obqa.dev.jsonl'
    sorted_k,pplsmallerk,ppllaegerk = ppl_evaluate(k_dir)
    

    # # commongen
    # data_dir = '/users5/znchen/commongen/commongen_data/commongen.dev.jsonl'
    k_dir = './outputfile/k_gened_by_prompt_commongen.dev.jsonl'
    sorted_k,pplsmallerk,ppllaegerk = ppl_evaluate(k_dir)

