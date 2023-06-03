from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
import jsonlines
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

# def calculate_perplexity(text):
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     model = GPT2LMHeadModel.from_pretrained('gpt2')
    
#     # 使用 perplexity 评价生成的内容困惑度
#     tokenized_text = tokenizer.encode(text, return_tensors="pt")
#     with torch.no_grad():
#         logits = model(tokenized_text).logits
#         loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), tokenized_text.view(-1))
#     perplexity = torch.exp(loss)
device = 0
device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.to(device)

def get_ppl(sentence):
  # 对句子进行分词，并添加起始和结束符
  tokens = tokenizer.encode(sentence, add_special_tokens=True)
  # 将分词转换为张量，并放到设备上
  input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
  # 获取语言模型的输出
  with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    # 计算损失函数，这里使用交叉熵损失
    loss = outputs[0]
    # print(loss)
    # 根据损失函数和句子长度计算PPL
    # ppl = math.exp(-loss.item() / len(tokens))
    ppl = torch.exp(loss).item()
    # ppl = math.exp(loss.item())
    # print(ppl)
    return round(ppl,2)

def getpplforfile1(data_dir,bs):
    pplsum = 0
    num = 0
    linecount = 0

    print(data_dir)
    data = []
    with jsonlines.open(data_dir,'r') as f:
        for line in tqdm(f):
            linecount+=1
            data.append(line[0])
    for item in data:
        pplsum += get_ppl(item)
    return pplsum/linecount

def getpplforfile(data_dir,bs):
    pplsum = 0
    num = 0
    linecount = 0

    print(data_dir)
    data = []
    with jsonlines.open(data_dir,'r') as f:
        for line in tqdm(f):
            linecount+=1
            data.append(line[0])
    for i in range(len(data)//bs):
        sentences = data[i:i+bs]
        print(sentences)
        ppl = ppl_gpt(sentences)
        pplsum+=ppl
        num+=1

    print(pplsum/num)
    return pplsum/num
                

def ppl_gpt(data_dir,tokenizer,bs):
    dataset = myDataset(data_dir,tokenizer)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=1)
    print('testing')
    sum_p = 0
    sum_att = 0
    for batch in tqdm(dataloader,mininterval=30):
        batch = {k: v.to(device) for k, v in batch.items()}
        # outputs = model(**batch)
        input_ids = batch["input_ids"]
        attention_id = batch['attention_mask']
        label_id = batch['labels']
        # print('[inputid]',input_ids)
        out_context,input_contexts = [],[]
        eos_token_id = tokenizer.eos_token_id
        # for id in input_ids:
        #     input_context = tokenizer.decode(id, skip_special_tokens=True)
        #     input_contexts.append(input_context)
        bs,_ = batch['input_ids'].size()
        # output = model(input_ids,attention_mask = attention_id,labels=input_ids)
        output = model(input_ids,attention_mask = attention_id,labels=label_id)
        
        print(output[0].size())
        # exit()

        loss = output[0]
        ppl = math.exp(loss.item())
        print(ppl)

        # 
        # 计算损失函数，这里使用交叉熵损失
        # logits = output[1]
        # shift_logits = logits[:, :-1, :].contiguous()
        # shift_labels = batch['input_ids'][:, 1:].contiguous()
        # shift_attentions = batch['attention_mask'][:, 1:].contiguous()
        # # Flatten the tokens
        # loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
        # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
        # sum_p += loss.sum(1)
        # sum_att += shift_attentions.sum(1)
        # meanloss = sum_p / sum_att
        # ppl = torch.exp(meanloss.to('cpu')).detach().numpy().tolist()
        # print('[ppl]',ppl)
    return ppl

class myDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = []
        with jsonlines.open(data_file) as f:
            for item in f:
                # print(item)
                input_text = item[0]
                target_text = item[0]
                self.data.append({"input_text": input_text, "target_text": target_text})
        self.tokenizer = tokenizer
        print(tokenizer)
        self.max_length = 50

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # print(item)
        # if idx<3:
        #     print('[input]',item["input_text"])
        #     print('[target]',item["target_text"])
        input_encoded = self.tokenizer(item["input_text"][:-1], padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(item["target_text"][1:],  padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze()
        input_ids = input_encoded.input_ids.squeeze()
        attention_mask_tensor = input_encoded.attention_mask.squeeze()
        
        # print(inputs)
        # print(f"Processed item: {idx}, input_ids: {input_ids}, labels: {labels}")  # 添加调试输出
        return {"input_ids": input_ids, 'attention_mask':attention_mask_tensor,"labels": labels}


def ppl_gpt2(data_dir):
    data = []
    ppl_sum,count = 0,0
    with jsonlines.open(data_dir) as f:
        for item in f:
            # print(item)
            data.append(item[0])
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    for item in data:
        # 对句子进行分词并且转为PyTorch tensors
        sentence = item
        input_ids = tokenizer.encode(sentence, return_tensors='pt')

        # 分隔输入和标签
        inputs = input_ids[:,:-1]
        labels = input_ids[:,1:]
        # 传入模型进行预测
        with torch.no_grad():
            outputs = model(inputs, labels=labels)

        # 获取预测的损失值
        loss = outputs.loss

        # 计算并返回困惑度
        ppl = torch.exp(loss)
        ppl_sum+=ppl
        # print(ppl.item())
        count+=1
    avg_ppl = ppl_sum/count
    print(avg_ppl)
    return avg_ppl


def getlength(data_dir):
    data = []
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            # for item in line:
            #     data.append(item)
            data.append(line[0])
    count = 0
    for item in tqdm(data):
        count+=len(item.split(' '))
    avglen = count/len(data)
    return round(avglen,3)

# 测试一下
# sentence = "This is a test sentence."
# ppl = get_ppl(sentence)
# print(f"The PPL of '{sentence}' is {ppl:.2f}")

if __name__=='__main__':
    dirs = {
        # '2example':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_fixed_Q2K0219_transfered.jsonl', #1,15
        # '10example':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_fixed_q10k_transfered.jsonl',#1.21
        # '5example':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_fixed_qk_transfered.jsonl',#1.16
        # 'rank':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_rank_qk_dev_transfered.jsonl',
        # 'AtLocation':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_logic_ind_512_transfered.jsonl',
        # 'hasprom':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_logic_abs_512.dev_transfered.jsonl',
        # 'Causes':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_logic_cau_512_transfered.jsonl',
        # 'faqprompt':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_faqprompt.dev._transfered.jsonl',
        # 'statement':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_s_promt0706.txt'
        # 'ft_q':'/users5/znchen/Question2Knowledge/SearchQasP/ft/gpt/res_k0307.txt'

        '1':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q1k_xlnet.j_transfered.jsonl',
        '2':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q2k_xlnet.j_transfered.jsonl',
        '3':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q3k_xlnet.j_transfered.jsonl',
        '4':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q4k_xlnet.j_transfered.jsonl',
        '5':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_qk_xlnet.j_transfered.jsonl',
        '6':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q6k_xlnet.j_transfered.jsonl',
        '7':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q7k_xlnet.j_transfered.jsonl',
        '8':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q8k_xlnet.j_transfered.jsonl',
        '9':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q9k_xlnet.j_transfered.jsonl',
        '10':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q10k_xlnet.j_transfered.jsonl',


        # '1':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q1k_xlnet.j_transfered.jsonl',
        # '2':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q2k_xlnet.j_transfered.jsonl',
        # '3':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q3k_xlnet.j_transfered.jsonl',
        # '4':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q4k_xlnet.j_transfered.jsonl',
        # '5':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_qk_xlnet.j_transfered.jsonl',
        # '6':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q6k_xlnet.j_transfered.jsonl',
        # '7':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q7k_xlnet.j_transfered.jsonl',
        # '8':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q8k_xlnet.j_transfered.jsonl',
        # '9':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q9k_xlnet.j_transfered.jsonl',
        # '10':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q10k_xlnet.j_transfered.jsonl',

        # '1':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q1k_t5.j_transfered.jsonl',
        # '2':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q2k_t5.j_transfered.jsonl',
        # '3':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q3k_xlnet.j_transfered.jsonl',
        # '4':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q4k_xlnet.j_transfered.jsonl',
        # '5':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q5k_t5.j_transfered.jsonl',
        # '6':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q6k_xlnet.j_transfered.jsonl',
        # '7':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q7k_t5.j_transfered.jsonl',
        # '8':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q8k_xlnet.j_transfered.jsonl',
        # '9':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q9k_xlnet.j_transfered.jsonl',
        # '10':'/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_q10k_t5.j_transfered.jsonl'

    }
    context = []
    bs = 4
    for key,value in dirs.items():
        print(key,value)
        data_dir = value
        # pplvalue = ppl_gpt(data_dir,tokenizer,bs)
        # pplvalue = ppl_gpt2(data_dir)
        pplvalue = getpplforfile1(data_dir,bs)
        # avglen = getlength(data_dir)
        context.append(pplvalue)
        # context.append(avglen)
        print(key+' : '+str(pplvalue))
    print(context)

#key: 1.1586183116002424
# key: 1.2126745143508406
# key: 1.1646957003824177
# key: 1.168182456742254
# key: 1.1659659918187482
# key: 1.1737312486423115
# key: 1.1455196118507862
# key: nan
