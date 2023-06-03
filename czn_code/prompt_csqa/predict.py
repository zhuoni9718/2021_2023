#%%
# from transformers import DebertaV2Tokenizer, DebertaV2Model,DebertaV2PreTrainedModel
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

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
def dataloader(dir):
    with jsonlines.open(dir,'r') as f:
        csqa_data=[]
        for line in f:
            # question=["question"]['stem']
            # choice=["question"]["choices"]
            csqa_data.append(line)
    return csqa_data
        
class csqaData(Dataset):
    def __init__(self,filename):
        csqa_data = dataloader(filename)
        # print(x)
        # print(y)
        self.len=len(csqa_datq)
        # print(self.len)
        self.x_data=csqa_data['question']
        self.y_data=csqa_data['answerKey']

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len


def pre(data):
    # tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    # model = DebertaForTokenClassification.from_pretrained("microsoft/deberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMultipleChoice.from_pretrained("roberta-base")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits


def getk_with_id(id_dir,k_dir):
    k=[]
    q_id=[]
    with jsonlines.open(id_dir,'r') as f:
        for line in f:
            q_id.append(line["id"])
    with open(k_dir,'r') as f:
        for line in f:
            k.append(line)
    id_k={}
    for i in range(len(k)):
        id_k[q_id[i]] = k[i]
    return id_k

def getacc(res,label):
    idx_word=['A','B','C','D','E']
    right_num=0
    for i in range(len(label)):
        # print(i)
        if label[i]==res[i]:
            right_num=right_num+1
    return right_num/len(label)
    
def get_mask_ids(ids):
    return torch.sign(ids)

def get_segment_ids(ids):
    # ids: batch_size * 4 * seq_len
    segment_matrix = torch.zeros(ids.shape)
    segment_idx = 102
    batch, choice,seq_len = ids.shape[0:3]
    seppos=seq_len
    for b in range(batch):
        for c in range(choice):
            # print((ids[b][c] == 102).nonzero())
            for k in range(seq_len):
                if ids[b][c][k] == 102:
                    seppos=k
                    break

            # seg_idx = (ids[b][c] == 102).nonzero().squeeze()
            
            segment_matrix[b][c][seppos:] = 1
    return segment_matrix.type(torch.LongTensor)

def get_lm_score_wo_index(model, batch, pad_token_id):
    """
    Get the lowest cross entropy loss for each instance (list of clarifications) in the batch
    using the langage model
    """
    # Batch: [num_clarifications, max_length]
    with torch.no_grad():
        num_clarifications, max_length = batch.shape
        shift_labels = batch[..., 1:].contiguous().view(-1).clone()
        shift_labels[shift_labels == pad_token_id] = -100
        lm_logits = model(batch).logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(num_clarifications, -1).mean(1).min().cpu().item()

    return loss

def init_model(model_name: str,device: torch.device):
    print('model name: ',model_name)
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    # logger.info(f'Initializing {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.pad_token is None:
    #     if tokenizer.eos_token is not None:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         tokenizer.pad_token = tokenizer.eos_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def get_lm_score(model, batch, pad_token_id):
    """
    Get the lowest cross entropy loss for each instance (list of clarifications) in the batch
    using the langage model
    """
    # Batch: [num_clarifications, max_length]
    with torch.no_grad():
        num_clarifications, max_length = batch.shape
        shift_labels = batch[..., 1:].contiguous().view(-1).clone()
        shift_labels[shift_labels == pad_token_id] = -100
        outputs = model(batch, return_dict=True)
        # print("model(batch): ", outputs)
        lm_logits = outputs.logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(num_clarifications, -1).mean(1)
        loss_min_index = loss.argmin().cpu().item()
        loss = loss.min().cpu().item()

    return loss, loss_min_index

def predict(batch,num_choice,choice_len):
    # print(batch)
    # exit()
    
    # input_token = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
    tokenized = [tokenizer(per_clar, return_tensors="pt", padding=True)["input_ids"].to(device) for per_clar in batch]
    # print('----------tokenized')
    # print(tokenized)
    num_choices = len(tokenized)
    batch_size = 64
    num_batches = int(math.ceil(len(tokenized[0]) / batch_size))
    per_choice_score = [1000] * num_choices
    for batch_index in range(0, num_batches):
        # print(batch_index)
        # print('--cur batch:',batch[batch_index])
        # curr_batch = [tokenized[i][batch_index*batch_size:(batch_index+1)*batch_size]
        #                                   for i in range(num_choices)]
        curr_batch = [tokenized[i][batch_index*batch_size:]
                                          for i in range(num_choices)]
        curr_scores = [get_lm_score(model, clars_choice, tokenizer.pad_token_id)
                        for clars_choice in curr_batch]
        # print("curr_scores: ", curr_scores) # 5
        per_choice_score = [min(per_choice_score[i], curr_scores[i][0]) for i in range(num_choices)]

    prediction = int(np.argmin(per_choice_score))
    k_idx = curr_scores[prediction][1]

    return prediction

def predict_diff(batch,num_choice,choice_len,num_k):
    # print(batch)
    # exit()
    # input_token = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
    tokenized = [tokenizer(per_clar, return_tensors="pt", padding=True)["input_ids"].to(device) for per_clar in batch]
    num_choices = len(tokenized)
    num_choice = 5
    batch_size = 64
    num_batches = int(math.ceil(len(tokenized[0]) / batch_size))
    per_choice_score = [1000] * num_choices

    for batch_index in range(0, num_batches):
        curr_batch = [tokenized[i][batch_index*batch_size:(batch_index+1)*batch_size]
                                          for i in range(num_choices)]
        # print("curr_batch: ", curr_batch) # 5 * 10
        # print("curr_batch, tensor.shape: ", curr_batch[0].shape, curr_batch[1].shape)
        curr_scores = [get_lm_score(model, clars_choice, tokenizer.pad_token_id)
                        for clars_choice in curr_batch]
        # print("curr_scores: ", curr_scores) # 5
        k_diff = []
        k_min_ser = []
        min_k = 0
        min_c = 0
        if len(curr_scores) < num_k * num_choice:
            num_k = int(len(curr_scores) / 5)
        for i in range(num_k):
            choice_scores = []
            for j in range(num_choice):
                choice_scores.append(curr_scores[i + num_k*j][0])
            k_diff.append(max(choice_scores) - min(choice_scores))
            k_min_ser.append(choice_scores.index(min(choice_scores)))
        min_k = k_diff.index(max(k_diff))
        min_c = k_min_ser[min_k]
        # print("per_choice_score: ", per_choice_score)
    prediction = int(min_k + num_k * min_c)
    k_idx = curr_scores[prediction][1]
    # fields["prediction"] = prediction
    # fields["k_idx"] = k_idx

    
    return prediction

def pre_use_entity_k(data_dir,k_dir):
    k = []
    data = []
    kqc_batch = []
    kqc_pre = []
    with jsonlines.open(k_dir,'r') as f:
        for line in tqdm(f):
            k.append(line[0]['sentence'][0])
            # print(k)
            # break
    with jsonlines.open(data_dir,'r') as f:
        for line in tqdm(f):
            data.append(line)
            # print(data)
            # break
    labels = []
    k_idx = 0
    for i in tqdm( range(len(data))):
        question = data[i]['question']['stem']
        choices = data[i]['question']['choices']
        answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])
        labels.append(answer_key)
        kqc = []
        k_num = 1
        for item in choices:
            num = 0
            k_item = k[k_idx]
            k_idx += 1
            if num>=k_num:
                break
            num += 1
            # print(item['text'])
            # print(question)
            kqc_tmp = 'Passage:'+k_item + '. Question:' + question + ' Answer: ' + item['text']+'.'
            # print(kqc_tmp)
            # exit()
            kqc.append(kqc_tmp)
        num_choice = 5
        choice_len = 0
        # print(len(kqc))
        # print(kqc.shape)
        # kqc_y = predict_wo_batch(kqc,num_choice,choice_len)
        kqc_y = predict(kqc,num_choice,choice_len)
        
        kqc_pre.append(int(kqc_y))
    with jsonlines.open('res_wogpu.txt','w') as f:
        for item in kqc_pre:
            f.write(item)
    acc = getacc(kqc_pre,labels)
    print(acc)

def predict_wo_batch(batch,num_choice,choice_len):
    # print(batch)
    tokenized = [tokenizer(per_clar, return_tensors="pt", padding=True)["input_ids"].to(device) for per_clar in batch]
    num_choices = len(tokenized)
    # print('tokenized shape:',[tokenized[i].shape for i in range(num_choics)])
    # batch_size = len(tokenized[0])
    # print('len(tokenized[0]):',len(tokenized[0]))
    # exit()
    batch_size = 32
    num_batches = int(math.ceil(len(tokenized[0]) / batch_size))   #len(tokenized[0])是最长一句话的长度吧
    per_choice_score = [1000] * num_choices

    for batch_index in range(0, num_batches):
        curr_batch = [tokenized[i][batch_index*batch_size:(batch_index+1)*batch_size]
                                          for i in range(num_choices)]
        # print("curr_batch: ", curr_batch) # 5 * 10
        # print("curr_batch, tensor.shape: ", curr_batch[0].shape, curr_batch[1].shape)
        curr_scores = [get_lm_score(model, clars_choice, tokenizer.pad_token_id)
                        for clars_choice in curr_batch]
        # print("curr_scores: ", curr_scores) # 5
        per_choice_score = [min(per_choice_score[i], curr_scores[i][0]) for i in range(num_choices)]
        # print("per_choice_score: ", per_choice_score)
    # exit()
    prediction = int(np.argmin(per_choice_score))
    # k_idx = curr_scores[prediction][1]

    return prediction
  
def pre_use_all_k(data_dir,k_dir,sentence_type = 'q',k_num = 0):
    k = []
    data = []
    kqc_batch = []
    kqc_pre = []
    ks_pre = []
    k_flag = True  
    # s_for_each_flag = True   # 每个陈述句使用其单独的知识 使用的时候需要对应知识dir
    s_for_each_flag = False   # 每个陈述句使用其单独的知识 使用的时候需要对应知识dir
    print("K flag: ",k_flag) 
    if k_dir=='':
        with jsonlines.open(data_dir,'r') as f:
            for line in tqdm(f):
                data.append(line['data'])
                k.append([line['k'].split('\n')[0]])

    else:
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

    for i in tqdm(range(len(data)),mininterval=60):
        question = data[i]['question']['stem']
        if 'statement' in data[i]:
            statements = data[i]["statements"]
        choices = data[i]['question']['choices']
        answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])
        labels.append(answer_key)
        kqc = []
        
        if sentence_type=='q':
            for item in choices:
                num = 0
                if k_num==0:
                    kqc.append('Question: ' + question + ' Answer: ' + item['text']+'.')
                    num+=1
                else:
                    
                    for k_item in k[i]:
                        # print('k_item:  ',k_item)
                        # exit()
                        if num>=k_num:
                            break
                        num += 1
                        kqc_tmp = k_item + ' Question: ' + question + ' Answer: ' + item['text']+'.'
                        kqc.append(kqc_tmp)
            if i<5:
                print(f'[predicting]{kqc}')
                    # kqc.append( 'Answer: ' + item['text']+'.')
            # print(kqc.shape)
            # print(kqc)
            # kqc_y = predict_diff(kqc,num_choice,choice_len,k_num)
            
            kqc_y = predict(kqc,num_choice,choice_len)
            # kqc_y = predict(kqc,num_choice,choice_len)
            if k_num!=0:kqc_pre.append(int(kqc_y/k_num))
            else:kqc_pre.append(int(kqc_y))
        elif sentence_type=='s':
            ks = []
            if s_for_each_flag==False:
                for item in statements:

                    if k_flag == True:
                        num=0
                        for k_item in k[i]:
                            if num>=k_num:break
                            num+=1
                            ks.append(k_item+' '+item["statement"])  #也可以试一下加自然语言描述的句子
                    else:
                        ks.append(item['statement'])
                ks_y = predict_wo_batch(ks,num_choice,choice_len)
                if k_flag == True:ks_pre.append(int(ks_y/k_num))
                else: ks_pre.append(int(ks_y))
            else:
                for j in range(len(statements)):
                    num = 0
                    for k_item in k[i*5+j]:
                        if num>=5:
                            break
                        ks.append(k_item+" "+statements[j]['statement'])
                        num+=1
                if num<5:
                    print(f'[predicting]{ks}')
                # exit()
                ks_y = predict_wo_batch(ks,num_choice,choice_len)
                if k_flag == True:ks_pre.append(int(ks_y/5))
                else: ks_pre.append(int(ks_y))

    if sentence_type=='q':res = kqc_pre 
    else:res = ks_pre
    with jsonlines.open('res_wogpu.txt','w') as f:
        for item in res:
            f.write(item)
    acc = getacc(res,labels)
    print(acc)
    return res,acc


def pre_use_s_w_mask(s_dir, k_dir,mode = 's',k_num = 0):
    k = []
    data = []
    kqc_batch = []
    kqc_pre = []
    ks_pre = []
    k_flag = True  
    # k_flag = False  
    # s_for_each_flag = True   # 每个陈述句使用其单独的知识 使用的时候需要对应知识dir
    s_for_each_flag = False   # 每个陈述句使用其单独的知识 使用的时候需要对应知识dir
    print("K flag: ",k_flag)
    if k_dir!='': 
        with jsonlines.open(k_dir,'r') as f:
            for line in tqdm(f):
                k.append(line)

    with jsonlines.open(s_dir,'r') as f:
        for line in tqdm(f):
            data.append(line)
            # print(data)
            # break
    labels = []        
    k_num = k_num
    num_choice = 5*k_num
    choice_len = 0

    for i in tqdm( range(len(data))):
        question = data[i]['question']['stem']
        statements = data[i]["statements"]
        statement_w_mask = data[i]['statements_with_mask']
        choices = data[i]['question']['choices']
        statements_sentences = []
        for choice in choices:
            statements_sentence = statement_w_mask.replace('[MASK]',choice['text']).strip()
            statements_sentences.append(statements_sentence)
        answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])

        labels.append(answer_key)
        kqc = []
        ks = []
        
        if s_for_each_flag==False:
            for item in statements_sentences:
                if k_flag == True:
                    num=0
                    for k_item in k[i]:
                        if num>=k_num:break
                        num+=1
                        ks.append(k_item+' '+item)  #也可以试一下加自然语言描述的句子
                else:
                    ks.append(item)
            ks_y = predict_wo_batch(ks,num_choice,choice_len)
            if k_flag == True:ks_pre.append(int(ks_y/k_num))
            else: ks_pre.append(int(ks_y))
            print((ks))
        else:
            for j in range(len(statements_sentences)):
                num = 0
                for k_item in k[i*5+j]:
                    if num>=5:
                        break
                    ks.append(k_item+" "+statements_sentences[j])
                    num+=1
        # print(len(ks))
        # exit()
        ks_pre.append(int(ks_y/k_num))

    res = ks_pre
    with jsonlines.open('res_s_w_mask.txt','w') as f:
        for item in res:
            f.write(item)
    acc = getacc(res,labels)
    print(acc)


def getdata(data_dir):
    data = []
    with jsonlines.open(data_dir,'r') as f:
        for line in tqdm(f):
            data.append(line)
    return data

def predict_api(data,k,k_num = 0):
    # data: {question:  choices:  answerKey: groundTruth:}
    kqc_pre = []
    labels = []
    
    for i in tqdm(range(len(data))):
    # for i in tqdm(range(10)):
        predict_contexts = []

        question = data[i]['question']
        choices = data[i]['choices']
        answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])
        labels.append(answer_key)
        this_k = k[i]
        choice_len = 0
        for item in choices:
            for j in range(k_num):
                # predict_context = this_k[j] + ' '+ question+' ' + item['text']+'.'
                predict_context = question+' ' + item['text']+'.'
                predict_contexts.append(predict_context)
        print(predict_contexts)
        # exit()
        num_choice = len(choices)
        kqc_y = predict(predict_contexts,num_choice,choice_len)
        # kqc_y = predict(kqc,num_choice,choice_len)
        if k_num!=0:kqc_pre.append(int(kqc_y/k_num))
        else:kqc_pre.append(int(kqc_y))
    res = kqc_pre
    print(res)
    print(labels)
    acc = getacc(res,labels)
    print(acc)
    return res

def predict_cb(premise_free,kqc,num_choice,choice_len,alpha):
    # kqc
    tokenized = [tokenizer(per_clar, return_tensors="pt", padding=True)["input_ids"].to(device) for per_clar in kqc]
    # print('----------tokenized')
    # print(tokenized)
    num_choices = len(tokenized)
    batch_size = 64
    num_batches = int(math.ceil(len(tokenized[0]) / batch_size))
    per_choice_score = [1000] * num_choices
    for batch_index in range(0, num_batches):
        # print(batch_index)
        # print('--cur batch:',batch[batch_index])
        # curr_batch = [tokenized[i][batch_index*batch_size:(batch_index+1)*batch_size]
        #                                   for i in range(num_choices)]
        curr_batch = [tokenized[i][batch_index*batch_size:]
                                          for i in range(num_choices)]
        # print('----------curbatch')
        
        # print("curr_batch: ", curr_batch)    # 5 * 10
        # print("curr_batch, tensor.shape: ", curr_batch[0].shape, curr_batch[1].shape)
        curr_scores = [get_lm_score(model, clars_choice, tokenizer.pad_token_id)
                        for clars_choice in curr_batch]
        # print("curr_scores: ", curr_scores) # 5
        per_choice_score = [min(per_choice_score[i], curr_scores[i][0]) for i in range(num_choices)]

    #premise_free
    premise_free_per_choice_score = [1000] * num_choices
    premise_free_tokenized = [tokenizer(per_clar, return_tensors="pt", padding=True)["input_ids"].to(device) for per_clar in premise_free]
    for batch_index in range(0, num_batches):
        curr_batch = [premise_free_tokenized[i][batch_index*batch_size:]
                                          for i in range(num_choices)]
        curr_scores = [get_lm_score(model, clars_choice, tokenizer.pad_token_id)
                        for clars_choice in curr_batch]
        # print("curr_scores: ", curr_scores) # 5
        premise_free_per_choice_score = [min(premise_free_per_choice_score[i], curr_scores[i][0]) for i in range(num_choices)]
    final_score=[]
    for i in range(num_choices):
        final_score.append((1-alpha)*per_choice_score[i]+alpha*premise_free_per_choice_score[i] )
    prediction = int(np.argmin(final_score))
    # print("per_choice_score:{}, premise_free_per_choice_score:{}, ",per_choice_score,premise_free_per_choice_score)
    # print("final score: ",final_score)
    k_idx = curr_scores[prediction][1]
    # fields["prediction"] = prediction
    # fields["k_idx"] = k_idx
    
    return prediction

def pre_cb_preprocess(data_dir,k_dir,sentence_type = 'q',k_num = 0,alpha=0.5):
    k = []
    data = []
    kqc_batch = []
    kqc_pre = []
    ks_pre = []
    k_flag = True  
    # s_for_each_flag = True   # 每个陈述句使用其单独的知识 使用的时候需要对应知识dir
    s_for_each_flag = False   # 每个陈述句使用其单独的知识 使用的时候需要对应知识dir
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

    for i in tqdm(range(len(data))):
        question = data[i]['question']['stem']
        statements = data[i]["statements"]
        choices = data[i]['question']['choices']
        answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])

        labels.append(answer_key)
        kqc = []
        qc = []
        c = []
        if sentence_type=='q':
            for item in choices:
                num = 0
                qc_tmp = question + ' ' + item['text']+'.'

                for k_item in k[i]:
                    if num>=k_num:
                        break
                    num += 1
                    qc.append(qc_tmp)
                    c.append(item['text'])
                    # print(item['text'])
                    # print(question)
                    kqc_tmp = k_item + ' Question: ' + question + ' Answer: ' + item['text']+'.'
                    kqc.append(kqc_tmp)
                
                if k_num==0:
                    qc.append(qc_tmp)
                    c.append(item['text'])
                    kqc.append('Question: ' + question + ' Answer: ' + item['text']+'.')
                    # kqc.append( 'Answer: ' + item['text']+'.')
            # print(kqc.shape)
            # print(kqc)
            # kqc_y = predict_diff(kqc,num_choice,choice_len,k_num)
            # premise_free = qc
            premise_free = qc
            kqc_y = predict_cb(premise_free,kqc,num_choice,choice_len,alpha)
            # kqc_y = predict(kqc,num_choice,choice_len)
            if k_num!=0:kqc_pre.append(int(kqc_y/k_num))
            else:kqc_pre.append(int(kqc_y))
        elif sentence_type=='s':
            ks = []
           
            if s_for_each_flag==False:
                for item in statements:

                    if k_flag == True:
                        num=0
                        for k_item in k[i]:
                            if num>=k_num:break
                            num+=1
                            ks.append(k_item+' '+item["statement"])  #也可以试一下加自然语言描述的句子
                    else:
                        ks.append(item['statement'])
                ks_y = predict_wo_batch(ks,num_choice,choice_len)
                if k_flag == True:ks_pre.append(int(ks_y/k_num))
                else: ks_pre.append(int(ks_y))
            else:
                for j in range(len(statements)):
                    num = 0
                    for k_item in k[i*k_num+j]:
                        if num>=5:
                            break
                        ks.append(k_item+" "+statements[j]['statement'])
                        num+=1
                # print(ks)
                # exit()
                ks_y = predict_wo_batch(ks,num_choice,choice_len)
                if k_flag == True:ks_pre.append(int(ks_y/5))
                else: ks_pre.append(int(ks_y))

    if sentence_type=='q':res = kqc_pre 
    else:res = ks_pre
    with jsonlines.open('res_wogpu.txt','w') as f:
        for item in res:
            f.write(item)
    acc = getacc(res,labels)
    print(acc)
    return res



def ensamble_predict(data_dir,k_dirs):
    k_all = []
    for item in k_dirs:
        with jsonlines.open(item,'r') as f:
            this_k = []
            for line in f:
                this_k.append(line[0])
            k_all.append(this_k)  # k_all[0:3][len(Data)]
    
    data = []
    kqc_batch = []
    kqc_pre = []
    with jsonlines.open(data_dir,'r') as f:
        for line in tqdm(f):
            data.append(line)

    labels = []        
    k_num = k_num
    num_choice = 5*k_num
    choice_len = 0

    for i in tqdm(range(len(data))):
        question = data[i]['question']['stem']
        if 'statement' in data[i]:
            statements = data[i]["statements"]
        choices = data[i]['question']['choices']
        answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])
        labels.append(answer_key)
        kqc = []
        
        if sentence_type=='q':
            for item in choices:
                num = 0
                if k_num==0:
                    kqc.append('Question: ' + question + ' Answer: ' + item['text']+'.')
                else:
                    for k_item in k[i]:
                        # print('k_item:  ',k_item)
                        # exit()
                        if num>=k_num:
                            break
                        num += 1
                        kqc_tmp = k_item + ' Question: ' + question + ' Answer: ' + item['text']+'.'
                        kqc.append(kqc_tmp)

            kqc_y = predict(kqc,num_choice,choice_len)
            if k_num!=0:kqc_pre.append(int(kqc_y/k_num))
            else:kqc_pre.append(int(kqc_y))
 

    res = kqc_pre 

    with jsonlines.open('res_wogpu.txt','w') as f:
        for item in res:
            f.write(item)
    acc = getacc(res,labels)
    print(acc)
    return res,acc




device = 0
device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")
print(device)
# model, tokenizer = init_model('gpt2-xl', device)
model, tokenizer = init_model('gpt2-large', device)
# summary(model)

#%%

if __name__=='__main__':
    # print(model)
    # exit()
    # data_dir="/users5/znchen/Question2Knowledge/train_rand_split.jsonl"
    # data_dir="/users5/znchen/MHGRN-master/MHGRN-master/data/csqa/statement/train.statement.jsonl"
    # k_dir='/users5/znchen/Question2Knowledge/SearchQasP/outputfile/-q_only_knowledges.txt'
    # k_dir = "/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_fixed_promt.txt"
    # id_k = getk_with_id(data_dir,k_dir)

    # # logic prompt预测    
    # promptnames = ["logic_abs","logic_con","logic_ana","logic_ind","logic_cau",'logic_comprehensive']
    # # promptnames = ["logic_abs"]
    # ensamble_res = []
    # data_dir = "/users5/znchen/MHGRN-master/MHGRN-master/data/csqa/statement/dev.statement.jsonl"
    # data = getdata(data_dir)
    # for item in promptnames:
    #     # time1 = time.localtime()
    #     promptname = item
    #     k_dir = './outputfile/k_gen_by_'+promptname+'_512_transfered.jsonl'
    #     k_num = 2
    #     print('knum: ',k_num)
    #     cur_res = pre_use_all_k(data_dir,k_dir,'q',k_num)
    #     print(promptname)
    #     ensamble_res.append(cur_res)
    #     print('=====================================================')
    # # 求ensamble准确率
    # final_res = []
    # ensamble_labels = []
    # for i in range(len(data)):
    #     answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])
    #     ensamble_labels.append(answer_key)
    #     reshape_re = {}  #把结果存在一个dict中 对dict排序之后 取最高值 然后放到res中 求准确率
    #     for j in range(len(promptnames)):
    #         if ensamble_res[j][i] in reshape_re:
    #             reshape_re[ensamble_res[j][i]] += 1
    #         else:
    #             reshape_re[ensamble_res[j][i]] = 1
    #     y = sorted(list(reshape_re.items()),key = lambda x:x[1])[-1][0]
    #     final_res.append(y)
    # acc = getacc(final_res,ensamble_labels)
    # print('ensamble acc : ',acc)

    # obqa
    # data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/obqa/obqa_Additional/dev_complete.jsonl'
    # k_dir = './outputfile/k_gened_by_prompt_obqa.dev.jsonl'
    # data = readObqa(data_dir)
    # promptname = "logic_abs"
    # # k = [[item['gt']] for item in data]
    # k = read_k(k_dir)
    # k_num = 2
    # # print(k)
    # predict_api(data,k,k_num)

    # # commongen 没有选项不能predict
    # data_dir = '/users5/znchen/commongen/commongen_data/commongen.dev.jsonl'
    # k_dir = './outputfile/k_gened_by_prompt_commongen.dev.jsonl'
    # data = readCommongen(data_dir)
    # k = read_k(k_dir)
    # promptname = "logic_abs"
    # predict_api(data,k)


    # # 用很多条知识进行预测_
    # print("using all k")
    # data_dir = "/users5/znchen/MHGRN-master/MHGRN-master/data/csqa/statement/dev.statement.jsonl"
    # # # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_fixed_promt0625.txt'
    # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_fixed_qk_transfered.jsonl'

    # # 不同知识数量
    # # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_fixed_Q2K0219_transfered.jsonl' #两个样例的prompt
    # # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_faqprompt.dev._transfered.jsonl'  
    # print('K: ',k_dir)
    # k_num = [0,3]
    # # # alpha_list = [0,0.1,0.3,0.5,0.7,0.9,1]
    # for knum in k_num:
    #     print('knum: ',knum)
    #     pre_use_all_k(data_dir,k_dir,'q',knum)
    

    # # 用T5模型生成的知识测试
    # # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_fixed_Q2K0219_transfered.jsonl' #两个样例的prompt
    # k_dirs = [
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_q1k_t5.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_q2k_t5.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_q5k_t5.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_q7k_t5.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_q10k_t5.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_what_gpt2.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_what_t5.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_where_gpt2.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_where_t5.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_why_gpt2.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_why_t5.j_transfered.jsonl',
    # ]  
    # xlnet
    # k_dirs = [
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q1k_xlnet.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q2k_xlnet.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q3k_xlnet.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_qk_xlnet.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q7k_xlnet.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q10k_xlnet.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_what_xlnet.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_where_xlnet.j_transfered.jsonl',
    #     '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_why_xlnet.j_transfered.jsonl'
    # ]  
    k_dirs = [
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q1k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q2k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q3k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q4k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_qk_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q6k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q7k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q8k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q9k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q10k_xlnet.j_transfered.jsonl',
        # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model/k_faqprompt_xlnet0421.dev.j_transfered.jsonl'
    # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model/k_faqprompt_gpt.dev.j_transfered.jsonl',
    # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model/k_faqprompt_gpt.dev.j_transfered.jsonl'
    # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model/k_faqprompt_t5.dev.j_transfered.jsonl'
    # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model/k_faqprompt_xlnet0421.dev.j_transfered.jsonl'
    # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/gpt/k_q6k_xlnet.j_transfered.jsonl',
    # '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_q7k_xlnet.j_transfered.jsonl'
    '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_10_t5.transfered.jsonl'
    ]

    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    writecontext = []
    knums = [0,1,2,3,4,5,6,7,8,9,10]
    # knums=[10]
    for k_dir in k_dirs:
        # try:
        for knum in knums:
            print(f'k_num:{knum}, kdir:{k_dir}')
            _,acc = pre_use_all_k(data_dir,k_dir,'q',knum)
            writecontext.append({'k_num':knum,"kdir":k_dir,'acc':acc})
        # except:
            # continue
    print(writecontext)
    




    # ft
    # # data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/ft/gpt/res_k0307.txt'  #qft
    # data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/res_qck0307.txt'  #qkft
    # k_num = 1
    # pre_use_all_k(data_dir,'','q',k_num)



    # for item in alpha_list:
    #     print("alpha: " ,item)
    #     pre_cb_preprocess(data_dir,k_dir,'q',k_num,item)
    # for k_num in range(0,20,5):
    #     print('k_num is : ',k_num)

    # 用entity k 预测
    # print("using entity_k")
    # data_dir = "/users5/znchen/MHGRN-master/MHGRN-master/data/csqa/statement/dev.statement.jsonl"
    # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/obqa_dev_qac_sentence_all.json'
    # pre_use_entity_k(data_dir,k_dir,'q')

    # #用陈述句生成的 K 预测
    # qc_flag = 's'
    # # print("using k_statement "+ qc_flag)
    # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_s_promt0706.txt'
    # # # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_sForEach_promt0706.txt' #每个陈述句对应生成自己的知识
    # data_dir =  "/users5/znchen/MHGRN-master/MHGRN-master/data/csqa/statement/dev.statement.jsonl"
    # k_num = 5
    # print("k=5")  #acc： 44.3 
    # pre_use_all_k(data_dir, k_dir,qc_flag,k_num)
    # k_num = 3
    # print('k=3')
    # pre_use_all_k(data_dir, k_dir,qc_flag,k_num)
    

    #用 K+陈述句 预测
    # print("using k+statement")
    # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_s_promt0706.txt'
    # data_dir =  "/users5/znchen/MHGRN-master/MHGRN-master/data/csqa/statement/dev.statement.jsonl"
    # k_num = 0
    # pre_use_all_k(data_dir, k_dir,'s',k_num)

    # # 用含mask陈述句预测
    # s_dir= '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement_w_mask.jsonl'
    # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_fixed_promt0625.txt'
    # k_num = 5
    # pre_use_s_w_mask(s_dir, k_dir,'s',k_num)

