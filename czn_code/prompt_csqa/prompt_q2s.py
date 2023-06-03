import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import re
import math
#模型初始化
def init_model(model_name: str,device: torch.device):
    print('model name: ',model_name)
    print('device: ',device)
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


device = 0
# device = 'cpu'
device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")

# model, tokenizer = init_model('gpt2-xl', device)
model, tokenizer = init_model('gpt2-large', device)

# 读问题
def getq(data_dir):
    questions = []
    choices = []
    answer_keys = []
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            questions.append(line['question']['stem'])
            choices.append(line['question']['choices'])
            answer_keys.append( ['A','B','C','D','E'].index(line['answerKey']))

    return questions,choices,answer_keys

#获取prompt
def get_prompt(questions,shot = 'zero'):
    #zero-shot
    prompts =[]
    if shot=='zero':
        prompt = 'Turn questions into statements. Question:'
        for item in questions:
            this_prompt= prompt+item+' Statements:'
            prompts.append(this_prompt)
            # print(this_prompt)
            # exit()
    #few-shot prompt
    if shot=='few':
        for item in questions:
            # 用mask 填充   还可以加选项 用选项填充
            prompt = 'Turn questions into statements.\n\
Question:Sammy wanted to go to where the people were.  Where might he go?\n\
Statement:Sammy wanted to go to where the people were. He might go [mask].\n\
Question:Bill is stuck in marsh when a man comes up to him peaking Cajun, where is he?\n\
Statement:Bill is stuck in marsh when a man comes up to him peaking Cajun, he is [mask]\n\
Question:What is it called when you slowly cook using a grill?\n\
Statement:It is called [mask] when you slowly cook using a grill?\n\
Question:The man was eating lunch, but rushed when he looked at his watch, why did he rush?\n\
Statement:The man was eating lunch, but rushed when he looked at his watch, he rushed because [mask]\n\
Question:Can you name a good reason for attending school?\n\
Statement:name a good reason for attending school [mask]\n\
Question:How does getting paid feel?\n\
Statement:Getting paid feel [mask]\n\
Question:{}\n\
Statement:'.format(item)
            # print(prompt)
            # exit()
            prompts.append(prompt)
    return prompts

#生成
def generate(sentence):
    input_text = sentence
    
    input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).input_ids
    # print(input_ids.shape)
    length = input_ids.shape[1] + 30
    # print(length)
    # exit()
    output = model.generate(input_ids=input_ids.to(device), max_length = length ,do_sample=True, top_k=10, top_p=0.9, repetition_penalty=1.2, return_dict=True, use_cache=True)
    # output = model.generate(input_ids=input_ids,do_sample=True, top_k=10, top_p=0.9, repetition_penalty=1.2, return_dict=True, use_cache=True)
    statement =  tokenizer.batch_decode(output)
    res = statement[0][len(sentence):]
    if '<|endoftext|>' in res:
        res = res[:res.find('<|endoftext|>')]
    elif '\n' in res:
        res = res[:res.find('\n')]
    return res

def get_s_from_out(file_dir):
    with open(filr_dir,'r') as f:
        for line in f:
            print(line)
            exit()
        # '----' 新的问题
        # ' ' 空格指
        # 'torch_size' 新的

def select(question,statements):
    # statements：list
    # print(statements)
    min_loss = 1000
    y = 0
    for i in range(len(statements)):
        statement = statements[i]
        if len(re.findall('(?=[mask]])', statement))==1:
            batch = statement.replace('[mask]','')
        else:
            batch = ''
        if question not in statement and batch!='':
            print('----',statement)
            batch = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"].to(device)
            loss = (get_lm_score(model, batch, tokenizer.pad_token_id)[0])
            if loss<min_loss:
                y = i
    return y


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
        # print("loss: ", loss.shape, loss)
        # print("loss.view: ", loss.view(num_clarifications, -1).shape, loss.view(num_clarifications, -1))
        # print("loss.view.mean: ", loss.view(num_clarifications, -1).mean(1).shape, loss.view(num_clarifications, -1).mean(1))
        # loss = loss.view(num_clarifications, -1).mean(1).min().cpu().item()
        loss = loss.view(num_clarifications, -1).mean(1)
        # print("loss: ", loss)
        loss_min_index = loss.argmin().cpu().item()
        loss = loss.min().cpu().item()

    return loss, loss_min_index


def con_s_c(statements,choices):
    # num = 0
    l_s = len(statements)
    l_c = len(choices)
    # print('!!!!',l_s,l_c)

    if l_c!=l_s:
        return None
    statement_w_c = []
    # l_s = 1
    # print("len s ")
    for i in range(l_s):

        sentences = []
        sentence = statements[i]
        choice = choices[i]
        for item in choice:
            # print('【choice:】',item)
            if '[mask]' in sentence:
                sc = sentence.replace('[mask]',item['text'])
            else:
                sc = sentence +' '+ item['text']
            sentences.append(sc)
            # print('[sc]',sc)
        # exit()
        statement_w_c.append(sentences)
    # print(statement_w_c)
    return statement_w_c

def predict(batch,num_choice):
    # print(batch)
    # exit()
    # input_token = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
    tokenized = [tokenizer(per_clar, return_tensors="pt", padding=True)["input_ids"].to(device) for per_clar in batch]
    num_choices = len(tokenized)
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
        per_choice_score = [min(per_choice_score[i], curr_scores[i][0]) for i in range(num_choices)]
        # print("per_choice_score: ", per_choice_score)
    prediction = int(np.argmin(per_choice_score))
    k_idx = curr_scores[prediction][1]
    
    return prediction



# def predict(batch,num_choice):
#     input_token = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]

#     # outputs = model(inputs_tensor,attention_mask = inputs_mask_ids, token_type_ids = inputs_segment_ids)
#     outputs = model(input_token.to(device))

#     lm_logits = outputs.logits  #[5, 37, 1024]

#     shift_labels = input_token[..., 1:].contiguous().view(-1).clone()  
   
#     pad_token_id = tokenizer.pad_token_id
#     shift_labels[shift_labels == pad_token_id] = -100
#     shift_logits = lm_logits[..., :-1, :].contiguous()  #[5, 36, 1024]
#     shift_logits = shift_logits.view(-1, shift_logits.size(-1))  #[2555, 1024]
#     loss_fct = CrossEntropyLoss(reduction="none")
#     loss = loss_fct(shift_logits.cpu(), shift_labels)

#     loss = loss.view(num_choice, -1).mean(1).cpu().detach().numpy()

#     y=np.argmin(loss)
#     # pre.append(y)
#     return y

def write_res_text(write_idx_dir,write_text):
    # write_idx_dir = './outputfile/read_s_pre_idx.json'
    with jsonlines.open(write_idx_dir,'w') as f:
        for item in  write_text:
            f.write(item)
        f.close()

def pre_with_sc(statement_w_c,questions,all_statements_res,idx_res,statements_res):
    # statement_w_c : 拼接好的statement和c
    pre = []
    write_text = []
    for i in range(len(statement_w_c)):
        statement_batch = statement_w_c[i]
        choice_idx = predict(statement_batch,5)
        pre.append(choice_idx)
        write_text.append([{'question':questions[i],'statements':all_statements_res[i],'idx':str(idx_res[i]),'selected_s':statements_res[i],'choice_pre':str(choice_idx)}])
    print(write_text)
    # sc_accuracy = accuracy_score(answerkeys, pre)
    # print('acc: ',sc_accuracy)
    return write_text,pre

def select_statement_with_choice_and_pre(statements,choices,questions):
    # statemente: [len(chocies)*statements_num_for_this_q]
    write_text = []
    print('begin select and predict')
    pre_idx = []
    for i in tqdm(range(len(statements))):
        this_choice = choices[i]
        # statement_for_one_question=[]
        statements_to_choice = []
        for item in this_choice:
            statements_for_one_choice = []
            choice_text = item['text']
            for sentence in statements[i]:
                # print("sentence : ",sentence)
            #todo 拼接后预测  如果mask在里面 替换mask
                if '[mask]' in sentence:
                    statements_for_one_choice.append(sentence.replace('[mask]',choice_text))
                else:
                    statements_for_one_choice.append(sentence+' '+ choice_text)
            statement_for_one_choice_idx = select(statements_for_one_choice)
            statement_for_one_choice = statements_for_one_choice[statement_for_one_choice_idx]  #对应此选项的statement
            statements_to_choice.append(statement_for_one_choice)

        # 在此处对每个词进行预测
        num_choice = 5
        # print("statement_for_one_choice: ",statement_for_one_choice)
        # print(len(statements_to_choice))
        choice_idx = predict(statements_to_choice,num_choice)
        write_text.append([{'question':questions[i],'statements':statements_to_choice,'statements_for_one_choice':statements_for_one_choice,'choice_pre':str(choice_idx)}])

        pre_idx.append(choice_idx)
    return pre_idx,write_text

def select_s_with_Achoice_and_pre(statements,choices,questions):
    # statemente: [len(chocies)*statements_num_for_this_q]
    write_text = []
    print('begin select and predict')
    pre_idx = []
    for i in tqdm(range(len(statements))):
        this_choice = choices[i]
        # statement_for_one_question=[]
        statements_to_choice = []
        choice_text = choices[i][0]['text']
        statements_for_one_choice = []
        # choice_text = item['text']
        for sentence in statements[i]:
                # print("sentence : ",sentence)
            #todo 拼接后预测  如果mask在里面 替换mask
            if '[mask]' in sentence:
                statements_for_one_choice.append(sentence.replace('[mask]',choice_text))
            else:
                statements_for_one_choice.append(sentence+' '+ choice_text)
        statement_for_one_choice_idx = select(questions[i],statements_for_one_choice)
        # statement_for_choice = statements_for_one_choice[statement_for_one_choice_idx]  #对应此选项的statement
        statement_for_choice = statements[i][statement_for_one_choice_idx]  #对应此选项的statement
        statements_for_one_choice_res = []
        for choice in choices[i]:
            
            one_choice_text =choice['text']
            print(one_choice_text)
            if '[mask]' in statement_for_choice:
                statements_for_one_choice_res.append(statement_for_choice.replace('[mask]',one_choice_text))
            else:
                statements_for_one_choice_res.append(statement_for_choice+' '+ one_choice_text)
        # statements_to_choice.append(statements_for_one_choice_res)

        # 在此处对每个词进行预测
        num_choice = 5
        # print("statement_for_one_choice: ",statement_for_one_choice)
        # print(len(statements_to_choice))
        # print(statements_for_one_choice_res)
        # exit()
        choice_idx = predict(statements_for_one_choice_res,num_choice)
        write_text.append([{'question':questions[i],'statements':statements_for_one_choice_res,'statements_for_one_choice':statements_for_one_choice,'choice_pre':str(choice_idx)}])

        pre_idx.append(choice_idx)
    return pre_idx,write_text

#main
if __name__=='__main__':
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    questions,choices,answerkeys = getq(data_dir)
    # # prompts = get_prompt(questions,'zero')
    # prompts = get_prompt(questions,'few')
    # write_text = []
    # num = 0
    # statements_res = []
    # all_statements_res = []
    # idx_res = []
    # for item in prompts:
    #     # if  num == 1:
    #     #     break
    #     this_statements = []
    #     gen_nums = 10
    #     for i in range(gen_nums): 
    #     # while(len(this_statements)<1 ):
    #         statement = generate(item)

    #         if '[mask]' in statement:
    #             this_statements.append(statement)
    #         if i==gen_nums-1 and len(this_statements)==0:
    #             this_statements.append(statement)

    #             # 因为后面要用语言模型的得分判断 有mask分数会低于没有mask 所以过滤一下  那把mask换成answer呢？
        
    #         # print(' ')

    #     idx = select(this_statements)
    #     idx_res.append(idx)
    #     statements_res.append(this_statements[idx])
    #     all_statements_res.append(this_statements)
    #     # print('[statements_res]',statements_res)
    #     num += 1    
    
    # # 用转换过的陈述句 拼接上答案 去选择
    # # print('1111111',statements_res,choices)
    # # print('【all statements: 】',all_statements_res)
    # statement_w_c = con_s_c(statements_res,choices[:len(statements_res)])
    # # 预测
    # write_text , pre = pre_with_sc(statement_w_c,questions,all_statements_res,idx_res,statements_res)
    # # sc_accuracy = accuracy_score(answerkeys, pre)
    # # print('acc: ',sc_accuracy)

    # write_dir = './outputfile/s.json'
    # with jsonlines.open(write_dir,'w') as f:
    #     for item in all_statements_res:
    #         f.write(item)
    #     f.close()
    # write_idx_dir = './outputfile/s_idx.json'
    # write_res_text(write_idx_dir,write_text)

    #read out
    # file_dir = '/users5/znchen/Question2Knowledge/SearchQasP/out/gens.out'
    # get_s_from_out(file_dir)
    

    res_idx_dir = './outputfile/s.json'
    choice_pre = []
    statements = []
    with jsonlines.open(res_idx_dir,'r') as f:
        for line in f:
            choice_pre.append(int(line[0]["choice_pre"]))
            statements.append(line[0]["statements"])
            # statement_pre
    new_statements = []
    new_statement = [] # 每个问题对应一个
    idx_res = []
    num = 0
    for i in tqdm(range(len(questions))):
        # if num==2:
        #     break
        s = statements[i]
        # 对生成的statement做处理之后 选得分最高的
        state_for_one_q = []
        for item in s:
            if '<|endoftext|>' in item:
                one_s = item[:item.find('<|endoftext|>')]
            elif '\n' in item:
                one_s = item[:item.find('\n')]
            state_for_one_q.append(one_s) 
        statement_idx = select(questions[i],state_for_one_q)
        idx_res.append(statement_idx)
        new_statements.append(state_for_one_q)
        print('state_for_one_q[statement_idx]: ',state_for_one_q[statement_idx])
        new_statement.append(state_for_one_q[statement_idx])
        num += 1
    # 用选项填充后选择每个选项中最大的 再去比较
    # print("new_statements: ",new_statements)
    # pre_idx,write_text = select_statement_with_choice_and_pre(new_statements,choices,questions)
    
    # # 先选statement，再把选项放进去预测
    statement_w_c = con_s_c(new_statement,choices[:len(new_statements)])
    write_text , pre_idx = pre_with_sc(statement_w_c,questions,new_statements,idx_res,new_statement)

    # 用一个选项拼进去选择一个statement 再把其他选项放进来去预测
    # pre_idx,write_text = select_s_with_Achoice_and_pre(new_statements,choices,questions)

    sc_accuracy = accuracy_score(answerkeys, pre_idx)
    print('acc: ',sc_accuracy)
    write_idx_dir = './outputfile/select_statement_with_choice_and_pre0629.json'
    write_res_text(write_idx_dir,write_text)
