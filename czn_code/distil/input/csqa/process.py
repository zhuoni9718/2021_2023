import math
import jsonlines
train_dir = '/users5/znchen/Question2Knowledge/SearchQasP/ft/gpt/data/qk/train.txt'
def process1(data_dir):
    train_src = './train.src.txt'
    train_tgt = './train.tgt.txt'
    val_src = './val.src.txt'
    val_tgt = './val.tgt.txt'
    test_src = './test.src.txt'
    test_tgt = './test.tgt.txt'
    source,target = [] , []
    with open(train_dir,'r') as f:
        for line in f.readlines():
            line = line.split('=')
            # print(line)
            # exit()
            source.append(line[0])
            target.append(line[1])
    index1 = math.ceil(len(source)*0.8)
    index2 = math.ceil(len(source)*0.9)
    # with open(train_src,'w') as f:
    #     for item in source[:index1]:
    #         f.write(item.strip()+'\n')
    # with open(train_tgt,'w') as f:
    #     for item in target[:index1]:
    #         f.write(item.strip()+'\n')

    with open(val_src,'w') as f:
        for item in source[index1:index2]:
            f.write(item.strip()+'\n')
    with open(val_tgt,'w') as f:
        for item in target[index1:index2]:
            f.write(item.strip()+'\n')

    with open(test_src,'w') as f:
        for item in source[index2:]:
            f.write(item.strip()+'\n')
    with open(test_tgt,'w') as f:
        for item in target[index2:]:
            f.write(item.strip()+'\n')

def process(data_dir,res_src_dir,res_tgt_dir):
    with jsonlines.open(data_dir,'r') as f:
        data = []
        write_context = []
        data_src,data_tgt = [],[]
        for line in f:
            data.append(line)
        noanswer_num=0
        for item in data:
            try:
                k_context = item['k'].split('\n')[1].strip('Knowledge: ')
                write_context.append({'data':item,'k:':k_context,'gpt_answer':item['k'].split('\n')[0]})
            except:
                noanswer_num+=1
                # if item['k'].startswith('There is not enough information') or item['k'].startswith('None of the choices accurately'):
                print(item['k'])
                continue
            # print(k_context)
                # exit()
            data_src.append(item['data']['question']['stem'])
            data_tgt.append(k_context)
        print(noanswer_num)
        
        writefile(res_src_dir,data_src)
        writefile(res_tgt_dir,data_tgt)
    if data_dir=='/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/test.gpt.jsonl':
        with jsonlines.open('processed_train.data.jsonl','w') as f:
            for item in write_context:
                f.write(item)
    return 

def process_answer_w_chain(data_dir,res_src_dir,res_tgt_dir):
    with jsonlines.open(data_dir,'r') as f:
        data = []
        write_context = []
        data_src,data_tgt = [],[]
        for line in f:
            data.append(line)
        noanswer_num=0
        for item in data:
            try:
                # k_context = item['k']
                item['k'].strip('Knowledge: ')
                k_context = item['k'].replace('\n',' | ')

                write_context.append({'data':item,'k:':k_context,'gpt_answer':item['k'].split('\n')[0]})
            except:
                noanswer_num+=1
                # if item['k'].startswith('There is not enough information') or item['k'].startswith('None of the choices accurately'):
                print(item['k'])
                continue
            # print(k_context)
                # exit()
            data_src.append(item['data']['question']['stem']+' Choices: '+
            item['data']['question']['choices'][0]['text']+
            ' | '+item['data']['question']['choices'][1]['text']+
            ' | '+item['data']['question']['choices'][2]['text']+
            ' | '+item['data']['question']['choices'][3]['text']+
            ' | '+item['data']['question']['choices'][4]['text'])
            data_tgt.append(k_context)
        print(noanswer_num)
        
        writefile(res_src_dir,data_src)
        writefile(res_tgt_dir,data_tgt)
    if data_dir=='/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/test.gpt.jsonl':
        with jsonlines.open('ck/processed_train.data.jsonl','w') as f:
            for item in write_context:
                f.write(item)
    return 


def process_chain_answer(data_dir,res_src_dir,res_tgt_dir):
    with jsonlines.open(data_dir,'r') as f:
        data = []
        write_context = []
        data_src,data_tgt = [],[]
        for line in f:
            data.append(line)
        noanswer_num=0
        for item in data:
            try:
                k_context = item['k'].split('\n')[1].strip('Knowledge: ')+' The answer is '+item['k'].split('\n')[0]

                write_context.append({'data':item,'k:':k_context,'gpt_answer':item['k'].split('\n')[0]})
            except:
                noanswer_num+=1
                # if item['k'].startswith('There is not enough information') or item['k'].startswith('None of the choices accurately'):
                print(item['k'])
                continue
            # print(k_context)
                # exit()
            data_src.append(item['data']['question']['stem']+' Choices: '+
            item['data']['question']['choices'][0]['text']+
            ' | '+item['data']['question']['choices'][1]['text']+
            ' | '+item['data']['question']['choices'][2]['text']+
            ' | '+item['data']['question']['choices'][3]['text']+
            ' | '+item['data']['question']['choices'][4]['text'])
            data_tgt.append(k_context)
        print(noanswer_num)
        
        writefile(res_src_dir,data_src)
        writefile(res_tgt_dir,data_tgt)
    if data_dir=='/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/test.gpt.jsonl':
        with jsonlines.open('ck/processed_test.data.jsonl','w') as f:
            for item in write_context:
                f.write(item)
    return 
    

# 用于训练打分器
def process_scorer_data(data_dir,res_src_dir,res_tgt_dir):
    with jsonlines.open(data_dir,'r') as f:
        data = []
        write_context= []
        data_src,data_tgt = [],[]
        for line in f:
            data.append(line)
        noanswer_num=0
        for item in data:
            try:
                # k_context = item['k']
                k_context = item['k'].split('\n')[1].strip('Knowledge: ')
                write_context.append({'data':item,'k:':k_context,'gpt_answer':item['k'].split('\n')[0]})
            except:
                noanswer_num+=1
                # if item['k'].startswith('There is not enough information') or item['k'].startswith('None of the choices accurately'):
                print(item['k'])
                continue
            # print(k_context)
                exit()
            options = ','.join([item['data']['question']['choices'][i]['text'] for i in range(5)])
            data_src.append(item['data']['question']['stem']+' '+ k_context+' Choose the answer from the given options. Options: '+options+' The answer is')

            answerkey = ['A','B','C','D','E'].index(item['data']["answerKey"])
            data_tgt.append(item['data']['question']['choices'][answerkey]['text'])
        print(noanswer_num)

        writefile(res_src_dir,data_src)
        writefile(res_tgt_dir,data_tgt)
    if data_dir=='/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/test.gpt.jsonl':
        with jsonlines.open('ck/processed_train.data.jsonl','w') as f:
            for item in write_context:
                f.write(item)
    return 

def writefile(res_dir,context):
    with open(res_dir,'w') as f:
        for item in context:
            f.write(item+'\n')
    print("write to:",res_dir)

def seperate_k(data_dir,res_dir):
    context = []

    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            data = line
            try:
                data['res'] = line['k'].split('\n')[1].strip('Knowledge: ')
                context.append(data)

            except:
                continue
    with jsonlines.open(res_dir,'w') as f:
        for item in context:
            f.write(item)
    print('write to:',res_dir)
    return


if __name__=='__main__':
    train_data_dir = '/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/train.gpt.jsonl'
    val_data_dir = '/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/valid.gpt.jsonl'
    test_data_dir = '/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/test.gpt.jsonl'
    
    # # src：q  tgt：k
    # train_src = './train.src.txt'
    # train_tgt = './train.tgt.txt'
    # val_src = './val.src.txt'
    # val_tgt = './val.tgt.txt'
    # test_src = './test.src.txt'
    # test_tgt = './test.tgt.txt'
    # process(train_data_dir,train_src,train_tgt)
    # process(val_data_dir,val_src,val_tgt)
    # process(test_data_dir,test_src,test_tgt)

    # # src:q c tgt:c\nk
    # train_src = './ck/train.src.txt'
    # train_tgt = './ck/train.tgt.txt'
    # val_src = './ck/val.src.txt'
    # val_tgt = './ck/val.tgt.txt'
    # test_src = './ck/test.src.txt'
    # test_tgt = './ck/test.tgt.txt'
    # process_answer_w_chain(train_data_dir,train_src,train_tgt)
    # process_answer_w_chain(val_data_dir,val_src,val_tgt)
    # process_answer_w_chain(test_data_dir,test_src,test_tgt)

    # # src:q c tgt:k the answer is xxx
    # train_src = './ka/train.src.txt'
    # train_tgt = './ka/train.tgt.txt'
    # val_src = './ka/val.src.txt'
    # val_tgt = './ka/val.tgt.txt'
    # test_src = './ka/test.src.txt'
    # test_tgt = './ka/test.tgt.txt'
    # process_chain_answer(train_data_dir,train_src,train_tgt)
    # process_chain_answer(val_data_dir,val_src,val_tgt)
    # process_chain_answer(test_data_dir,test_src,test_tgt)

    # src:q c the answer is tgt:a
    train_src = './scorerdata/train.src.txt'
    train_tgt = './scorerdata/train.tgt.txt'
    val_src = './scorerdata/val.src.txt'
    val_tgt = './scorerdata/val.tgt.txt'
    process_scorer_data(train_data_dir,train_src,train_tgt)
    process_scorer_data(val_data_dir,val_src,val_tgt)
    
    #把train 和dev中k的推理链条摘出来
    # train_resdir = './ftscorer/ftscorer.train.jsonl'
    # val_resdir = './ftscorer/ftscorer.val.jsonl'
    # seperate_k(train_data_dir,train_resdir)
    # seperate_k(val_data_dir,val_resdir)