import jsonlines
from tqdm import tqdm

data_dirs = ['train.gpt.jsonl','valid.gpt.jsonl','test.gpt.jsonl']
# data_dirs = ['/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/res/scorer.testwok.jsonl']
# data_dir = ['/users5/znchencd/commongen/CommonGen-master/methods/BART/fairseq_local/res/scorerWOCOT.testwok.jsonl']
for dir in data_dirs:
    print(dir)
    write_context = []
    gptpredict,answerr = [],[]
    count,sum = 0,0
    res_dir = 'right.'+dir
    with jsonlines.open(dir,'r') as f:
        for line in tqdm(f):
            # 读正确答案  读gpt预测的答案 
            sum+=1
            answerkey = ['A','B','C','D','E'].index(line['data']['answerKey'])
            answer = line['data']['question']['choices'][answerkey]['text']
            gptpredict = line["k"].split('\n')[0]
            # gptpredict = line['res'].strip()
            # 判断是否一致
            answer = answer.strip()
            gptpredict = gptpredict.strip()
            if answer==gptpredict:
                count+=1
                write_context.append(line)
        print(sum)
        print(count/sum)
    # with jsonlines.open(res_dir,'w') as f:
    #     for item in tqdm(write_context):
    #         f.write(item)

# train
# 8520
# 0.7272300469483568

# valid.gpt.jsonl
# 1221
# 0.7608517608517609

# test.gpt.jsonl
# 1221
# 0.7592137592137592