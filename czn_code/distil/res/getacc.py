import jsonlines
from tqdm import tqdm
def getacc(data_dir,res_dir):
    sum,count = 0,0
    res = []
    with open(res_dir,'r') as f:
        for item in f.readlines():
            res.append(item)
    with jsonlines.open(data_dir ,'r') as f:
        for line in tqdm(f):
            answerkey = ['A','B','C','D','E'].index(line['data']['answerKey'])
            answer = line['data']['question']['choices'][answerkey]['text']
            # gptpredict = line["k"].split('\n')[0]
            gptpredict = res[sum][1:-2]
            # 判断是否一致
            answer = answer.strip().lower()
            # answer = answer.strip()
            gptpredict = gptpredict.strip()
            while('<pad>' in gptpredict):
                gptpredict = gptpredict.replace('<pad>','')
            print(gptpredict,'|',answer)
            print(gptpredict in answer or answer==gptpredict)
            if gptpredict in answer or answer==gptpredict:
            # if gptpredict in answer:
                count+=1
                # write_context.append(line)
            sum+=1
            
    print(sum)
    print(count/sum)
    return

def get_a_acc(data_dir,res):
    sum,count = 0,0
    with jsonlines.open(data_dir ,'r') as f:
        for line in tqdm(f):
            answerkey = ['A','B','C','D','E'].index(line['data']['answerKey'])
            answer = line['data']['question']['choices'][answerkey]['text']
            # gptpredict = line["k"].split('\n')[0]
            gptpredict = res[sum]
            # 判断是否一致
            answer = answer.strip().lower()
            # answer = answer.strip()
            gptpredict = gptpredict.strip().lower()
            while('<pad>' in gptpredict):
                gptpredict = gptpredict.replace('<pad>','')
            print(gptpredict,'|',answer)

            if gptpredict in answer or answer==gptpredict:
            # if gptpredict in answer:
                count+=1
                # write_context.append(line)
            sum+=1
            
    print(sum)
    print(count/sum)
    acc = count/sum
    return acc



if __name__=='__main__':
    data_dir = '/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/test.gpt.jsonl'
    res_dir = '/users5/znchen/distil/res/t5large/RA0train.res'
    res_dir = '/users5/znchen/distil/res/t5large/WOCOT3train.res'
    getacc(data_dir,res_dir)