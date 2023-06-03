#TODO:写一个id:k 的文件
#%%
from tqdm import tqdm
import json
import jsonlines
def id_k():
    k=[]
    q_id=[]
    with jsonlines.open("/users5/znchen/Question2Knowledge/train_rand_split.jsonl",'r') as f:
        for line in f:
            q_id.append(line["id"])
    with open("/users5/znchen/Question2Knowledge/SearchQasP/unfixed_k_only.txt",'r') as f:
        for line in f:
            k.append(line)
    id_k={}
    for i in range(len(q_id)):
        id_k[q_id[i]] = k[i]
        # print(id_k[q_id[i]])
        # exit()

    # for keyvalue in id_k:
    #     print(item)
    #     exit()
    wri = json.dumps(id_k)
    with jsonlines.open("./k_withid.json",'w') as f:
        f.write(wri)
    
def getquestion_and_k():
    q=[]
    k=[]
    with jsonlines.open("/users5/znchen/Question2Knowledge/train_rand_split.jsonl",'r') as f:
        for line in f:
            q.append(line["question"]["stem"])
    # for i in range(60):
    #     print(q[i])
    with open("./k_only.txt","r") as f:
        for line in f:
            # print(line)
            # exit()
            k.append(line)

    with open("./only_question","w") as f:
        for i in tqdm(range(100)):
            # print(item)
            idx = i*5
            s = q[idx]+"\n"+k[idx]+"\n"+"\n"
            f.write(s)

def get_s_For_q():
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    s_dir = '/users5/znchen/yqsun/CommonsenseQA/zero_shot_cqa/data/dev.tgt'
    s = []
    res = []
    with open(s_dir,'r') as f:
        for line in f:
            s.append(line)
    with jsonlines.open(data_dir,'r') as f:
        num = 0
        for line in tqdm(f):
            # line = f[i]
            
            line['statements_with_mask'] = s[num]
            res.append(line)
            num += 1 
    write_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement_w_mask.jsonl'
    with jsonlines.open(write_dir,'w') as f :
        for item in tqdm(res):
            f.write(item)


if __name__=="__main__":
    # getquestion_and_k()
    get_s_For_q()


 # %%
