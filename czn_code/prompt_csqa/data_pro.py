#%%
import jsonlines
from tqdm import tqdm
res = []
with jsonlines.open("obqa_prompt.jsonl",'r') as f:
    for line in f:
        print(line[0])
        res.append(line[0].replace("_",""))
with jsonlines.open("obqa_prompt.jsonl",'w') as f:
    for key in tqdm(res):
        f.write(key)


# %%
