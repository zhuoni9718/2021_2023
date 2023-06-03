#%%
import torch
import jsonlines
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
# from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
def dataloader(dir):
    with jsonlines.open(dir,'r') as f:
        obqa_data=[]
        for line in f:
            obqa_data.append(line)
    return obqa_data
        
class obqaData(Dataset):
    def __init__(self,filename):
        obqa_data = dataloader(filename)
        # print(x)
        # print(y)
        self.len=len(csqa_datq)
        # print(self.len)
        self.x_data=obqa_data['question']
        self.y_data=obqa_data['answerKey']

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
    for i in range(len(q_id)):
        id_k[q_id[i]] = k[i]
    return id_k

def getacc(res,label):
    idx_word=['A','B','C','D','E']
    right_num=0
    for i in range(len(label)):
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



def init_model(model_name: str,device: torch.device):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    # logger.info(f'Initializing {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = tokenizer.unk_token

    # model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

#%%
if __name__=='__main__':
    #用obqa的语料预训练生成模型  
    pre_train_data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/obqa_Additional/train_complete.jsonl'

    # data_dir="/users5/znchen/Question2Knowledge/train_rand_split.jsonl"
    # k_dir="/users5/znchen/Question2Knowledge/SearchQasP/unfixed_k_only.txt"
    # k_dir = "/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_fixed_promt.txt"
    # id_k = getk_with_id(data_dir,k_dir)
    # print(type(id_k))#dict
    obqa_data=dataloader(pre_train_data_dir)
    lr = 2e-5
    warmup_steps=20
    batchsize=32
    #仅预测，所以不打乱 shuffle=False
    obqa_loader = DataLoader(dataset=obqa_data, batch_size=batchsize, shuffle=False,num_workers=0) 
    device = 0
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")
    # model, tokenizer = init_model('gpt2-large', device)
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # model = GPT2LMHeadModel.from_pretrained('gpt2')

    
    pre=[]
    num=0
    label=[]
    source=[]
    target = []
    for items in tqdm(obqa_loader):
        # print(items)

        question = items["question"]["stem"]
        fact = items['fact1']
        # source.append(question)
        # target.append(fact)
    
    #     batch = []

        input_token = tokenizer(question, return_tensors="pt", padding=True)["input_ids"]
        input_label = tokenizer(fact, return_tensors="pt", padding=True)["input_ids"]
        print(input_token.shape)  #torch.Size([32, 42])
        print(input_label.shape)  #torch.Size([32, 28])
        exit()
        '''
        model = model.cuda()
        model.train()
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
        )

    #     # outputs = model(inputs_tensor,attention_mask = inputs_mask_ids, token_type_ids = inputs_segment_ids)
        outputs = model(input_token.to(device),labels = input_label.to(device))
        loss = outputs.loss
        print(loss)
        loss.backward()


    #     y=np.argmin(loss)
    #     pre.append(y)
        

    # print("begin cal acc")
    # accuracy = accuracy_score(label, pre)
    # print(f"Accuracy: {accuracy:.3f}")
    # acc = getacc(pre,label)
    # print("acc: ",acc)
    # idx_word=['A','B','C','D','E']

    # with open("pre_wo_k.txt","w") as f:
    #     for key in (pre) :
    #         f.write(idx_word[key]+'\n')


'''

# %%

# %%
