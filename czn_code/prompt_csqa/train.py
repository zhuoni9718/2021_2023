#%%
# from transformers import DebertaV2Tokenizer, DebertaV2Model,DebertaV2PreTrainedModel
# from transformers import RobertaTokenizer, RobertaForMultipleChoice,RobertaModel
import torch
import jsonlines
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    for i in range(len(q_id)):
        id_k[q_id[i]] = k[i]
    return id_k

def getacc(res,label):
    idx_word=['A','B','C','D','E']
    right_num=0
    for i in range(len(label)):
        if label[i]==idx_word[res[i]]:
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

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer
if __name__=='__main__':
    data_dir="/users5/znchen/Question2Knowledge/train_rand_split.jsonl"
    # k_dir="/users5/znchen/Question2Knowledge/SearchQasP/unfixed_k_only.txt"
    k_dir = "/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_fixed_promt.txt"
    id_k = getk_with_id(data_dir,k_dir)
    # print(type(id_k))#dict
    csqa_data=dataloader(data_dir)
    
    batchsize=128
    #仅预测，所以不打乱 shuffle=False
    csqa_loader = DataLoader(dataset=csqa_data, batch_size=batchsize, shuffle=False,num_workers=0) 

    device = 0
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")
    model, tokenizer = init_model('facebook/bart-base', device)

    # tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    # model = RobertaModel.from_pretrained("roberta-large")
    # model = RobertaForMultipleChoice.from_pretrained("roberta-large")
    # model = model.to(device)
    # tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-base")
    # model = DebertaV2ForMultipleChoice.from_pretrained("microsoft/deberta-v2-base")
    
    pre=[]
    num=0
    label=[]
    
    for i,data  in enumerate(data_loader, 0):
    
        inputs=[]
        num=num+1
        # print()
        if num>100:
            print(num)
            break
        
        question = item["question"]["stem"]
        choices = item["question"]["choices"]
        label.append(item["answerKey"])
        k_id =  item["id"]
        this_k=id_k[k_id]
        this_q_c=[]
        for i in range(5):
            #拼接方式是 Q+C+K 还是Q+k+C呢     Q+C+K 效果不好
            # input_text = "[CLS]"+" "+question+" " +"[SEP]"+choices[i]["text"]
            # input_text = "[CLS]"+" "+question+" " +"[SEP]"+" "+this_k+" "+'[SEP]'+" "+choices[i]["text"]
            # 只使用Q C 拼接来预测（baseline
            input_text = "[CLS]"+" "+question+" " +'[SEP]'+choices[i]["text"]
            input_token = tokenizer.tokenize(input_text)
            # input_ids = tokenizer.convert_tokens_to_ids(input_token)
            
            input_ids = tokenizer.encode(input_text,padding = 'max_length')
            this_q_c.append(input_ids)
        # inputs.append(this_q_c)
        inputs = this_q_c 
        inputs_tensor = torch.tensor(inputs)
        # inputs_mask_ids = get_mask_ids(inputs_tensor)
        # inputs_segment_ids = get_segment_ids(inputs_mask_ids)

        # outputs = model(inputs_tensor,attention_mask = inputs_mask_ids, token_type_ids = inputs_segment_ids)
        outputs = model(inputs_tensor)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logits = outputs.last_hidden_state
        y=np.argmax(logits.detach().numpy())
        pre.append(y)
        break

    print("begin cal acc")

    # acc = getacc(pre,label)
    # print("acc: ",acc)
    # idx_word=['A','B','C','D','E']

    # with open("pre_wo_k.txt","w") as f:
    #     for key in range(len(pre)) :
    #         f.write(idx_word[key]+'\n')




 # %%
