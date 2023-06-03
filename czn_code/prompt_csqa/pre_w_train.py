from cProfile import label
from transformers import BertTokenizer, BertForMultipleChoice
import torch
import numpy as np
from utils import readCsqa,read_k,concarate
from tqdm import tqdm
# 定义模型
device = 0
device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
model.to(device)
learnrate=1e-6
optimizer=torch.optim.Adam(model.parameters(),lr=learnrate,betas=(0.9,0.99),eps=1e-08,weight_decay=1e-5,amsgrad=False)


def accuracy(out, labels,batchsize):
    #outputs = np.argmax(out) #取出out中最大的索引值
    outputs = np.argmax(out,axis=1) 

    num=0
    for i in range(batchsize):
        if outputs[i]==labels[i]:
            num=num+1
    #print(num)
    return num
    
def runOneepoch(data,train=True,labels = []):
    running_loss = 0.0
    trainaccnum=0.0
    thisepochloss=[]
    inputids=torch.Tensor()
    labels_array=np.array(labels)
    labeltensor=torch.tensor(labels_array.astype(float))
    # print(type(labeltensor))
    print("begin tokenize")
    datas = []
    for item in data:
        # print(item)
        for sentence in item:
            # print(len(sentence.split()))
            datas.append(sentence)
    input_tensor = tokenizer(datas,return_tensors="pt", padding="max_length", max_length=512,truncation=True)["input_ids"]
    # print(input_tensor.shape) # torch.Size([6105, 612])
    input_tensor = input_tensor.reshape(-1,5,input_tensor.shape[1])
    
    # print(input_tensor.shape)
    print("begin training")
    # inputids =  torch.tensor(np.array(inputids))  
    # return 0,0
    if (train==True):
        #print(labletensor)
        # exit()
        # outputs = model(inputidstensor.to(device), attention_mask =mask_ids.to(myconfig.device),token_type_ids =segment_ids.to(myconfig.device),labels=labletensor.to(myconfig.device))
        # 所有数据肯定放不下，需要分batch
        # print('input : ',input_tensor.shape,'label : ',labeltensor.shape)
        outputs = model(input_tensor.to(device),labels = labeltensor.to(device))
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        with torch.no_grad():
            outputs = model(input_tensor.to(device), labels=labeltensor.to(device))                
        loss = outputs.loss
            
    # print("loss: ",loss)
    # thisepochloss.append(loss)
    logits = outputs.logits
    thisbatchsize = len(data)
    tmp_accuracy = accuracy(logits.detach().cpu(),labeltensor,thisbatchsize)
    print('this batch acc: ',tmp_accuracy)
    # trainaccnum += tmp_accuracy
    # print(trainaccnum)
    # minloss=min(thisepochloss)        
    return tmp_accuracy,loss

def train(data,labels,batchsize):
    train=True
    acc_sum = 0
    for i in range(len(data)//batchsize+1):
    # for i in range(1):
        batchdata = data[i:i+batchsize]
        batchlabels = labels[i:i+batchsize]
        print('batch idx: ',i)
        acc,loss = runOneepoch(batchdata,train,batchlabels)
        acc_sum += acc
    print("train acc: ",acc_sum/len(data))

def test(data,labels,batchsize):
    train=False
    acc_sum = 0
    for i in range(len(data)//batchsize+1):
    # for i in range(1):
        batchdata = data[i:i+batchsize]
        batchlabels = labels[i:i+batchsize]
        print('batch idx: ',i)
        acc,loss = runOneepoch(batchdata,train,batchlabels)
        acc_sum += acc
    print("test acc: ",acc_sum/len(data))
    

# 输入是句子 输出是分类
if __name__=='__main__':


    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/train.statement.jsonl'
    # data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    data = readCsqa(data_dir)
    k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_logic_abs_512.train_transfered.jsonl'
    # k_dir = ''
    k = read_k(k_dir)
    k_num = 1
    mode = 'bert'
    concarated_data,labels  = concarate(data,k,k_num,mode)
    # print(concarated_data)
    batchsize = 16
    # train
    train(concarated_data,labels,batchsize)
    # test
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    data = readCsqa(data_dir)
    k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_logic_abs_512.dev_transfered.jsonl'
    # k_dir = ''
    k = read_k(k_dir)
    k_num = 1
    mode = 'bert'
    concarated_data,labels  = concarate(data,k,k_num,mode)
    test_data = concarated_data
    test(test_data,labels,batchsize)

