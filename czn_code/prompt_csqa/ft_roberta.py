from transformers import RobertaForMultipleChoice
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AdamW
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import jsonlines
from tqdm import tqdm
import os
import wandb
import random
import datetime
name = datetime.datetime.now().strftime(r'%Y%m%d%H%M%S')
def get_CSQA_sequence(data_dir,k_dir):
    data = []
    labels = []
    return data,labels

# 超参数


#随机数种子
seed_val = 42
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 模型
def initmodel(modelname = 'roberta-base'):
    device = 0
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu") 
    model = RobertaForMultipleChoice.from_pretrained(modelname,random_seed=seed_val)
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained(modelname)
    return model,device,tokenizer

class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        data = []
        self.question,self.options,self.labels = [],[],[]
        i = -1

        with jsonlines.open(data_path,'r') as f:
            for line in tqdm(f):
                i+=1
                data.append(line)
                # if i>1000:
                #     break
                if isinstance(line, list):
                    print(line)
                    print(i)
                    continue

                context = line['data:']
                if "knowledge:" in line:
                    self.question.append( '[CLS] ' + line["knowledge:"][0] +' [SEP] Q: '+context["question"]["stem"] )
                else:
                    self.question.append( '[CLS] ' + line["knowledge: "][0] +' [SEP] Q: '+context["question"]["stem"] )
                # self.question.append(context["question"]["stem"] )
                self.options.append([text['text'] for text in line['data:']["question"]["choices"]])
                self.labels.append(['A','B','C','D','E'].index(line['data:']["answerKey"]))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        text = [self.question[idx]+' [SEP] A: '+choice for choice in self.options[idx]]
        options = self.options[idx]
        label = self.labels[idx]
        # print(text)
        # exit()
        # encoded_input = self.tokenizer(text, *options,return_tensors="pt")
        encoded_input = self.tokenizer(text,padding='max_length',return_tensors="pt")
        # print(encoded_input['input_ids'])
        # exit()
        label_tensor = torch.tensor(label, dtype=torch.float)
        return encoded_input['input_ids'].squeeze(), encoded_input['attention_mask'].squeeze(), label_tensor


class MyDatasetForDitlScorer(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        data = []
        self.question,self.options,self.labels,self.generated_k = [],[],[],[]
        i = -1
        with jsonlines.open(data_path,'r') as f:
            for line in f:
                i+=1
                data.append(line)
                # if i>1000:
                #     break
                if isinstance(line, list):
                    print(line)
                    print(i)
                    continue

                context = line['data']
                self.question.append(context["question"]["stem"] )
                # self.question.append(context["question"]["stem"] )
                self.options.append([text['text'] for text in line['data']["question"]["choices"]])
                self.labels.append(['A','B','C','D','E'].index(line['data']["answerKey"]))
                self.generated_k.append(line['res'])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        text = ['[CLS] '+self.question[idx]+' [SEP] '+choice for choice in self.options[idx]]
        text = [item + ' [SEP] '+self.generated_k[idx]+' [SEP]' for item in text]
        # print(text)
        # exit()
        options = self.options[idx]
        label = self.labels[idx]

        # encoded_input = self.tokenizer(text, *options,return_tensors="pt")
        encoded_input = self.tokenizer(text,padding='max_length',return_tensors="pt")
        # print(encoded_input['input_ids'])
        # exit()
        label_tensor = torch.tensor(label, dtype=torch.float)
        return encoded_input['input_ids'].squeeze(), encoded_input['attention_mask'].squeeze(), label_tensor


def main(model_name,train_path,val_path,test_path,batch_size,epoch,learning_rate,max_length,train_flag=True,test_flag=True):
    batch_size = batch_size
    num_epochs = epoch
    wandb.init(project="scorer",name = name)

    # 创建保存模型的目录
    save_dir = "./saved_models/scorer/"
    os.makedirs(save_dir, exist_ok=True)


    # 定义模型和优化器
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device: ',device)
    model = RobertaForMultipleChoice.from_pretrained(model_name)
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # 训练模型
    if train_flag:
        # train_dataset = MyDatasetForDitlScorer(train_path, tokenizer, max_length)
        # val_dataset = MyDatasetForDitlScorer(val_path, tokenizer, max_length)
        train_dataset = MyDataset(train_path, tokenizer, max_length)
        val_dataset = MyDataset(val_path, tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        #优化器
        adam_betas = (0.9, 0.98)
        adam_eps = 1e-05
        weight_decay = 0.01
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        decay_steps = 20
        power = 1.5
        end_learning_rate = 1e-6
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / decay_steps) ** power
                                  if epoch < decay_steps else end_learning_rate / optimizer.param_groups[0]['lr'])
        for epoch in range(num_epochs):
            train_loss = 0.0
            model.train()
            for batch in tqdm(train_loader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                preds = torch.argmax(outputs.logits,axis=1)
                loss = outputs.loss
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * labels.size(0)
            train_loss /= len(train_dataset)
            wandb.log({'Epoch':epoch+1,"train_loss": train_loss})

            val_loss = 0.0
            num_correct = 0
            num_total = 0

            model.eval()

            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                with torch.no_grad():
                
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    preds = torch.argmax(outputs.logits,axis=1)
                    loss = outputs.loss

                val_loss += loss.item() * labels.size(0)

                # preds = torch.round(torch.sigmoid(outputs.logits))
                #需要打出来这里的维度 输出看一下
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

            val_loss /= len(val_dataset)
            accuracy = num_correct / num_total

            print(f'Epoch {epoch+1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val accuracy: {accuracy:.4f}')
            # modelpath = './modeltmp/'+str(epoch)+'.pth'
            wandb.log({'Epoch':epoch+1,"val_loss": val_loss, "val_acc": accuracy})
            modelpath = os.path.join(save_dir, str(epoch)+"ftmodel.pt")
            print("saving to ",modelpath)
            torch.save(model.state_dict(), modelpath)




    if test_flag:
        test_dataset = MyDataset(test_path, tokenizer, max_length)
        # test_dataset = MyDatasetForDitlScorer(test_path, tokenizer, max_length)
        test_loader = DataLoader(test_dataset)
        load_model_path = os.path.join(save_dir+"2ftmodel.pt")
        if load_model_path:
            model_path = load_model_path
            model.load_state_dict(torch.load(model_path))
        test_loss = 0.0
        num_correct = 0
        num_total = 0
        model.eval()
        for batch in tqdm(test_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            test_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits,axis=1)
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)
        test_loss /= len(test_dataset)
        test_accuracy = num_correct / num_total

        print(f'Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}')
        wandb.log({"Test loss": test_loss, "test_accuracy": test_accuracy})


def DistilMain(train_path,val_path,test_path,batch_size,epoch,learning_rate,max_length,train_flag=True,test_flag=True):
    batch_size = batch_size
    num_epochs = epoch
    # 创建保存模型的目录
    wandb.init(project="Distilscorer",name = name)
    save_dir = "saved_models/Distilscorer/"
    os.makedirs(save_dir, exist_ok=True)


    # 定义模型和优化器
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device: ',device)
    model = RobertaForMultipleChoice.from_pretrained('roberta-base')
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # 训练模型
    if train_flag:
        train_dataset = MyDatasetForDitlScorer(train_path, tokenizer, max_length)
        val_dataset = MyDatasetForDitlScorer(val_path, tokenizer, max_length)
  
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        #优化器
        adam_betas = (0.9, 0.98)
        adam_eps = 1e-05
        weight_decay = 0.01
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        decay_steps = 20
        power = 1.5
        end_learning_rate = 1e-6
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / decay_steps) ** power
                                  if epoch < decay_steps else end_learning_rate / optimizer.param_groups[0]['lr'])
        for epoch in range(num_epochs):
            train_loss = 0.0
            model.train()
            for batch in tqdm(train_loader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                preds = torch.argmax(outputs.logits,axis=1)
                loss = outputs.loss
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * labels.size(0)
            train_loss /= len(train_dataset)
            wandb.log({'Epoch':epoch+1,"train_loss": train_loss})

            val_loss = 0.0
            num_correct = 0
            num_total = 0

            model.eval()

            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                with torch.no_grad():
                
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    preds = torch.argmax(outputs.logits,axis=1)
                    loss = outputs.loss

                val_loss += loss.item() * labels.size(0)

                # preds = torch.round(torch.sigmoid(outputs.logits))
                #需要打出来这里的维度 输出看一下
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

            val_loss /= len(val_dataset)
            accuracy = num_correct / num_total

            print(f'Epoch {epoch+1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val accuracy: {accuracy:.4f}')
            # modelpath = './modeltmp/'+str(epoch)+'.pth'
            wandb.log({'Epoch':epoch+1,"val_loss": val_loss, "val_acc": accuracy})
            modelpath = os.path.join(save_dir, str(epoch)+"ftmodel.pt")
            print("saving to ",modelpath)
            torch.save(model.state_dict(), modelpath)




    if test_flag:
        # test_dataset = MyDataset(test_path, tokenizer, max_length)
        test_dataset = MyDatasetForDitlScorer(test_path, tokenizer, max_length)
        test_loader = DataLoader(test_dataset)
        load_model_path = os.path.join(save_dir, +"2ftmodel.pt")
        if load_model_path:
            model_path = load_model_path
            model.load_state_dict(torch.load(model_path))
        test_loss = 0.0
        num_correct = 0
        num_total = 0
        model.eval()
        for batch in tqdm(test_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            test_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits,axis=1)
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)
        test_loss /= len(test_dataset)
        test_accuracy = num_correct / num_total

        print(f'Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}')
        wandb.log({"Test loss": test_loss, "test_accuracy": test_accuracy})

if __name__=='__main__':
    batch_size = 16
    epoch = 8
    max_length = 256
    model_name = 'roberta-large'
    learning_rate = 1e-5
    # 微调基本模型
    train_path = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_qk_train.txt'
    val_path = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_qk_dev.txt'
    test_path = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/k_q2k_t5.jsonl'
    main(model_name ,train_path,val_path,test_path,batch_size,epoch,learning_rate,max_length,train_flag=True,test_flag=True)

    # 蒸馏scorer
    # train_path = '/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/ftscorer/ftscorer.train.jsonl'
    # val_path = '/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/input/csqa/ftscorer/ftscorer.val.jsonl'
    # test_path = '/users5/znchen/commongen/CommonGen-master/methods/BART/fairseq_local/res/csqa2answerK.test.jsonl'
    # DistilMain(train_path,val_path,test_path,batch_size,epoch,learning_rate,max_length,train_flag=True,test_flag=True)