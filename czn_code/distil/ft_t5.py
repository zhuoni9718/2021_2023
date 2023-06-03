import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import jsonlines
import wandb
from tqdm import tqdm
import datetime
import os
from torch.multiprocessing import spawn
from torch.utils.data.distributed import DistributedSampler
from dataloader import DistlCSQADatasetRA,DistlCSQADataset,DistlCSQADatasetR

def wrapper(local_rank, *args):
    finetune_t5_with_wandb_ddp(local_rank, *args)

def run_ddp(train_dir, val_dir, model_name="t5-small", batch_size=8, num_epochs=3, learning_rate=5e-5):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    n_gpus = torch.cuda.device_count()
    spawn(wrapper, args=(train_dir, val_dir, model_name, batch_size, num_epochs, learning_rate), nprocs=n_gpus, join=True)


def finetune_t5_with_wandb(dataLoader_key,train_dir,val_dir , model_name="t5-small", batch_size=8, num_epochs=3, learning_rate=5e-5):
    # 初始化wandb
    wandbname = datetime.datetime.now().strftime(r'%Y%m%d%H%M%S')
    wandb.init(project="t5_finetuning",name = wandbname)

    # 加载预训练的T5模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    

    # 准备输入数据
    DataLoader_dict = {'RA':DistlCSQADatasetRA,'WOCOT':DistlCSQADataset,'R':DistlCSQADatasetR,'WOCOT-base':DistlCSQADataset}
    dataloader_set = DataLoader_dict[dataLoader_key]
    train_dataset = dataloader_set(train_dir, tokenizer)
    val_dataset = dataloader_set(val_dir, tokenizer)
    num_workers = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers)
    # exit()
    # 定义优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # 训练和验证模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    model.to(device)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # loss = outputs.loss
            loss = torch.mean(outputs.loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({"train_loss": avg_train_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                # loss = outputs.loss
                loss = torch.mean(outputs.loss)
            total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        wandb.log({"val_loss": avg_val_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        # 保存微调后的模型ft    
        model_path_dict = {'RA':'./tmp/t5/ra/','WOCOT':'./tmp/t5/wocot/','R':'./tmp/t5/r/'}
        model_path = model_path_dict[dataLoader_key]+"t5_finetuned_"+str(epoch)
        print('saving to ',model_path)
        model.save_pretrained(model_path)
    wandb.finish()
    return 

def test(dataLoader_key,model_epochnum,test_dir,batch_size,model_name):

    # 加载预训练的T5模型
    model_path_dict = {'RA':'./tmp/t5/ra/','WOCOT':'./tmp/t5/wocot/','R':'./tmp/t5/r/','WOCOT-base':'./tmp/t5/wocot-base/'}
    model_path = model_path_dict[dataLoader_key]+"t5_finetuned_"+str(model_epochnum)
    # print('--model path--:',model_path)
    # model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # 使用没有微调过的T5
    print('--init t5--')
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 加载相应的T5分词器
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    #准备数据
    DataLoader_dict = {'RA':DistlCSQADatasetRA,'WOCOT':DistlCSQADataset,'R':DistlCSQADatasetR}
    dataloader_set = DataLoader_dict[dataLoader_key]
    test_dataset = dataloader_set(test_dir, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 测试
    model.eval()
    answers = []
    for batch in tqdm(test_dataloader):
        # batch = {k: v.to(device) for k, v in batch.items()}
        # outputs = model(**batch)
        input_ids = batch["input_ids"]
        output = model.generate(input_ids, max_length=200, num_return_sequences=1, early_stopping=True)
        # 对模型生成的输出进行解码
        # print(output.size())
        for output_id in output:
            answer = tokenizer.decode(output_id, skip_special_tokens=True)
            print('--\n',answer)
            answers.append(answer)

        # exit()
    res_save_path = './res/t5/'+dataLoader_key+str(model_epochnum)+'.res'
    # res_save_path = './res/t5/init_t5.res'
    with jsonlines.open(res_save_path,'w') as f:
        print("writing to ",res_save_path)
        for item in answers:
            f.write(item)
    return
    



def finetune_t5_with_wandb_ddp(local_rank,train_dir,val_dir , model_name="t5-small", batch_size=8, num_epochs=3, learning_rate=5e-5):
    # 初始化wandb
    wandbname = datetime.datetime.now().strftime(r'%Y%m%d%H%M%S')
    wandb.init(project="t5_finetuning",name = wandbname)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=local_rank, world_size=torch.cuda.device_count())
    # 加载预训练的T5模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to("cuda:0")
    model = DDP(model, device_ids=[0], output_device=0)

    # 准备输入数据
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_dataset = DistlCSQADataset(train_dir, tokenizer)
    val_dataset = DistlCSQADataset(val_dir, tokenizer)
    num_workers = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,sampler=train_sampler,num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler,batch_size=batch_size,num_workers=num_workers)

    # 定义优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # 训练和验证模型
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    # model.to(device)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # loss = outputs.loss
            loss = torch.mean(outputs.loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({"train_loss": avg_train_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
            total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        wandb.log({"val_loss": avg_val_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        # 保存微调后的模型
        model.save_pretrained("./tmp/t5/t5_finetuned_"+str(epoch))
    wandb.finish()
    return 

    

if __name__=='__main__':


    os.environ["WANDB_API_KEY"] = '023bbd3f90001ca7078a6a6f750d1ea178045671' # 将引号内的+替换成自己在wandb上的一串值
    os.environ["WANDB_MODE"] = "offline"  
    # 设置参数
    model_name = "t5-base"
    # model_name = "t5-large"
    train_dir = "./input/csqa/right.train.gpt.jsonl"
    val_dir = './input/csqa/right.valid.gpt.jsonl'
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-5
    # {'RA':'./tmp/t5/ra/','WOCOT':'./tmp/t5/wocot/','R':'./tmp/t5/r/','WOCOT-base':'./tmp/t5/wocot-base/'}
    # dataLoader_key = 'R'
    dataLoader_key = 'RA'
    # dataLoader_key = 'WOCOT-base'
    print('---mode---',dataLoader_key)

    # finetune_t5_with_wandb(dataLoader_key,train_dir,val_dir , model_name, batch_size, num_epochs, learning_rate=learning_rate)
    # run_ddp(train_dir, val_dir, model_name="t5-large", batch_size=batch_size, num_epochs=num_epochs, learning_rate=1e-5)

    
    #test
    model_epochnum = 9
    test_dir = './input/csqa/test.gpt.jsonl'
    save_dir = './res/t5/'+dataLoader_key+str(model_epochnum)+'.res'
    test(dataLoader_key,model_epochnum,test_dir,batch_size,model_name)