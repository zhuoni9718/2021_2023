import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import jsonlines
from torch.nn import DataParallel
import wandb
from tqdm import tqdm
import datetime
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import numpy as np
from transformers import  AdamW,get_linear_schedule_with_warmup
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    T5ForConditionalGeneration, AutoTokenizer,
    BartForConditionalGeneration, BartTokenizer,
    Trainer, TrainingArguments
)
from transformers import  AdamW
from res.getacc import get_a_acc
from dataloader import DistlCSQADatasetRA,DistlCSQADataset,DistlCSQADatasetR,cg_genDataset

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-medium': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-large': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-xl': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    't5-large': (T5ForConditionalGeneration, AutoTokenizer),
    'facebook/bart-large': (BartForConditionalGeneration, BartTokenizer),
}
Dataloader_classes = {
    'RA':DistlCSQADatasetRA,
    'WOCOT':DistlCSQADataset,
    'R':DistlCSQADatasetR,
    'WOCOT-base':DistlCSQADataset,
    'CG':cg_genDataset
}




def finetune_frame(dataLoader_key,train_dir,val_dir, model_name="t5-small", batch_size=8, num_epochs=3, learning_rate=5e-5,load_model=''):
    # 初始化wandb
    wandbname = datetime.datetime.now().strftime(r'%Y%m%d%H%M%S')
    wandb.init(project=f"{model_name.replace('/', '').replace('-', '_')}",name = wandbname)

    # 加载预训练的模型和分词器
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    if load_model!='':
        print(f'ft {multilearning_model_path}')
        model = model_class.from_pretrained(multilearning_model_path)
    else:
        model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name)
    

    # 准备输入数据
    DataLoader_dict = {'RA':DistlCSQADatasetRA,'WOCOT':DistlCSQADataset,'R':DistlCSQADatasetR,'WOCOT-base':DistlCSQADataset,'CG':cg_genDataset}
    dataloader_set = DataLoader_dict[dataLoader_key]
    train_dataset = dataloader_set(train_dir, tokenizer)
    val_dataset = dataloader_set(val_dir, tokenizer)
    num_workers = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers)
    # exit()
    # 定义优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    # optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # 设置dropout_rate
    model.config.attention_probs_dropout_prob = 0.1
    model.config.hidden_dropout_prob = 0.1
    optimizer = AdamW(model.parameters(), lr=learning_rate,weight_decay=0.01, eps=1e-6)
    num_training_steps = len(train_dataloader) * num_epochs
    warmup_steps = 500
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    # 训练和验证模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    accumulation_steps = 5

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        step = 0
        for batch in tqdm(train_dataloader, mininterval=30):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            # loss = outputs.loss
            loss = torch.mean(outputs.loss)
            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            # scheduler.step()
            total_train_loss += loss.item()
            step+=1

            if step % accumulation_steps == 0:  # 每累积8个小批量，执行一次梯度更新
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  # 清空梯度

            if step%100==0:
                current_loss = total_train_loss / step
                print(f"Step: {step}, Loss: {current_loss:.4f}")
                wandb.log({'step':step,"train_loss_step": current_loss})
        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({"train_loss": avg_train_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0
        res_contexts =[]
        if dataLoader_key=='WOCOT' or dataLoader_key=='RA':
            print(f'---val test in answer status---')
            for batch in tqdm(val_dataloader, mininterval=60):
                batch = {k: v.to(device) for k, v in batch.items()}
                # outputs = model(**batch)
                input_ids = batch["input_ids"]
                attention_id = batch['attention_mask']
                # print('[inputid]',input_ids)
                out_context,input_contexts = [],[]
                eos_token_id = tokenizer.eos_token_id
                for id in input_ids:
                    input_context = tokenizer.decode(id, skip_special_tokens=True)
                    input_contexts.append(input_context)
                output = model.generate(input_ids,attention_mask = attention_id, min_length=1, num_return_sequences=1, early_stopping=True,eos_token_id=eos_token_id)
                # 对模型生成的输出进行解码
                # print(output.size())
                # print('[output]',output)
                for i in range(len(output)):
                    output_id = output[i]
                    answer = tokenizer.decode(output_id, skip_special_tokens=True)
                    res_contexts.append(answer)
            val_acc = get_a_acc(val_dir,res_contexts)
            print(f'val acc:{val_acc}')
        else:
            print('---val test in loss status---')
            for batch in tqdm(val_dataloader, mininterval=30):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                    # loss = outputs.loss
                    loss = torch.mean(outputs.loss)
                total_val_loss += loss.item()
                if step%100==0:
                    current_val_loss = total_val_loss / step
                    print(f"Step: {step}, val_Loss: {current_val_loss:.4f}")
                    wandb.log({'step':step,"val_loss_step": current_val_loss})
            avg_val_loss = total_val_loss / len(val_dataloader)
            wandb.log({"val_loss": avg_val_loss})
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        # 保存微调后的模型ft    
        model_name_context = model_name.replace('-','')
        model_name_context = model_name_context.replace('/','')
        model_path = f'./tmp/generate_model/{model_name_context}/{dataLoader_key}/'
        model_path = model_path+f'{model_name_context}_'+str(epoch)
        print('saving to ',model_path)
        if isinstance(model, nn.DataParallel):
            model = model.module
            model.save_pretrained(model_path)
        else:
            model.save_pretrained(model_path)
    wandb.finish()
    return 

def test(dataLoader_key,model_epochnum,test_dir,batch_size,model_name):

    # 加载预训练的T5模型
    model_name_context = model_name.replace('-','')
    model_name_context = model_name_context.replace('/','')
    model_path = f'./tmp/generate_model/{model_name_context}/{dataLoader_key}/'
    model_path = model_path+f"{model_name_context}_"+str(model_epochnum)
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    # init_flag=True
    init_flag=False
    if os.path.isdir(model_path) and init_flag==False:
        print(f'using{model_path},data:{dataLoader_key}')
        model = model_class.from_pretrained(model_path)
    else:
        print(f'using unit {model_name}')
        model = model_class.from_pretrained(model_name)
    
    # 加载相应的T5分词器
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #准备数据
    DataLoader_dict = {'RA':DistlCSQADatasetRA,'WOCOT':DistlCSQADataset,'R':DistlCSQADatasetR}
    dataloader_set = DataLoader_dict[dataLoader_key]
    test_dataset = dataloader_set(test_dir, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 测试
    model.eval()
    answers = []
    for batch in tqdm(test_dataloader, mininterval=60):
        # batch = {k: v.to(device) for k, v in batch.items()}
        # outputs = model(**batch)
        input_ids = batch["input_ids"]
        attention_id = batch['attention_mask']
        # print('[inputid]',input_ids)
        out_context,input_contexts = [],[]
        eos_token_id = tokenizer.eos_token_id
        for id in input_ids:
            input_context = tokenizer.decode(id, skip_special_tokens=True)
            input_contexts.append(input_context)
        with torch.no_grad():
            output = model.generate(input_ids,attention_mask = attention_id, min_length=1, num_return_sequences=1, early_stopping=True,eos_token_id=eos_token_id)
        # 对模型生成的输出进行解码
        # print(output.size())
        # print('[output]',output)
        
        for i in range(len(output)):
            output_id = output[i]
            answer = tokenizer.decode(output_id, skip_special_tokens=True)
            # out_context.append(answer)
            answers.append(answer)
            print(f'第{i}个\n[input]{input_contexts[i]}\n[output]{answer}')
        # exit()
    os.makedirs(f'./res/{model_name_context}',exist_ok=True)
    if init_flag==True:
        res_save_path = f'./res/{model_name_context}/{dataLoader_key}'+str(model_epochnum)+'init.res'
    else:
        res_save_path = f'./res/{model_name_context}/{dataLoader_key}'+str(model_epochnum)+'res'

    with jsonlines.open(res_save_path,'w') as f:
        print("writing to ",res_save_path)
        for item in answers:
            f.write(item)
    return
    


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and test various models with custom data.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to be fine-tuned and tested.")
    parser.add_argument("--dataset_class", type=str, required=True, help="The dataset class to be used for loading data.")
    parser.add_argument("--train_data", type=str, required=True, help="The path to the training data file.")
    parser.add_argument("--dev_data", type=str, required=True, help="The path to the validation data file.")
    parser.add_argument("--test_data", type=str, help="The path to the test data file.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--test", action="store_true", help="Whether to test the model after fine-tuning.")
    parser.add_argument("--train", action="store_true", help="Whether to test the model after fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=4, help="batchsize")
    parser.add_argument("--best_epoch", type=int, default=4, help="batchsize")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="lr")
    parser.add_argument("--load_model", type=str, default='', help="modelpath")

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # 随机数种子
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.model_name not in MODEL_CLASSES:
        raise ValueError("Invalid model name. Please choose one from the available options.")
    
    if args.dataset_class not in Dataloader_classes:
        raise ValueError("Invalid dataset class. Please choose one from the available options.")
    model_name_context = args.model_name.replace('-','')
    model_name_context = model_name_context.replace('/','')
    wandbname = datetime.datetime.now().strftime(r'%Y%m%d%H%M%S')
    wandb.init(project=f"{model_name_context}",name = wandbname)

    if args.train:
        print(f"Fine-tuning {args.model_name}...")
        finetune_frame(args.dataset_class,args.train_data, args.dev_data, model_name=args.model_name, batch_size=args.batch_size, num_epochs=args.epochs, learning_rate=args.learning_rate,load_model = args.load_model)
    if args.test:
        if args.test_data is None:
            raise ValueError("Please provide a test data file path when using --test option.")
        print(f"Testing {args.model_name}...")
        tokenizer_class = MODEL_CLASSES[args.model_name][1]
        # test_model(args.model_name, tokenizer_class, args.test_data, Dataloader_classes[args.dataset_class])
        test(args.dataset_class,args.best_epoch,args.test_data,args.batch_size,args.model_name)

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = '023bbd3f90001ca7078a6a6f750d1ea178045671'
    os.environ["WANDB_MODE"] = "offline"  

    main()

