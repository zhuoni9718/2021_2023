from sklearn.metrics import accuracy_score
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMultipleChoice, RobertaTokenizer, RobertaForMultipleChoice, AdamW
from dataloader import MultipleChoiceDataset,MultipleChoiceDatasetForPromptK,MultipleChoiceDatasetQC,MultipleChoiceDatasetCommonGen,MultipleChoiceDatasetCgk
from transformers import  AdamW,get_linear_schedule_with_warmup
import argparse
import os
import datetime
import wandb
from tqdm import tqdm
import random

MODEL_CLASSES = {
    'roberta-base':(RobertaForMultipleChoice,RobertaTokenizer),
    'roberta-large':(RobertaForMultipleChoice,RobertaTokenizer),
    'bert-base': (BertForMultipleChoice,BertTokenizer)
}

Dataset_classes = {
    # 'RA':DistlCSQADatasetRA,
    # 'WOCOT':DistlCSQADataset,
    # 'R':DistlCSQADatasetR,
    # 'WOCOT-base':DistlCSQADataset,
    'McQKC':MultipleChoiceDataset,
    'McQ_promptKc':MultipleChoiceDatasetForPromptK,
    'QC':MultipleChoiceDatasetQC,
    'CG':MultipleChoiceDatasetCommonGen,
    'CGR':MultipleChoiceDatasetCgk
}
def train(model_name,num_epochs,train_dir,val_dir,batch_size,learning_rate,model_path,dataloader_mode):
    dataset = Dataset_classes[dataloader_mode]
    print(f'dataset:{dataset}')
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = dataset(train_dir, tokenizer)
    val_dataset = dataset(val_dir, tokenizer)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 微调模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置dropout_rate
    model.config.attention_probs_dropout_prob = 0.1
    model.config.hidden_dropout_prob = 0.1
    optimizer = AdamW(model.parameters(), lr=learning_rate,weight_decay=0.01, eps=1e-6)
    num_training_steps = len(train_loader) * num_epochs
    warmup_steps = 500
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    for epoch in range(num_epochs):
        print(f"training epoch{epoch}")
        total_train_loss=0
        model.train()
        step = 0
        for batch in tqdm(train_loader, mininterval=180):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping here
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            step+=1
            cur_loss = total_train_loss/step
            if step%10==0:
                print(cur_loss)
            wandb.log({'cur_train_loss':cur_loss})
        avg_train_loss = total_train_loss / len(train_loader)
        wandb.log({"train_loss": avg_train_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_labels = []
        val_preds = []
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, mininterval=30):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # print(outputs)
                logits = outputs.logits
                loss = outputs.loss
                preds = torch.argmax(logits, dim=1)
                total_val_loss += loss.item()

                val_labels.extend(labels.tolist())
                val_preds.extend(preds.tolist())
                cur_val_loss = total_train_loss/step
                if step%300==0:
                    print(cur_val_loss)
                wandb.log({'cur_train_loss':cur_val_loss})
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        wandb.log({"val_loss": avg_val_loss,"val_acc":val_accuracy})
        print(f"Epoch: {epoch+1}, Validation Accuracy: {val_accuracy},valid loss: {avg_val_loss}")

        # 保存微调后的模型
        model.save_pretrained(f"{model_path}{model_name}_{epoch}")
        tokenizer.save_pretrained(f"{model_path}{model_name}_{epoch}")
        print(f'save to {model_path}{model_name}_{epoch}')
    return

# 记录每次测试的准确率和步数
def log_accuracy(accuracy, step):
    wandb.log({'accuracy': accuracy, 'step': step})

def train_step_test(model_name,num_epochs,train_dir,val_dir,batch_size,learning_rate,model_path,dataloader_mode):
    dataset = Dataset_classes[dataloader_mode]
    print(f'dataset:{dataset}')
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = dataset(train_dir, tokenizer)
    val_dataset = dataset(val_dir, tokenizer)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 微调模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置dropout_rate
    model.config.attention_probs_dropout_prob = 0.1
    model.config.hidden_dropout_prob = 0.1
    optimizer = AdamW(model.parameters(), lr=learning_rate,weight_decay=0.01, eps=1e-6)
    num_training_steps = len(train_loader) * num_epochs
    warmup_steps = 500
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    step_acc = {}
    for epoch in range(num_epochs):
        print(f"training epoch{epoch}")
        total_train_loss=0
        model.train()
        plt_step,plt_acc = [],[]
        step = 0
        for batch in tqdm(train_loader, mininterval=180):
            optimizer.zero_grad()
            model.train()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping here
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            step+=1
            cur_loss = total_train_loss/step
            if step%100==0:
                val_labels = []
                val_preds = []
                total_val_loss = 0
                with torch.no_grad():
                    print('testing')
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        # print(outputs)
                        logits = outputs.logits
                        loss = outputs.loss
                        preds = torch.argmax(logits, dim=1)
                        total_val_loss += loss.item()

                        val_labels.extend(labels.tolist())
                        val_preds.extend(preds.tolist())
                        cur_val_loss = total_train_loss/step
                        wandb.log({'cur_val_loss':cur_val_loss})

                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = accuracy_score(val_labels, val_preds)
                log_accuracy(val_accuracy, step)
                step_acc[str(step)]=val_accuracy
                plt_acc.append(val_accuracy)
                plt_step.append(step)
                print(f"step: {step}, Validation Accuracy: {val_accuracy},valid loss: {avg_val_loss}")
                
        avg_train_loss = total_train_loss / len(train_loader)
        wandb.log({"train_loss": avg_train_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")


        # 保存微调后的模型
        model.save_pretrained(f"{model_path}{model_name}_{epoch}")
        tokenizer.save_pretrained(f"{model_path}{model_name}_{epoch}")
        print(f'save to {model_path}{model_name}_{epoch}')
    print(f'[draw data]\n{step_acc}')
    # 绘制准确率随步数变化的折线图
    wandb.log({'accuracy_plot': wandb.plot.line_series(
        xs=[entry for entry in plt_step],
        ys=[entry for entry in plt_acc],
        keys=['accuracy'],
        x_axis='步数',
        y_axis='准确率%',
        title="推理依据增强实验准确率随步数变化图"
    )})
    return

def test(test_dir,model_name,batch_size,best_epoch,model_path='',dataloader_mode='McQKC'):
    dataset = Dataset_classes[dataloader_mode]
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    model_path = f"{model_path}{model_name}_{str(best_epoch)}"
    # model_path=model_path+model_name+str(best_epoch)
    if model_path!='':
        print(f'testing with {model_path}')
        model = model_class.from_pretrained(model_path)
    else:
        print(f'testing with {model_name} without ft')
        model = model_class.from_pretrained(model_name)

    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    test_dataset = dataset(test_dir, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_labels,val_preds=[],[]
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        test_labels.extend(labels.tolist())
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.tolist())

    test_accuracy = accuracy_score(test_labels, val_preds)
    print(f"test Accuracy: {test_accuracy}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and test various models with custom data.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to be fine-tuned and tested.")
    parser.add_argument("--dataset_class", type=str, required=True, help="The dataset class to be used for loading data.")
    parser.add_argument("--model_path", type=str,  help="The path to model")
    parser.add_argument("--train_data", type=str, required=True, help="The path to the training data file.")
    parser.add_argument("--dev_data", type=str, required=True, help="The path to the validation data file.")
    parser.add_argument("--test_data", type=str, help="The path to the test data file.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--test", action="store_true", help="Whether to test the model after fine-tuning.")
    parser.add_argument("--train", action="store_true", help="Whether to test the model after fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=4, help="batchsize")
    parser.add_argument("--seed", type=int, default=42, help="batchsize")
    parser.add_argument("--best_epoch", type=int, default=4, help="best_epoch")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="lr")
    parser.add_argument("--dataloader", type=str, default='McQKC', help="Dataloader")

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

    wandbname = datetime.datetime.now().strftime(r'%Y%m%d%H%M%S')
    model_name_context = args.model_name.replace('-','').replace('/','')
    wandb.init(project=f"{model_name_context}",name = wandbname)

    if args.train:
        print(f'training {args.model_name}')
        train_step_test(args.model_name,args.epochs,args.train_data,args.dev_data,args.batch_size,args.learning_rate,args.model_path,args.dataloader)
        # train(args.model_name,args.epochs,args.train_data,args.dev_data,args.batch_size,args.learning_rate,args.model_path,args.dataloader)
    if args.test:
        print(f"testing {args.model_name}")
        test(args.test_data,args.model_name,args.batch_size,args.best_epoch,args.model_path,args.dataloader)


if __name__=='__main__':
    os.environ["WANDB_API_KEY"] = '023bbd3f90001ca7078a6a6f750d1ea178045671'
    os.environ["WANDB_MODE"] = "offline"  
    main()