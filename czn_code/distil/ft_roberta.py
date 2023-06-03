import torch
from torch.nn import DataParallel
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset,DataLoader

class RoBERTaDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = []
        with jsonlines.open(data_file) as f:
            for item in f:
                stem = item["data"]["question"]["stem"]
                choices = [choice["text"] for choice in item["data"]["question"]["choices"]]
                knowledge = item["k"].split('\n')[1].strip()
                correct_index = ord(item["data"]["answerKey"]) - ord("A")

                input_texts = [f"{stem} | {choice} | {knowledge}" for choice in choices]
                self.data.append({"input_texts": input_texts, "correct_index": correct_index})

        self.tokenizer = tokenizer
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_texts = item["input_texts"]
        input_ids = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).input_ids
        correct_index = item["correct_index"]

        return {"input_ids": input_ids, "correct_index": correct_index}

def finetune_roberta(model_name, train_data_file, dev_data_file, dataset_class, epochs):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)

    train_dataset = RoBERTaDataset(train_data_file, tokenizer)
    dev_dataset = RoBERTaDataset(dev_data_file, tokenizer)
    
    # 使用Transformers库中的Trainer和TrainingArguments进行训练
    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=1000,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    trainer.train()

# 使用以下代码进行微调
model_name = 'roberta-base'  # 更改为想要微调的 RoBERTa 模型名称
train_data_file = "path/to/your/train_data.jsonl"  # 更改为训练数据文件的路径
dev_data_file = "path/to/your/dev_data.jsonl"  # 更改为验证数据文件的路径
dataset_class = RoBERTaDataset  # 使用 RoBERTa 数据集类
epochs = 3  # 更改为您想要的训练周期数

finetune_roberta(model_name, train_data_file, dev_data_file, dataset_class, epochs)
