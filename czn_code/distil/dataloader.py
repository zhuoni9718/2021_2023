
from torch.utils.data import DataLoader, Dataset
import jsonlines
from tqdm import tqdm
import torch

# 自定义数据集类
# {'RA':DistlCSQADatasetRA,'WOCOT':DistlCSQADataset,'R':DistlCSQADatasetR}
# RA 是用Q  CHOICES R生成答案 打分器 
# WOCOT 是 Q C 生成答案
# R 是生成rationale 
class cg_genDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = []
        with jsonlines.open(data_file) as f:
            for item in f:
                # {"concept_set": "eat#hay#horse", "scene": ["A horse is eating hay.", "The horses are eating hay.", "A horse eats hay in the barn"]}
                input_text = ' '.join(item["concept_set"].split('#'))
                
                target_text = item["scene"][0]

                # exit()
                self.data.append({"input_text": input_text, "target_text": target_text})
        self.tokenizer = tokenizer
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # print(item["input_text"])
        if idx<3:
            print('[input]',item["input_text"])
            print('[target]',item["target_text"])
        input_encoded = self.tokenizer(item["input_text"],  padding="max_length" if self.max_length else "longest",return_tensors="pt")
        labels = self.tokenizer(item["target_text"], padding="max_length" if self.max_length else "longest", return_tensors="pt").input_ids.squeeze()
        input_ids = input_encoded.input_ids.squeeze()
        attention_mask_tensor = input_encoded.attention_mask.squeeze()
        
        # print(inputs)
        # print(f"Processed item: {idx}, input_ids: {input_ids}, labels: {labels}")  # 添加调试输出
        return {"input_ids": input_ids, 'attention_mask':attention_mask_tensor,"labels": labels}

class DistlCSQADataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = []
        with jsonlines.open(data_file) as f:
            for item in f:
                input_text = item["data"]["question"]["stem"]
                # options = ', '.join([item['data']['question']['choices'][i]['text'] for i in range(5)])
                options = ''
                for i in range(5):
                    options += ['A','B','C','D','E'][i]+') '+item['data']['question']['choices'][i]['text']+', '
                input_text = 'Question: '+input_text+' <s> Options: '+options+' <s>'
                # target_text = item["k"].strip()
                target_text = item["k"].split('\n')[0]

                # exit()
                self.data.append({"input_text": input_text, "target_text": target_text})
        self.tokenizer = tokenizer
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # print(item["input_text"])
        # if idx<3:
            # print('[input]',item["input_text"])
            # print('[target]',item["target_text"])
        input_encoded = self.tokenizer(item["input_text"],  padding="max_length" if self.max_length else "longest",return_tensors="pt")
        labels = self.tokenizer(item["target_text"], padding="max_length" if self.max_length else "longest", return_tensors="pt").input_ids.squeeze()
        input_ids = input_encoded.input_ids.squeeze()
        attention_mask_tensor = input_encoded.attention_mask.squeeze()
        
        # print(inputs)
        # print(f"Processed item: {idx}, input_ids: {input_ids}, labels: {labels}")  # 添加调试输出
        return {"input_ids": input_ids, 'attention_mask':attention_mask_tensor,"labels": labels}


class DistlCSQADatasetR(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = []
        with jsonlines.open(data_file) as f:
            for item in f:
                input_text = item["data"]["question"]["stem"]
                # options = ', '.join([item['data']['question']['choices'][i]['text'] for i in range(5)])
                # 如果没有推理路径就不要这条数据
                rationale = ''
                try:
                    cur_rationale = item['k'].split('\n')[1].strip('Knowledge:').strip()
                    tmps = cur_rationale.split('. ')
                    # print(tmps)
                    # 去除the other options的句子
                    for tmp in tmps:
                        # print(tmp)
                        if 'The other options' in tmp:
                            continue
                        else:
                            rationale+=tmp
                except:
                    rationale = ''
                options = ''
                # print(f'---{rationale}')
                for i in range(5):
                    options += ['A','B','C','D','E'][i]+') '+item['data']['question']['choices'][i]['text']+', '
                # input_text = 'Generate rationale for the question and choices. <s> Question: '+input_text+' Options: '+options+'\nRationale: '
                input_text = 'Question: '+input_text+' Options: '+options
                # target_text = item["k"].strip()
                target_text = rationale
                # print('[INPUT]',input_text)
                # print('[out]:',target_text)
                # exit()
                self.data.append({"input_text": input_text, "target_text": target_text})
        self.tokenizer = tokenizer
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # print(item["input_text"])
        input_encoded = self.tokenizer(item["input_text"],  padding="max_length" if self.max_length else "longest", return_tensors="pt")
        input_ids = input_encoded.input_ids.squeeze()
        attention_mask_tensor = input_encoded.attention_mask.squeeze()
        labels = self.tokenizer(item["target_text"], padding="max_length" if self.max_length else "longest",  return_tensors="pt").input_ids.squeeze()
        print(f'[length]{input_ids.shape,attention_mask_tensor.shape,labels.shape}')

        return {"input_ids": input_ids, "attention_mask": attention_mask_tensor,"labels": labels}


class DistlCSQADatasetRA(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = []
        with jsonlines.open(data_file) as f:
            for item in f:
                input_text = item["data"]["question"]["stem"]
                # options = ', '.join([item['data']['question']['choices'][i]['text'] for i in range(5)])
                options = ''
                try:
                    rationale = item['k'].split('\n')[1]
                    target_text = item["k"].split('\n')[0]
                except:
                    rationale = ''
                    target_text = ''
                for i in range(5):
                    options += ['A','B','C','D','E'][i]+') '+item['data']['question']['choices'][i]['text']+', '
                input_text = 'Question: '+input_text+' <s> '+rationale+' <s> Options: '+options+' Answer is'
                # target_text = item["k"].strip()
                
                # print('[input]',input_text)
                # print('[target]',target_text)
                self.data.append({"input_text": input_text, "target_text": target_text})
        
        
        self.tokenizer = tokenizer
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # if idx<3:
        #     print('[input]',item["input_text"])
        #     print('[target]',item["target_text"])
        # print(item["input_text"])
        input_all = self.tokenizer(item["input_text"],  padding="max_length" if self.max_length else "longest",return_tensors="pt")
        input_ids = input_all.input_ids.squeeze()
        labels = self.tokenizer(item["target_text"], padding="max_length" if self.max_length else "longest", return_tensors="pt").input_ids.squeeze()
        attention_mask_tensor = input_all.attention_mask.squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask_tensor,"labels": labels}


class MultipleChoiceDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data = []
        with jsonlines.open(data_dir,'r') as f:
            for line in f:
                self.data.append(line)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_data = self.data[idx]
        question = question_data['data']["question"]["stem"]
        choices = question_data['data']["question"]["choices"]
        rationale = ''
        try:
            cur_rationale = item['k'].split('\n')[1].strip('Knowledge:').strip()
            tmps = cur_rationale.split('. ')
            # print(tmps)
            # 去除the other options的句子
            for tmp in tmps:
                # print(tmp)
                if 'The other options' in tmp:
                    continue
                else:
                    rationale+=tmp
        except:
            rationale = ''
        # 对问题和选项进行编码
        input_ids = []
        attention_masks = []
        for choice in choices:
            text = '<s> Q: '+question+' A: '+choice["text"]+' <s> '+rationale+' </s>'
            # text = '[CLS] Q: '+question +' [SEP] '+rationale+ ' [SEP] A: ' + choice["text"] 
            # print(text)
            encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids.append(encoding["input_ids"])
            attention_masks.append(encoding["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label = ord(question_data["data"]["answerKey"]) - ord("A")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "label": torch.tensor(label, dtype=torch.long)
        }


class MultipleChoiceDatasetQC(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data = []
        with jsonlines.open(data_dir,'r') as f:
            for line in f:
                self.data.append(line)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_data = self.data[idx]
        question = question_data['data']["question"]["stem"]
        choices = question_data['data']["question"]["choices"]
        rationale = ''
        try:
            cur_rationale = item['k'].split('\n')[1].strip('Knowledge:').strip()
            tmps = cur_rationale.split('. ')
            # print(tmps)
            # 去除the other options的句子
            for tmp in tmps:
                # print(tmp)
                if 'The other options' in tmp:
                    continue
                else:
                    rationale+=tmp
        except:
            rationale = ''
        # 对问题和选项进行编码
        input_ids = []
        attention_masks = []
        for choice in choices:
            text = '<s> Q: '+question+' A: '+choice["text"]+' <s> '
            # text = '[CLS] Q: '+question +' [SEP] '+rationale+ ' [SEP] A: ' + choice["text"] 
            # print(text)
            encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids.append(encoding["input_ids"])
            attention_masks.append(encoding["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label = ord(question_data["data"]["answerKey"]) - ord("A")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "label": torch.tensor(label, dtype=torch.long)
        }

class MultipleChoiceDatasetForPromptK(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data = []
        with jsonlines.open(data_dir,'r') as f:
            for line in f:
                self.data.append(line)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    # {"data:": {"answerKey": "C", "id": "1d24f406b6828492040b405d3f35119c", "question": {"question_concept": "playing guitar", "choices": [{"label": "A", "text": "cry"}, {"label": "B", "text": "hear sounds"}, {"label": "C", "text": "singing"}, {"label": "D", "text": "arthritis"}, {"label": "E", "text": "making music"}], "stem": "What do people typically do while playing guitar?"}, "statements": [{"label": false, "statement": "Cry do people typically do while playing guitar."}, {"label": false, "statement": "Hear sounds do people typically do while playing guitar."}, {"label": true, "statement": "Singing do people typically do while playing guitar."}, {"label": false, "statement": "Arthritis do people typically do while playing guitar."}, {"label": false, "statement": "Making music do people typically do while playing guitar."}]}, "knowledge: ": ["Guitarists play to relax or get inspired by their music; they don't necessarily want a song that will make them happy all day long (or even every night", "Guitarists play to their own personal music style (i.e., they don't always follow a standard chord progression). They also often improvise on stage or", ]}
        #
        question_data = self.data[idx]
        question = question_data['data:']["question"]["stem"]
        choices = question_data['data:']["question"]["choices"]
        try:
            if 'knowledge: ' in question_data:
                rationale = question_data['knowledge: '][0]
            elif 'knowledge:' in question_data:
                rationale = question_data['knowledge:'][0]
        except:
            # print(e)
            rationale = ''
            print('data without k')
        # 对问题和选项进行编码
        input_ids = []
        attention_masks = []
        for choice in choices:
            # text = '[CLS] Q: '+question +' [SEP] '+rationale+ ' [SEP] A: ' + choice["text"] 
            # text = '[CLS] Q: '+question + ' [SEP] A: ' + choice["text"] +' [SEP] '+rationale
            text = '<s> Q: '+question + ' </s> A: ' + choice["text"] +' </s> '+rationale
            # print(text)
            encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids.append(encoding["input_ids"])
            attention_masks.append(encoding["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label = ord(question_data["data:"]["answerKey"]) - ord("A")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "label": torch.tensor(label, dtype=torch.long)
        }

class MultipleChoiceDatasetCommonGen(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data = []
        with jsonlines.open(data_dir,'r') as f:
            for line in f:
                self.data.append(line)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    # {"data:": {"answerKey": "C", "id": "1d24f406b6828492040b405d3f35119c", "question": {"question_concept": "playing guitar", "choices": [{"label": "A", "text": "cry"}, {"label": "B", "text": "hear sounds"}, {"label": "C", "text": "singing"}, {"label": "D", "text": "arthritis"}, {"label": "E", "text": "making music"}], "stem": "What do people typically do while playing guitar?"}, "statements": [{"label": false, "statement": "Cry do people typically do while playing guitar."}, {"label": false, "statement": "Hear sounds do people typically do while playing guitar."}, {"label": true, "statement": "Singing do people typically do while playing guitar."}, {"label": false, "statement": "Arthritis do people typically do while playing guitar."}, {"label": false, "statement": "Making music do people typically do while playing guitar."}]}, "knowledge: ": ["Guitarists play to relax or get inspired by their music; they don't necessarily want a song that will make them happy all day long (or even every night", "Guitarists play to their own personal music style (i.e., they don't always follow a standard chord progression). They also often improvise on stage or", ]}
        #
        question_data = self.data[idx]
        question = question_data['data:']["question"]["stem"]
        choices = question_data['data:']["question"]["choices"]
        rationale = question_data['knowledge:']
        # 对问题和选项进行编码
        input_ids = []
        attention_masks = []
        for j in range(5):
            choice = choices[j]
            # text = '[CLS] Q: '+question +' [SEP] '+rationale+ ' [SEP] A: ' + choice["text"] 
            text = '<s> Q: '+question + ' <s> A: ' + choice["text"] +' <s> '+rationale[j]+' </s>'

            print(text)
            encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids.append(encoding["input_ids"])
            attention_masks.append(encoding["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label = ord(question_data["data:"]["answerKey"]) - ord("A")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "label": torch.tensor(label, dtype=torch.long)
        }

class MultipleChoiceDatasetCgk(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data = []
        with jsonlines.open(data_dir,'r') as f:
            for line in f:
                self.data.append(line)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    # {"data:": {"answerKey": "C", "id": "1d24f406b6828492040b405d3f35119c", "question": {"question_concept": "playing guitar", "choices": [{"label": "A", "text": "cry"}, {"label": "B", "text": "hear sounds"}, {"label": "C", "text": "singing"}, {"label": "D", "text": "arthritis"}, {"label": "E", "text": "making music"}], "stem": "What do people typically do while playing guitar?"}, "statements": [{"label": false, "statement": "Cry do people typically do while playing guitar."}, {"label": false, "statement": "Hear sounds do people typically do while playing guitar."}, {"label": true, "statement": "Singing do people typically do while playing guitar."}, {"label": false, "statement": "Arthritis do people typically do while playing guitar."}, {"label": false, "statement": "Making music do people typically do while playing guitar."}]}, "knowledge: ": ["Guitarists play to relax or get inspired by their music; they don't necessarily want a song that will make them happy all day long (or even every night", "Guitarists play to their own personal music style (i.e., they don't always follow a standard chord progression). They also often improvise on stage or", ]}
        #
        question_data = self.data[idx]
        question = question_data['data']["question"]["stem"]
        choices = question_data['data']["question"]["choices"]
        rationale = question_data['k']
        cgk = question_data['cgk']
        # 对问题和选项进行编码
        input_ids = []
        attention_masks = []
        for j in range(5):
            choice = choices[j]
            # text = '[CLS] Q: '+question +' [SEP] '+rationale+ ' [SEP] A: ' + choice["text"] 
            text = '<s> Q: '+question + ' <s> A: ' + choice["text"] +' <s> '+cgk[j]+' <s> '+rationale+' </s>'

            # print(text)
            encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids.append(encoding["input_ids"])
            attention_masks.append(encoding["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label = ord(question_data["data"]["answerKey"]) - ord("A")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "label": torch.tensor(label, dtype=torch.long)
        }


# class DistlCSQADatasetRA(Dataset):
#     def __init__(self, data_file, tokenizer):
#         self.data = []
#         with jsonlines.open(data_file) as f:
#             for item in f:
#                 input_text = item["data"]["question"]["stem"]
#                 # options = ', '.join([item['data']['question']['choices'][i]['text'] for i in range(5)])
#                 options = ''
#                 try:
#                     rationale = item['k'].split('\n')[1]
#                     target_text = item["k"].split('\n')[0]
#                 except:
#                     rationale = ''
#                     target_text = ''
#                 for i in range(5):
#                     options += ['A','B','C','D','E'][i]+': '+item['data']['question']['choices'][i]['text']+','
#                 input_text = 'Question: '+input_text+' Options: '+options+' '+rationale+' Answer is'
#                 # target_text = item["k"].strip()
                
#                 print('--input--',input_text)
#                 print('--target--',target_text)
#                 self.data.append({"input_text": input_text, "target_text": target_text})
        
        
#         self.tokenizer = tokenizer
#         self.max_length = 128

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         # print(item["input_text"])
#         input_ids = self.tokenizer(item["input_text"],  padding="max_length" if self.max_length else "longest",return_tensors="pt").input_ids.squeeze()
#         labels = self.tokenizer(item["target_text"], padding="max_length" if self.max_length else "longest", return_tensors="pt").input_ids.squeeze()
#         return {"input_ids": input_ids, "labels": labels}