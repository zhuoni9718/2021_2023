from transformers import AutoModelWithLMHead, AutoTokenizer
import jsonlines
from tqdm import tqdm
from lm_text_generator import LMTextGenerator
import torch
device = 0
device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")

def init_model(model_name: str,device: torch.device):
    print('model name: ',model_name)
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    # logger.info(f'Initializing {model_name}')
  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(device)
    model.eval
    # if tokenizer.pad_token is None:
    #     if tokenizer.eos_token is not None:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         tokenizer.pad_token = tokenizer.eos_token = tokenizer.unk_token

    return model, tokenizer
def gen_sentence(words, max_length=32):
    input_text = words
    features = tokenizer([input_text], return_tensors='pt')
    num_samples = 1
    sentences = []
    for i in range(num_samples):
        output = model.generate(input_ids=features['input_ids'].to(device),
                attention_mask=features['attention_mask'].to(device),
                max_length=max_length)
        sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        sentences.append(sentence)
    return sentences



if __name__=='__main__':
    model,tokenizer = init_model('mrm8488/t5-base-finetuned-common_gen', device)
    words = []
    res = []
    sample_num = 500
    # sentence_num = 5 #生成5条知识
    num = 0
    # data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/qac/csqa.train.qac.src'
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/qac/csqa.dev.qac.src'
    with open(data_dir,'r') as f:
        for line in f:
            words.append(line.strip())

    print('begin generating')
    # generator = LMTextGenerator('mrm8488/t5-base-finetuned-common_gen',device)
    for item in tqdm(words):
        # if num>=sample_num:
        #     break
        # num += 1
        sentences = []
        print('item: ',item)
        sentence = gen_sentence(item)
        # sentence = generator.generate(item, p= 0.2, k= 0.0, temperature= 1.0,length = 25, num_samples = 10, stop_token='\n')
        sentences.append(sentence)
        res.append([{'concepts':item,'sentence':sentences}])
        print(res)
        exit()
    # write_dir = './outputfile/obqa_train_qac_sentence_all.json'
    write_dir = './outputfile/obqa_dev_qac_sentence_all.json'
    print('begin writing')
    with jsonlines.open(write_dir,'w') as f:
        for item in tqdm(res):
            f.write(item)
        f.close()
    
