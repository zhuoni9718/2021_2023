import os

from numpy import single
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, TextGenerationPipeline, GPT2Tokenizer
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                      "mask_token": "[MASK]",
                      "bos_token": "[BOS]", "eos_token": "[EOS]"}
def load_tokenizer(vocab_file, special_token_path=None):
    '''
    加载tokenizer
    :param tokenizer_path:
    :param special_token_path:
    :return:
    '''
    # print('tokenizer loadding...')
    # vocab_file = '/users5/znchen/Question2Knowledge/model/bert/vocab.txt'
    # tokenizer = BertTokenizer(vocab_file)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    if special_token_path:
        tokenizer.add_special_tokens(special_token_path)
    return tokenizer


def load_pretrained_model(tokenizer, config_path, load_model_path, SPECIAL_TOKENS=None):
    '''
    加载 pretrained model
    :param tokenizer:
    :param load_model_path:
    :param SPECIAL_TOKENS:
    :return:
    '''
    print("pretrained model loadding from: ",load_model_path)

    # model_path = model_path_dir + 'pytorch_model.bin'
    # vocab_path = model_path_dir + 'vocab.txt'
    # config_path = model_path_dir + 'config.json'

    config_ = GPT2Config.from_json_file(config_path)
    config_.bos_token_id=tokenizer.bos_token
    config_.eos_token_id=tokenizer.eos_token
    config_.sep_token_id=tokenizer.sep_token
    config_.unk_token_id=tokenizer.unk_token
    config_.pad_token_id=tokenizer.pad_token
    # config_.output_hidden_states=False
    model = GPT2LMHeadModel.from_pretrained(load_model_path, config=config_)

    if SPECIAL_TOKENS:
        # 添加special token,model embedding size需要作调整
        model.resize_token_embeddings(len(tokenizer))

    '''
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
    '''

    return model

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def filter_token_logits(next_token_logits, tokenizer, filtered_tokens):
    for t in filtered_tokens:
        next_token_logits[tokenizer.convert_tokens_to_ids(t)] = -float("Inf")
    return next_token_logits

def add_token(token, tensor, tokenizer):
    ret = torch.tensor(tokenizer.encode(token))
    ret = ret[1:2]
    ret = ret.unsqueeze(0)
    ret = torch.cat((tensor, ret), dim=1)
    ret = ret.to(DEVICE)
    return ret

class ProseGenerator():
    def __init__(self, logger):
        self.logger = logger
        self.SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                      "mask_token": "[MASK]",
                      "bos_token": "[BOS]", "eos_token": "[EOS]"}
        # model_path_dir = '/users5/znchen/Question2Knowledge/model/trained_model/'
        model_path_dir = '/users5/znchen/Question2Knowledge/model/GPT2/'
        model_path = model_path_dir + 'pytorch_model.bin'
        vocab_path = model_path_dir + 'vocab.txt'
        config_path = model_path_dir + 'config.json'

        self.tokenizer = load_tokenizer(vocab_file=vocab_path, special_token_path=self.SPECIAL_TOKENS)
        self.model = load_pretrained_model(self.tokenizer, config_path, model_path, self.SPECIAL_TOKENS)
        self.model.to(DEVICE)
        self.model.eval()

    def model_caption_generate(self, caption, min_sentence_length, max_sentence_length):
        repitition_penalty = 2.0
        top_k = 8
        top_p = 0
        temperature = 1
        tokenizer = self.tokenizer
        model = self.model
        SPECIAL_TOKENS = self.SPECIAL_TOKENS

        input_text = caption 
        input_text_tensor = torch.tensor(tokenizer.encode(input_text))
        input_text_tensor = input_text_tensor[:-1]
        input_text_tensor = input_text_tensor.unsqueeze(0)
        input_text_device_tensor = input_text_tensor.to(DEVICE)
        generated = input_text_device_tensor
        filtered_tokens = ['[CLS]', '[SEP]', '[UNK]', '[PAD]']
        l = len(caption)

        while True:
            inputs = {"input_ids": generated}
            # print(tokenizer.decode(inputs['input_ids'].tolist()[0]))
            outputs = model(
                    **inputs
                )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]

            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature

            # 筛掉不能出现的token和标点们
            next_token_logits = filter_token_logits(next_token_logits, tokenizer, filtered_tokens)
            
            #防止一直重复一个词
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            # print("next_token_logits: ",next_token_logits)
            next_token = torch.multinomial(
                # F.softmax(filtered_logits, dim=-1), num_samples=1
                F.softmax(next_token_logits, dim=-1), num_samples=1
            )
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            print("generated[0][-1] : ",generated[0][-1])
            l += 1
            #TODO 
            if l > min_sentence_length and l < max_sentence_length and tokenizer.decode(generated[0][-1]) == '[EOS]':
                break
            if l > max_sentence_length:
                break
        print("next_token_logits: ", next_token_logits.tolist())
        
        print(tokenizer.decode(generated.tolist()[0]))
    
    def test(self, caption):
        input_text = caption + '，'
        input_text_tensor = torch.tensor(self.tokenizer.encode(input_text))
        input_text_tensor = input_text_tensor[:-1]
        input_text_tensor = input_text_tensor.unsqueeze(0)
        print(input_text_tensor)



def model_generate(tokenizer, model, sentence_start, min_sentence_length, max_sentence_length, SPECIAL_TOKENS):
    '''
    Finetune model is used to generate text!
    :param model_path:
    :param SPECIAL_TOKENS
    :param MAX_LEN
    :return:
    '''
    repitition_penalty = 2.0
    top_k = 10
    top_p = 0.9
    temperature = 1
    print('SPECIAL_TOKENS',SPECIAL_TOKENS)

    input_text = sentence_start 
    input_text_tensor = torch.tensor(tokenizer.encode(input_text))
    input_text_tensor = input_text_tensor[:-1]
    input_text_tensor = input_text_tensor.unsqueeze(0)
    input_text_device_tensor = input_text_tensor.to(DEVICE)
    generated = input_text_device_tensor

    filtered_tokens = ['[CLS]', '[SEP]', '[UNK]', '[PAD]']
    # '？','；','：','“','”','、','!'

    l = len(sentence_start)

    while True:
        inputs = {"input_ids": generated}
        # print(tokenizer.decode(inputs['input_ids'].tolist()[0]))
        outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
        next_token_logits = outputs[0][0, -1, :]

        for id in set(generated):
            next_token_logits[id] /= repitition_penalty
        next_token_logits = next_token_logits / temperature

        # 筛掉不能出现的token和标点们
        # next_token_logits = filter_token_logits(next_token_logits, tokenizer, filtered_tokens)

        filtered_logits = next_token_logits

        filtered_logits = top_k_top_p_filtering(
            next_token_logits, top_k=top_k, top_p=top_p
        )

        next_token = torch.multinomial(
            F.softmax(filtered_logits, dim=-1), num_samples=1
        )
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        l += 1
        if l > min_sentence_length and l < max_sentence_length and tokenizer.decode(generated[0][-1]) == '[EOS]':
            break
        if l > max_sentence_length:
            break
    print(tokenizer.decode(generated.tolist()[0]))

def get_tokenizer_model(model_path, SPECIAL_TOKENS):
    tokenizer = load_tokenizer(model_path,SPECIAL_TOKENS)
    config_path = model_path+'config.json'
    model = load_pretrained_model(tokenizer,config_path, model_path, SPECIAL_TOKENS)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

if __name__ == '__main__':
    import logging
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    prose_gen = ProseGenerator(logger)
    sentence = 'The sun is responsible for '
    keywords = 'The sun is responsible for '
    # prose_gen.model_generate(keywords, 4)
    # prose_gen.model_caption_generate(sentence, 32, 128)

    # prose_gen.test('Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as')

    # # model_path = root_path+model_dir + 'pytorch_model.bin'
    # SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
    #                   "mask_token": "[MASK]",
    #                   "bos_token": "[BOS]", "eos_token": "[EOS]"}

    # model_path = '/users5/znchen/Question2Knowledge/model/trained_model/'
    model_path = '/users5/znchen/Question2Knowledge/model/GPT2/'
    tokenizer , model = get_tokenizer_model(model_path, SPECIAL_TOKENS)
    model_generate(tokenizer, model, keywords,  4, 40, SPECIAL_TOKENS)





