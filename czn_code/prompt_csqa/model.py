import math
import torch
import torch.nn as nn
import numpy as np
import wandb

from utils import log_sum_exp


import logging
logger = logging.getLogger(__name__)

class Prefix_GPT(nn.Module):
    '''
    Prompt tuning.
    '''
    def __init__(self, decoder, args): # 
        super(Prefix_GPT, self).__init__()
        self.decoder = decoder #GPT2LMHeadModel


        self.decoder_config = decoder.config
        self.decoder_num_layer = self.decoder_config.n_layer
        self.decoder_hidden_size = self.decoder_config.n_embd
        self.decoder_num_head = self.decoder_config.n_head
        

        

        self.args = args
        self.seq_len = args.seq_len
        self.flat = args.flat
        self.inter_dim = args.inter_dim
        assert self.decoder_hidden_size % self.decoder_num_head == 0
        self.embed_size_per_head = self.decoder_hidden_size // self.decoder_num_head
        if args.flat:
            self.latent = nn.Parameter(nn.init.normal_(torch.rand(1, self.decoder_num_layer, 2, self.decoder_num_head, self.seq_len, self.embed_size_per_head), std=0.02))
        else:
            self.input_tokens = torch.arange(self.seq_len).long()
            self.wte = nn.Embedding(self.seq_len, self.decoder_hidden_size)
            self.control_trans = nn.Sequential(
                torch.nn.Linear(self.decoder_hidden_size, self.inter_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(self.inter_dim, self.decoder_num_layer * 2 * self.decoder_hidden_size),
            )
        self.dropout = nn.Dropout(self.decoder_config.attn_pdrop)
        self.fix_decoder()


    def convert_to_past_key_values(self, batch_size):
        if self.flat:
            x = self.latent.expand(batch_size,-1,-1,-1,-1,-1).permute(1,2,0,3,4,5)
        else:
            device = next(self.parameters()).device
            x = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
            x = self.control_trans(self.wte(x))
            x = x.view(batch_size, self.decoder_num_layer, 2, self.decoder_num_head, self.seq_len, self.embed_size_per_head).permute(1,2,0,3,4,5)
        x = self.dropout(x)
        past_key_values = []
        for i in range(self.decoder_num_head):
            past_key_values.append((x[i][0],x[i][1],))
        assert past_key_values[0][0].requires_grad == True
        return tuple(past_key_values)


    def fix_decoder(self):
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False


    def forward(self, input_ids, attention_mask):
        '''
        Forward method for training which returns a reconstruction loss.
        Args:
            input_ids,
            attention_mask:
                Outputs of GPT2Tokenizer(List of Strings, return_tensors='pt', padding=True)

        '''
        batch_size = input_ids.shape[0]
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(input_ids.device)
        attention_mask = torch.cat([infix_attn, attention_mask], dim=-1)

        past_key_values = self.convert_to_past_key_values(batch_size)

        outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, past_key_values=past_key_values, return_dict=True)
        loss = outputs.loss

        return (loss,)
    

    def generate(self, batch_size=None, input_ids=None, attention_mask=None, generate_len=50, topk=5, use_cache=True):
        '''
        Generate text with given latent represention.
        '''
        device = next(self.parameters()).device

      
        
        if input_ids is None:
            if batch_size is None:
                batch_size = 1
            input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]]*batch_size).to(device), max_length=3, do_sample=True)[:,1:]
            attention_mask = torch.ones(batch_size, 2).bool()
        else:
            if batch_size is None:
                batch_size = input_ids.shape[0]
            input_ids = input_ids.to(device)
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, 2).bool()

        
        cur_len = input_ids.shape[1]
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(device)
        attention_mask = torch.cat([infix_attn, attention_mask.to(device)], dim=-1)
        
        past_key_values = self.convert_to_past_key_values(batch_size)
        ###
        if cur_len == 0:
            raise Exception('length of input_ids error')
        elif cur_len == 1:
            result = self.decoder.generate(input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, do_sample=True, top_k=topk, top_p=0.9, length_penalty=1.0, max_length=50, min_length=30)
        else:
            past_key_values = self.decoder(input_ids=input_ids[:,:-1], attention_mask=attention_mask[:,:-1], past_key_values=past_key_values, return_dict=True, use_cache=True).past_key_values
            result = self.decoder.generate(input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, do_sample=True, top_k=topk, top_p=0.9, length_penalty=1.0, max_length=50, min_length=30)

        return result
