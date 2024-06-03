import os
import sys
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
import yaml
import random
from torch.utils.data.sampler import Sampler
from ELECTRA.squad import *
import math
from armin_tools.NLP import squad
device='cpu'
#%%
# s = SQuAD('c:/users/armin/Downloads/dev-v2.0.json')
# qa = s.QandA_with_token_index()
# qa = s.padding(qa)
# wob = s.WoB()
# vocab_size = len(wob)+5
from datasets import load_dataset

dataset = load_dataset("squad_v2")

#%%
# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/position.py
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
#%%
def replace_token(seq, mask_percent, vocab_size= 20000, cls_token=1, sep_token=2, mask_token=3):
    token_mask_count = round(seq_len * mask_percent)
    i = -1
    while True:
        index = random.randint(0, len(seq) - 1)
        if seq[index] == cls_token or seq[index] == sep_token:
            i -= 1
        else:
            rnd = random.random()
            if rnd < 0.8:    
                seq[index] = mask_token
            elif 0.8 <= rnd < 0.9:
                seq[index] = random.randint(4, vocab_size + 1)
            else:
                pass
        i += 1 
        if i == token_mask_count:
            break
    return seq

#%%    
def MLM(seq, mask_percent=0.15):
    seq_len = len(seq) - 2 # Ignore [cls] and [sep]
    token_mask_count = round(seq_len * mask_percent) # How many token in a given seq should be masked. For example in 30 token seq is 4
    replaced_tokens = replace_token(seq, num_token=4, 100)
    return replaced_tokens

def NSP()
    
    
#%%
class Bert_Encoder(nn.Module):
    def __init__(self, N, H, src_vocab_size, d_model=768):
        super().__init__()
        self.pe = Positional_Encoding(30, d_model)
        self.d_model = d_model
        self.H = H
        self.N = N
        self.token_embedding = nn.Embedding(src_vocab_size, d_model)
        self.segment_embedding = nn.Embedding(3, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_len=30)
        self.LN = nn.LayerNorm(d_model)
        self.d_model_H = d_model/H
        self.q = nn.Linear(d_model, int(d_model/H))
        self.k = nn.Linear(d_model, int(d_model/H))
        self.v = nn.Linear(d_model, int(d_model/H))
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(d_model, d_model)
        self.FF1 = nn.Linear(d_model, 2048)
        self.FF2 = nn.Linear(2048, d_model)
        self.relu = nn.ReLU()
        
    def Multi_Head_Attn(self, qkv):
        q = self.q(qkv)
        k = self.k(qkv).transpose(1,2)
        v = self.v(qkv)
        q_k = torch.matmul(q, k) / math.sqrt(self.d_model/self.H)
        attn = self.softmax(q_k)
        attn = torch.matmul(attn, v)
        return attn
    
    def Encoder_Block(self, src):
        attn = self.Multi_Head_Attn(src)
        concated = attn
        for i in range(self.H-1):
            attn = self.Multi_Head_Attn(src)
            concated = torch.cat((concated, attn), 2)
        multi_attn = self.linear(concated)
        multi_attn = src + multi_attn
        multi_attn = self.LN(multi_attn)
        FF = self.FF1(multi_attn)
        FF = self.relu(FF)
        FF = self.FF2(FF)
        FF = FF + multi_attn
        FF = self.LN(FF)
        return FF
    
    def forward(self, src):
        src = self.encoder_embedding(src)
        src = src + self.pe
        for i in range(self.N):
            src = self.Encoder_Block(src)
        return src
    

class Bert(nn.Module):
    def __init__ (self, layers, heads, src_vocab_size, d_model=768):
        super().__init__()
        self.Bert_Encoder = Bert_Encoder(layers, heads, src_vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.LogSoftmax(dim=2)
        self.output = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, src):
        output_bert = self.Bert_Encoder(src)
        output_bert = self.linear(output_bert)
        output_bert = self.softmax(output_bert)
        output_bert = self.output(output_bert)
        return output_bert
    
    
class create_dataset(torch.utils.data.Dataset):
    def __init__(self, qa):
        self.qa = qa

    def __getitem__(self, idx):
        return torch.tensor(qa[idx]).to(torch.float)
    
    def __len__(self):
        return len(self.qa)
    
ds = create_dataset(qa)
dl = torch.utils.data.DataLoader(ds, batch_size=16, drop_last=True, shuffle=False)
model = Transformer(N=6, H=12, src_vocab_size=vocab_size, trg_vocab_size=vocab_size).to(torch.float)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
loss_func = nn.NLLLoss()
next(iter(dl)).size()
#%%
for i, seq in enumerate(dl):
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        out = model(seq)
        out = torch.argmax(out, dim=2)
        out = out.to(torch.float).requires_grad_()
        seq = seq.to(torch.float).requires_grad_()
        loss = loss_func(out, seq)
        print(loss)
        loss.backward()
        optimizer.step()
        if i == 20:
            sys.exit()