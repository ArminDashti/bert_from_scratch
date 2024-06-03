'''
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
https://arxiv.org/pdf/1810.04805.pdf
'''
import os
import sys
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
import random
import transformers
from torch.utils.data.sampler import Sampler
import math
from datasets import load_dataset
from transformers import BertTokenizer
device='cpu'
sys_path = sys.path
#%%
imdb_dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#%%
raw_dataset = []
max_len = 0
for i in imdb_dataset['train']:
    max_len_src = len(i['text'])
    if max_len_src < 1000:
        tokenized = tokenizer(i['text'])
        raw_dataset.append([i['text'],tokenized,i['label']])
#%%
# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/position.py
class PositionalEmbedding(nn.Module):
    def __init__(self, bs=2, d_model=512, max_len=1000):
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
        self.pe = torch.stack((self.pe,self.pe),dim=0)
        return self.pe[:, :x.size(1)]
    
#%%
class encoder_transformer(nn.Module):
    '''
    Transformer Encoder
    Attention Is All You Need: https://arxiv.org/pdf/1706.03762.pdf
    '''
    def __init__(self, vocab_size=30522, seq_len=1000, H=8, N=4, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.H = H # Head
        self.N = N # Number of layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(3, d_model) # It's using in BERT
        self.positional_embedding = PositionalEmbedding(d_model, max_len=seq_len)
        self.LN = nn.LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=1)
        self.FF1 = nn.Linear(d_model, 2048)
        self.FF2 = nn.Linear(2048, d_model)
        self.relu = nn.ReLU()
        # Create Linear of q,k,v in H times
        head_module = nn.ModuleList([nn.Linear(d_model, int(d_model/H)),
                       nn.Linear(d_model, int(d_model/H)),
                       nn.Linear(d_model, int(d_model/H)),
                       nn.Linear(d_model, d_model)])
        self.heads_module = nn.ModuleList([head_module for i in range(H)])
        
        
    def MHA(self, x):
        ''' Multi Head Attention'''
        concat_attention = None
        
        for module in self.heads_module:
            q = module[0](x)
            k = module[1](x).transpose(1,2)
            v = module[2](x)
            linear = module[3]
            q_k = torch.matmul(q, k) / math.sqrt(self.d_model/self.H)
            attention = self.softmax(q_k)
            attention = torch.matmul(attention, v)
            if concat_attention is not None:
                concat_attention = torch.cat((concat_attention, attention), dim=2)
            else:
                concat_attention = attention
        
        return linear(concat_attention)
    
        
    def forward(self, x):
        embeded = self.token_embedding(x)
        
        x = self.positional_embedding(embeded)
        for layer in range(self.N):
            attention = self.MHA(x)
            attention_add = attention + x
            attention_add_norm = self.LN(attention_add)
            FF1_output = self.FF1(attention_add_norm)
            FF2_output = self.FF2(FF1_output)
            output_add = attention_add_norm + FF2_output
            x = self.LN(output_add)
        
        return x
        

class BERT(nn.Module):
    def __init__(self, seq_len=1000, mask_percent=15):
        super().__init__()
        self.encoder_transformer = encoder_transformer(vocab_size=30522, seq_len=1000).to(device)
        self.mask_percent = mask_percent
        self.seq_len = 1000
        self.linear = nn.Linear(512,2)
        self.softmax = nn.Softmax(1)

    
    def forward(self, x):
        
        encoder_output = self.encoder_transformer(x)

        return self.softmax(self.linear(encoder_output[:,0,:])), encoder_output[:,0,:]
        
    
class bert_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        seq = raw_dataset[idx][1]['input_ids']
        seq = torch.tensor(seq)
        padseq = torch.ones(1000)
        seq = torch.nn.utils.rnn.pad_sequence([seq, padseq], batch_first=False, padding_value=100.0)
        seq = seq[:,0]
        return seq.to(device), raw_dataset[idx][2] 
        

    def __len__(self):
        return len(self.dataset)

ds_train = bert_dataset(raw_dataset)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=0)       
            

x = torch.randint(102, 30000, (1, 50))
model = BERT(1000)
model = model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
for epoch in range(100):
    for i, data in enumerate(dl_train):
        pred, clss = model(data[0])
        loss = criterion(pred, data[1])
        sys.exit()
        
#%%
data[0].size()