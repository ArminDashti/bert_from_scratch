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
import random
from datasets import load_dataset
from transformers import BertTokenizer
device='cpu'
sys_path = sys.path
#%%
squad = load_dataset("squad_v2")
imdb = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#%%
dataset = []
max_len = 0
for i in imdb['train']:
    max_len_src = len(i['text'])
    if max_len_src < 1000:
        dataset.append([i['text'],i['label']])
max_len_src
#%%
max_len = 0
qa = {}
questions = dataset['train']['question']
answers = dataset['train']['answers']
for i, question in enumerate(questions):
    question = question.strip()
    answer = answers[i]['text']
    if len(answer) != 0:
        answer = answers[i]['text'][0].strip()
        question_len = len(question)
        answer_len = len(answer)
        if (question_len + answer_len) > max_len:
            max_len = (question_len + answer_len)
            if max_len == 315: sys.exit()
        answer = answers[i]['text']
        qa[question.strip()] = answers[i]['text'][0].strip()
#%%
qa_list = []
for question, answer in qa.items():
    question_answer = question + ' [SEP] ' + answer
    qa_list.append(tokenizer(question_answer)['input_ids'])
#%%
qa_list[0]
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
class encoder_transformer(nn.Module):
    '''
    Transformer Encoder
    Attention Is All You Need: https://arxiv.org/pdf/1706.03762.pdf
    '''
    def __init__(self, vocab_size, seq_len, H=8, N=4, d_model=512):
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
    def __init__(self, seq_len, mask_percent=15):
        super().__init__()
        self.encoder_transformer = encoder_transformer(vocab_size=20000, seq_len=50).to(device)
        self.mask_percent = mask_percent
        self.seq_len = 50
        
    def MLM(self, x):
        random_num = random.random()
        token_mask_num = int(self.seq_len * self.mask_percent)
        x
        if random_num <= 80:
            pass
    
    
    def forward(self, x):
        return self.encoder_transformer(x)
        
        
            

x = torch.rand(1,50,512)
model = BERT(50)
pred = model(x)
pred.size()
#%%


def pick_random_numbers_without_duplicates(tensor):
  """Picks 3 random numbers from the tensor without duplicates."""
  random_numbers = []
  seen = set()
  for _ in range(2):
    while True:
      random_index = random.randint(0, len(tensor) - 1)
      if random_index not in seen:
        break
    random_numbers.append(tensor[random_index])
    seen.add(random_index)
  return random_numbers

random_numbers = pick_random_numbers_without_duplicates([1,2,3,4,5,6])
print(random_numbers)