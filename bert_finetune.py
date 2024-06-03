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
from transformers import AutoTokenizer, BertModel
import torch

from transformers import AutoTokenizer, BertForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
#%%
question, text = "What framework ramin use?", "Ramin is a Game developer and He working with Unity. He was working in Medrick company. Ramin don't like Iran and He love video games."

inputs = tokenizer(question, text, return_tensors="pt")
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
print(tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))
#%%
# target is "nice puppet"
target_start_index = torch.tensor([12])
target_end_index = torch.tensor([12])

opt = torch.optim.Adam(model.parameters(), lr=0.00005)
model.train()
outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss
round(loss.item(), 2)
print(loss.item())
opt.zero_grad()
loss.backward()
opt.step()
#%%
# tokenizer.decode(torch.tensor([14384]), skip_special_tokens=True)
model
#%%
question, text = "Where Armin working?", "Armin is a AI developer. Ramin was working in Medrick But Armin is working in HiWEB."

inputs = tokenizer(question, text, return_tensors="pt")
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
print(tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))