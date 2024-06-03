import torch
from torch import nn
#%%
lstm = torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=1, bias=False)

for param in lstm.parameters():
    # param = nn.parameter(torch.ones(param.size()))
    # print(param)
    print(param.size())
    
rnn = nn.LSTM(10, 40, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 40)
c0 = torch.randn(2, 3, 40)
output, (hn, cn) = rnn(input, (h0, c0))
#%%
# input = torch.randn(5, 3, 10) * torch.rand(20,10)
torch.matmul(torch.randn(5, 3, 10) , torch.rand(20,10).T)
#%%
param.weight