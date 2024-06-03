import torch
from torch import nn
# http://www.songho.ca/dsp/convolution/convolution2d_example.html
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
#%%
in_cahnnel = 3
out_channel = 6
img = torch.rand(3,5,5)
#%%
conv2d = nn.Conv2d(3, 6, 2, bias=False)
# conv2d.weight.data.fill_(1)
conv2d_weight = conv2d.weight.data
print("Shape of conv2d is {}".format(conv2d.weight.size()))
# conv2d(img)
#%%
sum_w = 0
for i in range(in_cahnnel):
    ml = img[0] * conv2d_weight[0][0]
    
    
#%%
img[0].size()
torch.matmul(img[0].unsqueeze(0) , conv2d_weight[0][0].T)