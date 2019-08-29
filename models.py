import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Gets list of channels required to get size (assumed int[3], and square)
# separate for encoding (CHANNELSxHxW -> 512xH_SIZExH_SIZE)
def get_ch_list_encoding(shape, h_size):
    assert len(shape) == 3
    assert shape[1] == shape[2]
    size = shape[1]
    f = shape[0] # Keep track of current filter count
    ch = [shape[0]] # List of num filters (channels)
    while size != h_size and size != 1:
        size = size // 2
        if f == shape[0]: f = 64
        else: f *= 2
        ch.append(f)
    return ch

def get_ch_list_decoding(h_shape, shape):
    assert len(shape) == 3
    assert shape[1] == shape[2]
    size = shape[0]
    h_size = h_shape[1]
    f = h_shape[0]
    ch = [f]
    while h_size != size:
        h_size *= 2
        if f == 64: f = shape[0]
        else: f = f // 2
        ch.append(f)
    return ch

class Encoder(nn.Module):
    # The first two are required
    # CHANNELS is 1 if gray, 3 if RGB
    # The rest are optionial 
    def __init__(self, input_size, latent_dim, CHANNELS=1, H_SIZE=4,
                 use_bn=False, use_dropout=False, dp_prob=0.25):
        super(Encoder, self).__init__()
        
        # init some constants for model
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.CHANNELS = CHANNELS
        self.H_SIZE = H_SIZE
        self.dp_prob = dp_prob
        
        # Need to figure out required channels to get from
        # [CHANNELS,input_size,input_size] -> [512,H_SIZE,H_SIZE]
        ch = get_ch_list_encoding([CHANNELS,input_size,input_size], H_SIZE)
        self.n_layers = len(ch) - 1
        # Module lists for layers
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList() #if batch norm is enabled
        self.dp = nn.ModuleList() #if dropout is enabled
    
        # Make conv layers
        for i in range(self.n_layers):
            self.conv.append(self.myconv(ch[i],ch[i+1]))

        # Make batch norm layers
        if use_bn:
            for i in range(self.n_layers):
                self.bn.append(nn.BatchNorm2d(ch[i+1]))

        # Make dropout layers
        if use_dropout:
            for i in range(self.n_layers):
                self.dp.append(nn.Dropout2d(dp_prob))


    def myconv(self, fi, fo, k=4,s=2,p=1):
        return nn.Conv2d(fi,fo,k,s,p)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.conv[i](x)
            if self.use_bn: x = self.bn[i](x)
            if self.use_dropout: x= self.dp[i](x)
        return x
