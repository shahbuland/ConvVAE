from VAE import VAE
from drawboxes import *
import torch
import numpy as np


data = get_dataset(32)
data = np.expand_dims(data,1)
print(data.shape)
data = torch.Tensor(data).float()
thingy = VAE(32,32, CHANNELS=1, use_bn = True)
thingy.train(data, 2000, 8, sample_interval=1000)
