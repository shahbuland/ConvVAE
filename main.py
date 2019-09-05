from VAE import VAE
from drawboxes import *
import torch
import numpy as np

data = get_dataset(64)
data = np.expand_dims(data,3)
data = np.moveaxis(data,3,1)
data = data*2 - 1
print(data.shape)
data = torch.Tensor(data).float()
thingy = VAE(64,4, use_bn = True)
thingy.train(data, 1000, 4)
