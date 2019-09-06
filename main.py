from VAE import VAE
from drawboxes import *
import torch
import numpy as np
import ezdatasets as ezd


data = ezd.GetTrainingData("pokemon", shape=(128,128),norm_style="tanh")
data = np.moveaxis(data,3,1)
print(data.shape)
data = torch.Tensor(data).float()
thingy = VAE(128,32, CHANNELS=3, use_bn = True, use_dropout = False)
thingy.train(data, 2000, 16)
