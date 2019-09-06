from VAE import VAE
import torch
import numpy as np
import ezdatasets as ezd
import matplotlib.pyplot as plt

def show(a):
    a = a.detach().cpu().numpy()
    a = np.squeeze(a)
    a = np.moveaxis(a,0,2)
    a = 0.5*a + 0.5
    plt.imshow(a)
    plt.show()
    plt.close()
    
data = ezd.GetTrainingData("pokemon", shape=(128,128),norm_style="tanh")
data = np.moveaxis(data,3,1)
print(data.shape)
data = torch.Tensor(data).float()
model = VAE(128,32,CHANNELS=3,use_bn=True)
model.load("VAEparams.pt")

vectors = model.encode(data.cuda())
vector = vectors[0] + vectors[3]

result = model.decode(0.5*vector)

for i in [data[0],data[3],result]:
    show(i)
