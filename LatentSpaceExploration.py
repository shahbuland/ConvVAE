from VAE import VAE
import torch
import numpy as np
import matplotlib.pyplot as plt

# Converts image into format for PLT
# Then shows image
def show(a):
	a = a.detach().cpu().numpy()
	a = np.squeeze(a)
	a = 0.5*a + 0.5
	plt.imshow(a, cmap='gray')
	plt.show()
	plt.close()
    
# Make model and load parameters
# In this case loaded for images with size 128,
# Latent dim of 32
model = VAE(128,32,CHANNELS=3,use_bn=True)
model.load("VAEparams.pt")

while(True):	
	vec = np.random.randn(32)
	vec = torch.from_numpy(vec).float()
	result = model.decode(vec)
	show(result)
