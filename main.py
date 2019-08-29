import torch
from models import Encoder
from torchsummary import summary

E = Encoder(input_size=128, latent_dim=1024,
            CHANNELS=3)
E.cuda()

summary(E,input_size=(3,128,128))
print("checkpoint")
X = torch.ones(10,3,128,128).cuda()
Y = E(X)
print("done")
print(X.shape)
print(Y.shape)
