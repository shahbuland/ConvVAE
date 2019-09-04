import torch

# Weight initialization function
def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)

# Generating random indices (between a and b)
def randint(size, a, b):
	assert a < b
	n = torch.rand(size) # [0,1]
	n *= (b-a) # [0, b-a]
	n += a # [a, b]
	return n.round().long() # has to be long to work as indices

# KL Loss
def KL_Loss(mu, logvar):
	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
