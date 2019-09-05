import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from VAE import VAE

model = VAE(64,2, use_bn = True)
model.load("VAEparams.pt")

fig,axs = plt.subplots(1,2)
axs[0].plot(50,50)
axs[0].plot(-50,-50)

V = np.array([[1,1]])
origin = [0],[0]
def update():
	x,y = V[0][0], V[0][1]
	print(x,y)
	sqr = model.decode_np(np.array([x,y]))
	axs[1].imshow(sqr,cmap='gray')	
	fig.canvas.draw_idle()

def x_update(x):
	V[0][0] = x
	update()

def y_update(y):
	V[0][1] = y
	update()

def onclick(event):
	ix, iy = event.xdata, event.ydata
	x_update(ix)
	y_update(iy)
		
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

