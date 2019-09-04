import matplotlib.pyplot as plt
import numpy as np

class Canvas:
	def __init__(self, height, width):
		self.height = height
		self.width = width
		self.canvas = np.zeros((height,width))
	def add_box(self, r, c, w, h):
		self.canvas[r:min(r+h,self.height-1),c:min(c+w,self.width-1)] = 1
	def draw(self):
		plt.imshow(self.canvas,cmap='gray')
		plt.show()
		plt.close()

def get_dataset(size):
	data = []
	for i in range(500):
		c = Canvas(size,size)
		c.add_box(np.random.randint(0,size),
				  np.random.randint(0,size),
				  np.random.randint(0,size),
				  np.random.randint(0,size))
		data.append(c.canvas)
	return np.asarray(data)

