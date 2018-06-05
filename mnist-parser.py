import struct
import gzip
import numpy as np

def read_idx(filename):
	with gzip.open(filename) as f:
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def show(image):
	"""
	Render a given numpy.uint8 2D array of pixel data.
	"""
	from matplotlib import pyplot
	import matplotlib as mpl
	fig = pyplot.figure()
	ax = fig.add_subplot(1,1,1)
	imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
	imgplot.set_interpolation('nearest')
	ax.xaxis.set_ticks_position('top')
	ax.yaxis.set_ticks_position('left')
	pyplot.show()

dataset_train_images_path = "/home/zahid/Desktop/tensorflow-tut/datasets/train-images-idx3-ubyte.gz"
dataset_train_labels_path = "/home/zahid/Desktop/tensorflow-tut/datasets/train-labels-idx1-ubyte.gz"
dataset_test_images_path = "/home/zahid/Desktop/tensorflow-tut/datasets/t10k-images-idx3-ubyte.gz"
dataset_test_images_path = "/home/zahid/Desktop/tensorflow-tut/datasets/t10k-labels-idx1-ubyte.gz"

#dataset_train_images = read_idx(dataset_train_images_path)
#print(dataset_train_images)

#image = dataset_train_images[0]
#show(image)
