import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mimage
import cv2, os


def plot_img_grid(img_list, nrow, ncol):

	gs = gridspec.GridSpec(nrow, ncol, 
		top=1., bottom=0., right=1., left=0., 
		hspace=0., wspace=0.)

	for i, grid in enumerate(gs):
		row = i // ncol
		col = i % ncol
		print(row, col)
		ax = plt.subplot(grid)
		ax.imshow(img_list[i])
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('auto')

	#plt.show()
	plt.savefig('save.jpg')