import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

image_path = 'fish_images/fredfish30.jpg'
img = imread(image_path)
gray_img = imread(image_path, mode='L')
y, x = np.nonzero(gray_img)

x = x - np.mean(x)
y = y - np.mean(y)

coords = np.vstack([x, y])
cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)

sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
x_v2, y_v2 = evecs[:, sort_indices[1]]

scale = 150

extent = [x.min(), x.max(), y.min(), y.max()]  

plt.imshow(img, extent=extent, origin='lower')

plt.plot([x_v1*-scale*2, x_v1*scale*2],
         [y_v1*-scale*2, y_v1*scale*2], color='red')
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left

plt.show()