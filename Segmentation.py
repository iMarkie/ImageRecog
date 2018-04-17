
import matplotlib.pyplot as plt 
import numpy as np 
import skimage 
from skimage.data import astronaut 
from skimage.color import rgb2gray 
from skimage.filters import sobel 

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed 
from skimage.segmentation import mark_boundaries 
from skimage.io import imread 
from skimage.util import img_as_float 

img = imread('water3.tif')[:500,:500] 
#img2 = imread('water2.tif') 

segments_fz = felzenszwalb(img, scale=200, sigma=0.95, min_size=100) 
segments_fz2 = felzenszwalb(img, scale=200, sigma=0.95, min_size=100) 
#segments_fz2 = felzenszwalb(img2, scale=200, sigma=0.95, min_size=100) 

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz)))) 
#print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz2)))) 

fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True) 
ax[0].imshow(mark_boundaries(img, segments_fz)) 
ax[0].set_title("Felzenszwalbs's method") 
#ax[1].imshow(mark_boundaries(img2, segments_fz2)) 
#ax[1].set_title("Felzenszwalbs's method") 
for a in ax.ravel():
    a.set_axis_off() 
    plt.tight_layout() 
    plt.show()

#####
# Zijn de segmenten gelijk?
#####
np.array_equal(segments_fz,segments_fz2) 
