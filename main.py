'''
    Main
'''
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
from six.moves.urllib.request import urlopen

import glob
import os
from itertools import accumulate

from Defs2LandmarksRecognitionCNN import *

if __name__ == "__main__":
    np.random.seed(10)
    IMAGE_1_JPG = 'DJI_0816.JPG'
    IMAGE_2_JPG = 'DJI_0817.JPG'
    show_images([IMAGE_1_JPG, IMAGE_2_JPG])
    pass