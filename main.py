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
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.FATAL)

    m = hub.Module('https://tfhub.dev/google/delf/1')
    image_placeholder = tf.placeholder(
    tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs = {
        'image': image_placeholder,
        'score_threshold': 100.0,
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        'max_feature_num': 1000,
    }

    module_outputs = m(module_inputs, as_dict=True)

    image_tf = image_input_fn([IMAGE_1_JPG, IMAGE_2_JPG])
    pass