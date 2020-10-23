from skimage import io
import numpy as np
import sys
from train_model import own_rel
from skimage.io import imshow, show, imsave, imread
import copy
import os
import shutil

IMAGE_SIZE = own_rel.IMAGE_SIZE


###################
# START OF SCRIPT #
###################

unique_names = set()
unique_colors = set()





### generate aleph ###




generate_rel_ex_and_bk(image, mask, important_superpixels, true_class)


