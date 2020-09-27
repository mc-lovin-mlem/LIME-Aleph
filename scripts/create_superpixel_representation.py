from skimage.segmentation import slic, quickshift, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float32
from skimage import io
from skimage.draw import circle
from skimage.transform import resize
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image as imagepkg
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from train_model import own_rel
import webcolors
from lime import lime_image
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from sources.own_segmentation.scikit_image import SegmentationAlgorithm
from skimage.io import imshow, show, imsave
import copy
import time
import xml.etree.ElementTree as ET
import webbrowser
import os


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", default="cat.png", help="Path to the image.")
ap.add_argument("-s", "--samples", required=False, default=1000, help="Number of samples for LIME.")
ap.add_argument("-f", "--features", required=False, default=3, help="Number of features to show in LIME result.")
ap.add_argument("-c", "--checkpoint_dir", required=False, default="../models_to_explain/model.h5", help="Path to checkpoint.")
args = vars(ap.parse_args())

IMAGE_SIZE = own_rel.IMAGE_SIZE
NUM_SAMPLES_LIME = int(args['samples'])
NUM_FEATURES_TO_SHOW_LIME = int(args['features'])
CHECKPOINT_DIR = str(args['checkpoint_dir'])
RANDOM_SEED = 42

OUTPUTS_PATH = "../outputs/outputs_lime/"
if not os.path.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)

#import model
model = own_rel.own_rel()

#load weights from checkpoint
model.load_weights(CHECKPOINT_DIR)

#compile model (required to make predictions)
#model.compile(loss = 'binary_crossentropy',
#              optimizer = optimizers.RMSprop(lr = 1e-4),
#              metrics = ['acc'])

#Predict probabilities
def predict_fn(images):
    transformed_images = own_rel.transform_img_fn(images)
    preds = model.predict_proba(np.array(transformed_images))
    return preds

"""
The following two functions are from the stackoverflow post
'https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python'
It is therefore under the CC-BY-SA License: https://creativecommons.org/licenses/by-sa/2.0/legalcode
The author is 'fraxel'
"""
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name



###################
# START OF SCRIPT #
###################
##Reference Image
# load in and resize original image
image = img_as_float32(io.imread(args["image"]))

#resize image to match network default size
image = resize(image, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
io.imsave(OUTPUTS_PATH + "reference_image.png", image) # save the original as reference image

true_class = -1

predictions = predict_fn([image]) #Probabilities
predictions = np.squeeze(predictions) #to get rid of the third dimension
true_class = np.argmax(predictions)
print("True class: ", true_class)

# the softmax output for the two classes:
print("Negative estimator:", predictions[0])
print("Positive estimator:", predictions[1])


# here, we start to create an xml annotation file
root = ET.Element("image")
root.set("true-class", str(true_class))
tree = ET.ElementTree(root)

explainer = lime_image.LimeImageExplainer(verbose=True)

print('Starting the explanation generation process. This may take several minutes. Seriously. Grab a cup of coffee.')

sf = SegmentationAlgorithm('quadratic', n_quads=8) # our new segmentation algorithm. A uniform grid.

tmp = time.time()
explanation = explainer.explain_instance(image, predict_fn, segmentation_fn=sf, top_labels=2, num_samples=NUM_SAMPLES_LIME, hide_color=0, random_seed=RANDOM_SEED) #, batch_size = 1)
print("Elapsed time: " + str(time.time() - tmp))

lime_output, lime_mask = explanation.get_image_and_mask(true_class, positive_only=True, num_features=NUM_FEATURES_TO_SHOW_LIME, hide_rest=True)
io.imshow(lime_output)
io.show()
io.imsave(OUTPUTS_PATH + "lime_output.png", mark_boundaries(lime_output, lime_mask))

# Beware: Only applicable to binary tasks
counter_class = -1
if true_class == 0:
	counter_class = 1
else:
	counter_class = 0

lime_output_counter, lime_mask_counter = explanation.get_image_and_mask(counter_class, positive_only=True, num_features=NUM_FEATURES_TO_SHOW_LIME, hide_rest=True)
io.imsave(OUTPUTS_PATH + "lime_output_counter.png", mark_boundaries(lime_output_counter, lime_mask_counter))

top_class_weights = explanation.local_exp[true_class]
#print("Top Class Weights:", top_class_weights)
segments = explanation.segments

print("Number of superpixels: " + str(len(top_class_weights)))

for f,w in top_class_weights:
    #Print weights
    print("Weight of sp: ", f, "is: ", w)

	# get the size of a superpixel
    sp_size_in_pixels = len(segments[segments == f])
	
	# get the mean color for the superpixel
    sp_mean_color = (
        np.mean(image[segments == f][:, 0] * 255.0),
        np.mean(image[segments == f][:, 1]) * 255.0,
        np.mean(image[segments == f][:, 2]) * 255.0)
	
	# from stackoverflow
    _, closest_name = get_colour_name(sp_mean_color)
	
	# get the coordinates of the middle of the superpixel
    occurrences = np.where(segments==f)
    x = (np.max(occurrences[0]) + np.min(occurrences[0]))/2.0
    y = (np.max(occurrences[1]) + np.min(occurrences[1]))/2.0
	
	# add to the xml
    sp_element = ET.SubElement(root, 'superpixel')
    sp_element.set('id', 'sp_' + str(f))
    sp_element.set('name', 'sp_' + str(f))
    sp_element.set('color', str(closest_name))
    sp_element.set('size', str(sp_size_in_pixels))
    sp_element.set('lime-weight', str(w))
    sp_element.set('coord-x', str(x))
    sp_element.set('coord-y', str(y))
    tree.write(OUTPUTS_PATH + "annotation.xml")

# save sp integer mask
np.save(OUTPUTS_PATH + "sp_mask.npy", segments)