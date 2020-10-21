import argparse
from skimage.util import img_as_float32
from skimage.transform import resize
from train_model import own_rel
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from skimage import io
from skimage.io import imshow, show, imsave
import shutil

import lime_aleph as la


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="../datasets/single_relation/test/pos/pos9000.png", help="Path to the image.")
ap.add_argument("-s", "--samples", required=False, default=1000, help="Number of samples for LIME.")
ap.add_argument("-k", "--keep", required=False, default=3, help="Number of important superpixels.")
ap.add_argument("-c", "--checkpoint_dir", required=False, default="../models_to_explain/model.h5", help="Path to checkpoint.")
ap.add_argument("-o", "--output_dir", required=False, default="../output/", help="Path to the directory of output files.")
ap.add_argument("-t", "--theta", required=False, default=0.9, help="The threshold of the classifier estimator for being a true example.")
ap.add_argument("-n", "--noise", required=False, default=10, help="Percentage of false positives allowed for Aleph.")
args = vars(ap.parse_args())


NUM_SAMPLES_LIME = int(args['samples'])
N_KEEP = int(args['keep'])
CHECKPOINT_DIR = str(args['checkpoint_dir'])
OUTPUT_DIR = str(args['output_dir'])
THRESHOLD_TRUE_CLASS = float(args['theta'])
NOISE = int(args['noise'])

# completely remove the output directory and create a new one
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR)

# load in and resize original image to match network default size
image = img_as_float32(io.imread(args["image"]))
image = resize(image, (own_rel.IMAGE_SIZE, own_rel.IMAGE_SIZE), anti_aliasing=True)

# import model
model = own_rel.own_rel()

# load weights from checkpoint
model.load_weights(CHECKPOINT_DIR)

# get the annotated image
annotated_image = la.annotate_image_parts(image, model, OUTPUT_DIR, NUM_SAMPLES_LIME)

# get the list of the important superpixels
important_superpixels = la.find_important_parts(annotated_image, N_KEEP)

# find the spatial relations between them
relations = la.find_spatial_relations(important_superpixels)

for rel in relations:
    print("Name:", rel.name)
    print("Start:", rel.start)
    print("To:", rel.to)

# Build the dataset of perturbed versions of the image
perturbed_dataset = la.perturb_instance(annotated_image, relations, model, THRESHOLD_TRUE_CLASS)

for ex in perturbed_dataset:
    print(ex.positive)
    print(ex.superpixels)
    for rel in ex.relations:
        print("\t", rel.name)
        print("\t", rel.start)
        print("\t", rel.to)