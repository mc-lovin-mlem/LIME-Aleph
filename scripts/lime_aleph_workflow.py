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
import cv2
from graphviz import Digraph

from lime_aleph import lime_aleph as la


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="../datasets/tower/towers_for_experiment/test/pos/pos9002.png", help="Path to the image.")
ap.add_argument("-s", "--samples", required=False, default=1000, help="Number of samples for LIME.")
ap.add_argument("-k", "--keep", required=False, default=4, help="Number of important superpixels.")
ap.add_argument("-c", "--checkpoint_dir", required=False, default="../models_to_explain/model_tower.h5", help="Path to checkpoint.")
ap.add_argument("-o", "--output_dir", required=False, default="../output/", help="Path to the directory of output files.")
ap.add_argument("-t", "--theta", required=False, default=0.8, help="The threshold of the classifier estimator for being a true example.")
ap.add_argument("-n", "--noise", required=False, default=10, help="Percentage of false positives allowed for Aleph.")
args = vars(ap.parse_args())



NUM_SAMPLES_LIME = int(args['samples'])
N_KEEP = int(args['keep'])
CHECKPOINT_DIR = str(args['checkpoint_dir'])
OUTPUT_DIR = os.path.abspath(str(args['output_dir'])) + "/"
THRESHOLD_TRUE_CLASS = float(args['theta'])
NOISE = int(args['noise'])

# completely remove the output directory and create a new one
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR)

# load in and resize original image to match network default size

image_filepath = os.path.abspath(args["image"])
image = img_as_float32(io.imread(image_filepath))
image = resize(image, (own_rel.IMAGE_SIZE, own_rel.IMAGE_SIZE), anti_aliasing=True)

# import model
model = own_rel.own_rel()

# load weights from checkpoint
model.load_weights(CHECKPOINT_DIR)

# get the annotated image
annotated_image = la.annotate_image_parts(image, model, OUTPUT_DIR, NUM_SAMPLES_LIME)

# get the list of the important superpixels
important_superpixels, labeled_image = la.find_important_parts(annotated_image, N_KEEP)

# Displaying the labeled image 
io.imshow(labeled_image)
io.show()

# find the spatial relations between them
relations, graph = la.find_spatial_relations(important_superpixels)

for rel in relations:
    print(rel)

graph.render('relations-graph.gv', view=True)


# Build the dataset of perturbed versions of the image
perturbed_dataset = la.perturb_instance(annotated_image, relations, model, THRESHOLD_TRUE_CLASS)

for ex in perturbed_dataset:
    print(ex.positive)
    print(ex.superpixels)
    for rel in ex.relations:
        print("\t", rel.name)
        print("\t", rel.start)
        print("\t", rel.to)

# Write the Aleph files from the background knowledge
used_relations = None # 'None' if you want to allow all relations, otherwise list with following possibilities: ["left_of", "right_of", "top_of", "bottom_of", "on", "under"]
la.write_aleph_files(annotated_image, perturbed_dataset, used_relations, OUTPUT_DIR, NOISE)

# Run Aleph and obtain the rules
la.run_aleph(OUTPUT_DIR)
