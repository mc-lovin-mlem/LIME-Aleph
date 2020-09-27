from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float32
from skimage import io
from skimage.draw import circle
from skimage.transform import resize
import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys
from lime import lime_image
from train_model import own_rel
from skimage.io import imshow, show, imsave, imread
from xml.etree.ElementTree import parse
import copy
import os
import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import optimizers
from skimage.segmentation import mark_boundaries
import shutil

#np.set_printoptions(threshold=np.nan)

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--annotation_file", required=False,
                default="../outputs/outputs_lime/annotation.xml", help="Path to the annotation file.")
ap.add_argument("-s", "--samples", required=False, default=20,
                help="The number of perturbed training examples for Aleph.")
ap.add_argument("-c", "--checkpoint_dir", required=False,
                default="../models_to_explain/model.h5", help="Path to checkpoint.")
ap.add_argument("-k", "--keep", required=False,
                default=4, help="Number of important superpixels.")
ap.add_argument("-t", "--theta", required=False,
                default=0.9, help="The threshold of the classifier estimator for being a true example.")
ap.add_argument("-n", "--noise", required=False,
                default=10, help="Percentage of false positives allowed for Aleph.")
args = vars(ap.parse_args())

IMAGE_SIZE = own_rel.IMAGE_SIZE
NUM_SAMPLES_LIME_ALEPH = int(args['samples'])
BATCH_SIZE = 10
ANNOTATION_FILE = args['annotation_file']
N_KEEP = int(args['keep'])
THRESHOLD_TRUE_CLASS = float(args['theta'])
CHECKPOINT_DIR = str(args['checkpoint_dir'])
NOISE = int(args['noise'])


# clear all perturbed examples
OUTPUTS_PATH = "../outputs/outputs_aleph/"
shutil.rmtree(OUTPUTS_PATH, ignore_errors=True)
os.makedirs(OUTPUTS_PATH + 'perturbed/')
os.makedirs(OUTPUTS_PATH + 'aleph_files/')

aleph_b = open(OUTPUTS_PATH + 'aleph_files/sp.b', 'w')
aleph_f = open(OUTPUTS_PATH + 'aleph_files/sp.f', 'w')
aleph_n = open(OUTPUTS_PATH + 'aleph_files/sp.n', 'w')
aleph_bk = open(OUTPUTS_PATH + 'aleph_files/sp.bk', 'w')


class Relation:
    pass

class Superpixel:
    pass


def dissim_sps(sp_mask, original_image):
    output = []
    n_superpixels = np.unique(sp_mask).shape[0]

    original_image_mean_r = np.mean(original_image[:, 0])
    original_image_mean_g = np.mean(original_image[:, 1])
    original_image_mean_b = np.mean(original_image[:, 2])

    for i in range(n_superpixels):
        r_sum = np.sum(np.abs(original_image[sp_mask == i][:, 0] - original_image_mean_r))
        g_sum = np.sum(np.abs(original_image[sp_mask == i][:, 1] - original_image_mean_g))
        b_sum = np.sum(np.abs(original_image[sp_mask == i][:, 2] - original_image_mean_b))

        mean_sp = (r_sum + g_sum + b_sum) / 3.0 * (1.0/(300*300))
        output.append(mean_sp)

    return output


def dissim(row, dissims):
    n_superpixels = len(row)

    dissim_image = 1.0 / n_superpixels

    additive = 0.0
    for i in range(len(row)):
        if row[i] == 0:
            additive += dissims[i]
    dissim_image += additive

    return dissim_image


#import model
model = own_rel.own_rel()

# load weights from checkpoint
model.load_weights(CHECKPOINT_DIR)

# compile model (required to make predictions)
# model.compile(loss = 'binary_crossentropy',
#              optimizer = optimizers.RMSprop(lr = 1e-4),
#              metrics = ['acc'])


def predict_fn(images):
    transformed_images = own_rel.transform_img_fn(images)
    preds = model.predict_proba(np.array(transformed_images))
    return preds


def write_example(example_index, superpixels, label, dissim):  # currently not needed
    # write f or n
    if label == 1:
        aleph_f.write("true_class(example_" + str(example_index) + ").\n")
    else:
        aleph_n.write("true_class(example_" + str(example_index) + ").\n")

    # write bk
    # the example is declared as type example
    aleph_bk.write("example(example_" + str(example_index) + ").\n")
    #aleph_bk.write("dissim(true_class(example_" + str(example_index) + "), " + str(dissim) + ").\n")
    for sp in superpixels:  # for now, all superpixels are contained
        sp_identifier = sp.identifier
        aleph_bk.write("contains(example_" + str(example_index) + ", sp_" + str(i) + ").\n")
    # for i in range(len(superpixel_mask)):
    #	if superpixel_mask[i] == 1:
    #		aleph_bk.write("contains(example_" + str(example_index) + ", sp_" + str(i) + ").\n")


def generate_rel_ex_and_bk(image, mask, all_superpixels, important_superpixels, true_class):
    example_index = 1
    # first, write original example and relations into bk
    preds = predict_fn(np.array([image]))
    p = preds[0]
    if p[true_class] >= THRESHOLD_TRUE_CLASS:
        aleph_f.write("true_class(example_" + str(example_index) + ").\n")
        io.imsave(OUTPUTS_PATH + "perturbed/example_" + str(example_index) + "_pos.png", image)
    else:  # TODO could be better to see if threshold for counterclass is exceeded
        aleph_n.write("true_class(example_" + str(example_index) + ").\n")
        io.imsave(OUTPUTS_PATH + "perturbed/example_" + str(example_index) + "_neg.png", image)

    rels_between_imp_sps = []
    for isp in important_superpixels:  # get all relations
        # write contains for important superpixels
        aleph_bk.write("contains(" + isp.identifier + ", example_" + str(example_index) + ").\n")
        for rel in isp.relations:
            rels_between_imp_sps.append(rel)
            aleph_bk.write(rel.name + "_in_ex" + "(" + str(rel.start) + ", " +
                       str(rel.to) + ", example_" + str(example_index) + ").\n")


    for rel in rels_between_imp_sps:
        rel_name = rel.name
        rel_start = int(rel.start[3:])
        rel_to = int(rel.to[3:])
        tmp = copy.deepcopy(image)

        example_index += 1

        # write contains for important superpixels
        for isp in important_superpixels:
            aleph_bk.write("contains(" + isp.identifier + ", example_" + str(example_index) + ").\n")

        # flip superpixels
        rr1, cc1 = np.where(mask == rel_start)
        rr2, cc2 = np.where(mask == rel_to)
        values1 = tmp[rr1, cc1]
        values2 = tmp[rr2, cc2]
        tmp[rr2, cc2] = values1
        tmp[rr1, cc1] = values2
        

        preds = predict_fn(np.array([tmp]))
        p = preds[0]
        if p[true_class] >= THRESHOLD_TRUE_CLASS:
            aleph_f.write("true_class(example_" + str(example_index) + ").\n")
            io.imsave(OUTPUTS_PATH + "perturbed/example_" + str(example_index) + "_pos.png", tmp)
        else:  # TODO could be better to see if threshold for counterclass is exceeded
            aleph_n.write("true_class(example_" + str(example_index) + ").\n")
            io.imsave(OUTPUTS_PATH + "perturbed/example_" + str(example_index) + "_neg.png", tmp)

        # write changed relations into bk
        changed_relations = copy.deepcopy(rels_between_imp_sps)
        for crel in changed_relations:
            crel_name = crel.name
            crel_start = int(crel.start[3:])
            crel_to = int(crel.to[3:])
            if crel_start == rel_start:
                crel_start = rel_to
            elif crel_start == rel_to:
                crel_start = rel_start
            if crel_to == rel_start:
                crel_to = rel_to
            elif crel_to == rel_to:
                crel_to = rel_start
            
            aleph_bk.write(crel_name + "_in_ex" + "(sp_" + str(crel_start) + ", sp_" +
                       str(crel_to) + ", example_" + str(example_index) + ").\n")



"""
This function is an adaption of a function from 'https://github.com/marcotcr/lime'.
Therefore the license for it is given again here and as text file in the folder.

Copyright (c) 2016, Marco Tulio Correia Ribeiro
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
def data_labels(image, fudged_image, segments, true_class, dissims_single):
    # print segments
    n_features = np.unique(segments).shape[0]
    # print n_features
    from sklearn.utils import check_random_state
    random_state = check_random_state(None)

    data = random_state.randint(0, 2, NUM_SAMPLES_LIME_ALEPH *
                                n_features).reshape((NUM_SAMPLES_LIME_ALEPH, n_features))

    labels = []
    dissims = []
    data[0, :] = 1
    imgs = []
    num_batches = NUM_SAMPLES_LIME_ALEPH // BATCH_SIZE
    batch_counter = 1
    for row in data:
        temp = copy.deepcopy(image)
        zeros = np.where(row == 0)[0]
        mask = np.zeros(segments.shape).astype(bool)
        for z in zeros:
            mask[segments == z] = True
        temp[mask] = fudged_image[mask]
    # hier muss evtl. das Bild gespeichert werden und dann neu eingelesen und dann erst an die Liste angehängt
        #io.imsave("rule_image.png", temp)
        #temp = load_img("rule_image.png", target_size=(IMAGE_SIZE,IMAGE_SIZE))
        imgs.append(temp)
        dissims.append(dissim(row, dissims_single))
        if len(imgs) == BATCH_SIZE:
            print("Batch " + str(batch_counter) + "/" + str(num_batches))
            batch_counter += 1
            preds = predict_fn(np.array(imgs))
            print(preds)
            lblTmp = []
            for p in preds:
                # print(p[true_class]) #p ist für alle images immer gleich
                if p[true_class] >= THRESHOLD_TRUE_CLASS:
                    lblTmp.append(1)
                else:
                    lblTmp.append(0)
            labels.extend(lblTmp)
            imgs = []
    if len(imgs) > 0:
        preds = predict_fn(np.array(imgs))
        lblTmp = []
        for p in preds:
            if p[true_class] >= THRESHOLD_TRUE_CLASS:
                lblTmp.append(1)
            else:
                lblTmp.append(0)
        labels.extend(lblTmp)

    return data, np.array(labels), dissims


###################
# START OF SCRIPT #
###################
# load mask from previous step
mask = np.load("../outputs/outputs_lime/sp_mask.npy")

print(mask)
n_superpixels = np.unique(mask).shape[0]
print("Superpixels in the mask: " + str(n_superpixels))

unique_names = set()
unique_colors = set()
unique_manual_relations = set()

# get the superpixel info from the annotation
tree = parse(ANNOTATION_FILE)
root = tree.getroot()

superpixels = []
for sp in root:
    identifier = sp.attrib.get('id')
    name = sp.attrib.get('name')
    color = sp.attrib.get('color')
    lime_weight = float(sp.attrib.get('lime-weight'))
    size = sp.attrib.get('size')
    # yes, it has to be like this!! TODO change indices of occurrences in la_create... (would be better!)
    coord_x = float(sp.attrib.get('coord-y'))
    coord_y = float(sp.attrib.get('coord-x'))
    spTmp = Superpixel()
    spTmp.identifier = identifier
    spTmp.name = name
    unique_names.add(name)
    spTmp.color = color
    unique_colors.add(color)
    spTmp.lime_weight = lime_weight
    spTmp.size = int(size)
    spTmp.coord_x = coord_x
    spTmp.coord_y = coord_y

    id_as_int = int(identifier[3:])  # get rid of the 'sp_'
    spTmp.occurrences = np.where(mask == id_as_int)  # TODO correct like this?

    relations = []

    spTmp.relations = relations

    superpixels.append(spTmp)

sps_sorted_by_weight = sorted(superpixels, key=lambda x: np.abs(x.lime_weight), reverse=True)
weights = [sp.lime_weight for sp in sps_sorted_by_weight]
min_weight = np.abs(weights[N_KEEP])
print("Min weight: " + str(min_weight))
important_superpixels = [sp for sp in sps_sorted_by_weight if np.abs(sp.lime_weight) > min_weight]

print("Number of important superpixels: " + str(len(important_superpixels)))

for partner in important_superpixels:
    print("Currently at superpixel", partner.identifier)
    for reference in important_superpixels:
        if reference.identifier == partner.identifier:
            continue

        reference_x = reference.coord_x
        reference_y = reference.coord_y
        partner_x = partner.coord_x
        partner_y = partner.coord_y

        # top_of
        # if reference_y < partner_y:
        #	rel = Relation()
        #	rel.name = 'top_of'
        #	rel.to = partner.identifier
        #	reference.relations.append(rel)

        # on(x,y)
        if np.abs(partner_x-reference_x) < 2.0 and (partner_y - reference_y) > 2.0 and (partner_y - reference_y) < 6.0:
            rel = Relation()
            rel.name = 'on'
            rel.start = reference.identifier
            rel.to = partner.identifier
            reference.relations.append(rel)
        
        # under(x,y)
        if np.abs(partner_x-reference_x) < 2.0 and (reference_y - partner_y) > 2.0 and (reference_y - partner_y) < 6.0:
            rel = Relation()
            rel.name = 'under'
            rel.start = reference.identifier
            rel.to = partner.identifier
            reference.relations.append(rel)

        
        # TODO not entirely accurate
        # left_of
        if reference_x < partner_x:
            if (reference_y > (partner_y - (partner_x - reference_x))) and (reference_y < (partner_y + (partner_x - reference_x))):
            #if np.abs(partner_y-reference_y) < 2.0:
                rel = Relation()
                rel.name = 'left_of'
                rel.start = reference.identifier
                rel.to = partner.identifier
                reference.relations.append(rel)

        # right_of
        if reference_x > partner_x:
            if (reference_y < (partner_y + (reference_x - partner_x))) and (reference_y > (partner_y - (reference_x - partner_x))):
            #if np.abs(partner_y-reference_y) < 2.0:
                rel = Relation()
                rel.name = 'right_of'
                rel.start = reference.identifier
                rel.to = partner.identifier
                reference.relations.append(rel)

        
        # top_of
        if reference_y < partner_y:
            if (reference_x < (partner_x + (partner_y - reference_y))) and (reference_x > (partner_x - (partner_y - reference_y))):
            #if np.abs(partner_x-reference_x) < 2.0:
                rel = Relation()
                rel.name = 'top_of'
                rel.start = reference.identifier
                rel.to = partner.identifier
                reference.relations.append(rel)
        
        # bottom_of
        if reference_y > partner_y:
            if (reference_x > (partner_x - (reference_y - partner_y))) and (reference_x < (partner_x + (reference_y - partner_y))):
            #if np.abs(partner_x-reference_x) < 2.0:
                rel = Relation()
                rel.name = 'bottom_of'
                rel.start = reference.identifier
                rel.to = partner.identifier
                reference.relations.append(rel)

### generate aleph ###

aleph_b.write(":- use_module(library(lists)).\n")
aleph_b.write(":- modeh(1, true_class(+example)).\n")

# background rules
aleph_b.write("larger(X, Y) :- X > Y.\n")
#aleph_b.write("left_of_in_ex(SP1, SP2, Ex) :- left_of(SP1, SP2), contains(Ex, SP1), contains(Ex, SP2).\n")
#aleph_b.write("right_of_in_ex(SP1, SP2, Ex) :- right_of(SP1, SP2), contains(Ex, SP1), contains(Ex, SP2).\n")
#aleph_b.write("top_of_in_ex(SP1, SP2, Ex) :- top_of(SP1, SP2), contains(Ex, SP1), contains(Ex, SP2).\n")
#aleph_b.write("bottom_of_in_ex(SP1, SP2, Ex) :- bottom_of(SP1, SP2), contains(Ex, SP1), contains(Ex, SP2).\n")
#aleph_b.write("touches_in_ex(SP1, SP2, Ex) :- touches(SP1, SP2), contains(Ex, SP1), contains(Ex, SP2).\n")
#aleph_b.write("touches_in_ex(SP1, SP2, Ex) :- touches(SP2, SP1), contains(Ex, SP1), contains(Ex, SP2).\n")

# modes
aleph_b.write(":- modeb(*, contains(-superpixel, +example)).\n")
aleph_b.write("%:- modeb(*, contains(#superpixel, +example)).\n")

aleph_b.write(":- modeb(*, has_color(+superpixel, #color)).\n")
aleph_b.write(":- modeb(*, has_size(+superpixel, -size)).\n")
aleph_b.write(":- modeb(*, larger(+size, +size)).\n")

aleph_b.write("%:- modeb(*, has_name(+superpixel, #name)).\n")

aleph_b.write(":- modeb(*, on_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, under_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, left_of_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, left_of_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, right_of_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, right_of_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, top_of_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, top_of_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, bottom_of_in_ex(+superpixel, +superpixel, +example)).\n")
aleph_b.write(":- modeb(*, bottom_of_in_ex(+superpixel, +superpixel, +example)).\n")
# aleph_b.write(":- modeb(*, touches_in_ex(+superpixel, #superpixel, +example)).\n")
# aleph_b.write(":- modeb(*, touches_in_ex(#superpixel, +superpixel, +example)).\n")

# determinations
aleph_b.write(":- determination(true_class/1, contains/2).\n")

aleph_b.write(":- determination(true_class/1, has_color/2).\n")
aleph_b.write(":- determination(true_class/1, has_size/2).\n")
aleph_b.write(":- determination(true_class/1, larger/2).\n")

aleph_b.write("%:- determination(true_class/1, has_name/2).\n")

aleph_b.write(":- determination(true_class/1, on_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, under_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, left_of_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, left_of_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, right_of_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, right_of_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, top_of_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, top_of_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, bottom_of_in_ex/3).\n")
aleph_b.write(":- determination(true_class/1, bottom_of_in_ex/3).\n")
#aleph_b.write(":- determination(true_class/1, touches_in_ex/3).\n")
#aleph_b.write(":- determination(true_class/1, touches_in_ex/3).\n")


# manual relations # TODO no manual relations for now, but could be implemented later
# for rel in unique_manual_relations:
#	aleph_b.write(":- modeb(*, " + rel + "(+superpixel, +superpixel)).\n")
#	aleph_b.write(":- determination(true_class/1, " + rel + "/2).\n")

# write settings
aleph_b.write(":- set(i, 4).\n") #TODO was 3
aleph_b.write(":- set(clauselength, 20).\n")  # TODO change
aleph_b.write(":- set(minpos, 2).\n")
aleph_b.write(":- set(minscore, 0).\n")

aleph_b.write(":- set(verbosity, 0).\n")
aleph_b.write(":- set(noise, " + str(NOISE) + ").\n")  # maybe not needed because user defined eval fn
aleph_b.write(":- set(nodes, 10000).\n")

#aleph_b.write(":- consult(\'cost_function.pl\').\n")

aleph_b.write(":- consult(\'sp.bk\').\n")

# load in original image
image = imread("../outputs/outputs_lime/reference_image.png").astype('float32',casting='same_kind')
image /= 255.0

print(image)

# write the types
for n in unique_names:
    aleph_bk.write("name(" + str(n) + ").\n")

for c in unique_colors:
    aleph_bk.write("color(" + str(c) + ").\n")

# write non example dependent bk of the superpixels
for sp in superpixels:
    aleph_bk.write("superpixel(" + str(sp.identifier) + ").\n")
    #aleph_bk.write("has_name(" + str(sp.identifier) + ", " + sp.name + ").\n")
    aleph_bk.write("has_color(" + str(sp.identifier) + ", " + sp.color + ").\n")
    aleph_bk.write("has_size(" + str(sp.identifier) + ", " + str(sp.size) + ").\n")

    # for rel in sp.relations:
    #	aleph_bk.write(rel.name + "(" + str(sp.identifier) + ", " + rel.to + ").\n")

segmented_image = mark_boundaries(image, mask, color=(0.9, 0.9, 0.0))

true_class = int(root.attrib.get('true-class'))

dissims_single = dissim_sps(mask, image)


fudged_image = image.copy()

# for x in range(n_superpixels):
#	fudged_image[mask == x] = (
#		np.mean(image[:][:, 0]),
#		np.mean(image[:][:, 1]),
#		np.mean(image[:][:, 2]))

# for x in range(n_superpixels):
#	fudged_image[mask == x] = (0, 0, 0)


generate_rel_ex_and_bk(image, mask, superpixels, important_superpixels, true_class)


aleph_b.close()
aleph_f.close()
aleph_n.close()
aleph_bk.close()