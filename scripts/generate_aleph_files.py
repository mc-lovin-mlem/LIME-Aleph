from skimage import io
import numpy as np
import sys
from train_model import own_rel
from skimage.io import imshow, show, imsave, imread
import copy
import os
import shutil

IMAGE_SIZE = own_rel.IMAGE_SIZE









aleph_b = open(OUTPUTS_PATH + 'aleph_files/sp.b', 'w')
aleph_f = open(OUTPUTS_PATH + 'aleph_files/sp.f', 'w')
aleph_n = open(OUTPUTS_PATH + 'aleph_files/sp.n', 'w')
aleph_bk = open(OUTPUTS_PATH + 'aleph_files/sp.bk', 'w')



###################
# START OF SCRIPT #
###################

unique_names = set()
unique_colors = set()





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

# write settings
aleph_b.write(":- set(i, 4).\n") #TODO was 3
aleph_b.write(":- set(clauselength, 20).\n")  # TODO change
aleph_b.write(":- set(minpos, 2).\n")
aleph_b.write(":- set(minscore, 0).\n")

aleph_b.write(":- set(verbosity, 0).\n")
aleph_b.write(":- set(noise, " + str(NOISE) + ").\n")  # maybe not needed because user defined eval fn
aleph_b.write(":- set(nodes, 10000).\n")

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


generate_rel_ex_and_bk(image, mask, important_superpixels, true_class)


aleph_b.close()
aleph_f.close()
aleph_n.close()
aleph_bk.close()