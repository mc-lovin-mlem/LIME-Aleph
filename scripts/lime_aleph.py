import numpy as np
import webcolors
from lime import lime_image
from sources.own_segmentation.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import time
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from skimage import io
from skimage.io import imshow, show, imsave
import copy

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

class Image:
    pass

class Superpixel:
    pass

class Relation:
    pass

# predicted_class
# image
# superpixels
# relations
class PerturbedExample:
    pass

def annotate_image_parts(image, model, output_directory, n_lime_samples=1000):

    ### Run LIME to get the importance of each superpixel ###
    print("Running LIME...")

    true_class = -1
    predictions = model.predict_proba(np.array([image])) #Probabilities
    predictions = np.squeeze(predictions) #to get rid of the third dimension
    true_class = np.argmax(predictions)
    print("True class of the image is: ", true_class)

    # the softmax output for the two classes:
    print("Negative estimator:", predictions[0])
    print("Positive estimator:", predictions[1])

    annotated_image = Image()
    annotated_image.true_class = true_class

    # save the original image in the Image instance
    annotated_image.original_image = image

    explainer = lime_image.LimeImageExplainer(verbose=True)
    print('Starting the explanation generation process. This may take several minutes. Seriously. Grab a cup of coffee.')
    sf = SegmentationAlgorithm('quadratic', n_quads=8) # our own segmentation algorithm. A uniform grid.
    tmp = time.time()
    explanation = explainer.explain_instance(image, model.predict_proba, segmentation_fn=sf, top_labels=2, num_samples=n_lime_samples, hide_color=0)
    print("Elapsed time: " + str(time.time() - tmp))

    lime_output, lime_mask = explanation.get_image_and_mask(true_class, positive_only=True, hide_rest=True)
    io.imsave(output_directory + "lime_output.png", mark_boundaries(lime_output, lime_mask))

    # Beware: Only applicable to binary tasks
    counter_class = -1
    if true_class == 0:
        counter_class = 1
    else:
        counter_class = 0

    lime_output_counter, lime_mask_counter = explanation.get_image_and_mask(counter_class, positive_only=True, hide_rest=True)
    io.imsave(output_directory + "lime_output_counter.png", mark_boundaries(lime_output_counter, lime_mask_counter))

    top_class_weights = explanation.local_exp[true_class]
    segments = explanation.segments
    annotated_image.superpixel_mask = segments

    print("Number of superpixels: " + str(len(top_class_weights)))


    ### Annotate the superpixels ###
    print("Annotating the superpixels...")

    superpixels = []
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
        x = (np.max(occurrences[1]) + np.min(occurrences[1]))/2.0
        y = (np.max(occurrences[0]) + np.min(occurrences[0]))/2.0
        
        # add a superpixel with all the information to the list of superpixels
        sp = Superpixel()
        sp.id = f
        sp.name = "sp_" + str(f)
        sp.color = str(closest_name)
        sp.size = sp_size_in_pixels
        sp.lime_weight = w
        sp.x_coord = x
        sp.y_coord = y
        superpixels.append(sp)
    
    annotated_image.superpixels = superpixels

    # save sp integer mask
    np.save(output_directory + "sp_mask.npy", segments)

    return annotated_image

def find_important_parts(annotated_image, k=3):
    
    # find the important superpixels according to the user-given parameter
    sps_sorted_by_weight = sorted(annotated_image.superpixels, key=lambda x: np.abs(x.lime_weight), reverse=True)
    weights = [sp.lime_weight for sp in sps_sorted_by_weight]
    min_weight = np.abs(weights[k])
    print("Min weight: " + str(min_weight))
    important_superpixels = [sp for sp in sps_sorted_by_weight if np.abs(sp.lime_weight) > min_weight]

    return important_superpixels

def find_spatial_relations(important_superpixels):

    ### Play through all combinations to find spatial relations ###

    relations = []

    for partner in important_superpixels:
        print("Currently at superpixel", partner.id)
        for reference in important_superpixels:
            if reference.id == partner.id:
                continue

            reference_x = reference.x_coord
            reference_y = reference.y_coord
            partner_x = partner.x_coord
            partner_y = partner.y_coord

            # on(x,y)
            if np.abs(partner_x-reference_x) < 2.0 and (partner_y - reference_y) > 2.0 and (partner_y - reference_y) < 6.0:
                rel = Relation()
                rel.name = 'on'
                rel.start = reference.id
                rel.to = partner.id
                relations.append(rel)
            
            # under(x,y)
            if np.abs(partner_x-reference_x) < 2.0 and (reference_y - partner_y) > 2.0 and (reference_y - partner_y) < 6.0:
                rel = Relation()
                rel.name = 'under'
                rel.start = reference.id
                rel.to = partner.id
                relations.append(rel)

            
            # left_of
            if reference_x < partner_x:
                if (reference_y > (partner_y - (partner_x - reference_x))) and (reference_y < (partner_y + (partner_x - reference_x))):
                #if np.abs(partner_y-reference_y) < 2.0:
                    rel = Relation()
                    rel.name = 'left_of'
                    rel.start = reference.id
                    rel.to = partner.id
                    relations.append(rel)

            # right_of
            if reference_x > partner_x:
                if (reference_y < (partner_y + (reference_x - partner_x))) and (reference_y > (partner_y - (reference_x - partner_x))):
                #if np.abs(partner_y-reference_y) < 2.0:
                    rel = Relation()
                    rel.name = 'right_of'
                    rel.start = reference.id
                    rel.to = partner.id
                    relations.append(rel)

            
            # top_of
            if reference_y < partner_y:
                if (reference_x < (partner_x + (partner_y - reference_y))) and (reference_x > (partner_x - (partner_y - reference_y))):
                #if np.abs(partner_x-reference_x) < 2.0:
                    rel = Relation()
                    rel.name = 'top_of'
                    rel.start = reference.id
                    rel.to = partner.id
                    relations.append(rel)
            
            # bottom_of
            if reference_y > partner_y:
                if (reference_x > (partner_x - (reference_y - partner_y))) and (reference_x < (partner_x + (reference_y - partner_y))):
                #if np.abs(partner_x-reference_x) < 2.0:
                    rel = Relation()
                    rel.name = 'bottom_of'
                    rel.start = reference.id
                    rel.to = partner.id
                    relations.append(rel)
    
    return relations

def get_imp_sps_from_relations(relations):

    imp_sps = []

    for rel in relations:
        start = rel.start
        to = rel.to
        if not start in imp_sps:
            imp_sps.append(start)
        if not to in imp_sps:
            imp_sps.append(to)

    
    return imp_sps

def perturb_instance(annotated_image, relations, model, threshold_true_class):

    true_class = annotated_image.true_class
    original_image = annotated_image.original_image
    mask = annotated_image.superpixel_mask
    important_superpixels = get_imp_sps_from_relations(relations)

    preds = model.predict_proba(np.array([original_image]))
    p = preds[0]

    perturbed_dataset = []

    # the original image is always part of the perturbation dataset    
    ex = PerturbedExample()
    ex.positive = p[true_class] >= threshold_true_class
    ex.superpixels = important_superpixels
    ex.relations = relations
    perturbed_dataset.append(ex)

    for rel in relations:
        rel_name = rel.name
        rel_start = rel.start
        rel_to = rel.to
        tmp_image = copy.deepcopy(original_image)

        ex = PerturbedExample()
        ex.superpixels = important_superpixels

        # flip superpixels
        rr1, cc1 = np.where(mask == rel_start)
        rr2, cc2 = np.where(mask == rel_to)
        values1 = tmp_image[rr1, cc1]
        values2 = tmp_image[rr2, cc2]
        tmp_image[rr2, cc2] = values1
        tmp_image[rr1, cc1] = values2

        preds = model.predict_proba(np.array([tmp_image]))
        p = preds[0]
        ex.positive = p[true_class] >= threshold_true_class

        # write changed relations into bk
        changed_relations = copy.deepcopy(relations)
        new_relations = []
        for crel in changed_relations:
            crel_name = crel.name
            crel_start = crel.start
            crel_to = crel.to
            if crel_start == rel_start:
                crel_start = rel_to
            elif crel_start == rel_to:
                crel_start = rel_start
            if crel_to == rel_start:
                crel_to = rel_to
            elif crel_to == rel_to:
                crel_to = rel_start
            
            new_rel = Relation()
            new_rel.name = crel_name
            new_rel.start = crel_start
            new_rel.to = crel_to
            new_relations.append(new_rel)
        ex.relations = new_relations

        perturbed_dataset.append(ex)
    
    return perturbed_dataset