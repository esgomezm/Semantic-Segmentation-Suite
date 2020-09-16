import cv2
import numpy as np
import itertools
import operator
import os, csv
import tensorflow as tf
from scipy import ndimage
from skimage import morphology

import time, datetime

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!
    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            if len(row)==4:
                class_names.append(row[0])
                label_values.append([int(row[1]), int(row[2]), int(row[3])])
            else:
                class_names.append(row[0])
                label_values.append([int(row[1])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        if len(label.shape) == 3:
            class_map = np.all(equality, axis = -1)
        else:
            class_map = equality.astype(np.int8)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# class_dict = get_class_dict("CamVid/class_dict.csv")
# gt = cv2.imread("CamVid/test_labels/0001TP_007170_L.png",-1)
# gt = reverse_one_hot(one_hot_it(gt, class_dict))
# gt = colour_code_segmentation(gt, class_dict)

# file_name = "gt_test.png"
# cv2.imwrite(file_name,np.uint8(gt))
    
def foreground_binarize (image, thereshold):
    upper, lower = 1, 0
    fore_bin= np.where(image>thereshold, upper, lower)
    return fore_bin



def remove_small (image, min_size):
#    The min_size is the area of pixels at which the data is going to be dismissed
    blobs_labels,nlabels = ndimage.measurements.label(image)
#    Compute the properties of the region (uncomment the next one)
#    properties = regionprops(blobs_labels)   
    clean_image = morphology.remove_small_objects(blobs_labels,min_size, connectivity=2)
    clean_image[clean_image>0.5]=1
    return clean_image

def remove_edge_seg(img):
    img=1-img
    blobs_labels,_ = ndimage.measurements.label(img)
    labels2remove =[]
    edges = np.copy(blobs_labels)
    edges[1:-1, 1:-1] =0
    edges[edges>0.5]=1
    edge_labels = np.unique(np.multiply(edges, blobs_labels))
    if np.sum(edge_labels) > 0:
        for i in edge_labels[1:]: 
            labels2remove = blobs_labels == int(i) # Esta parte no sé si python funcionará, creo que sí. Si no, haces un loop sobre los valores edge_labels[1:]
            blobs_labels = np.multiply(1-labels2remove, blobs_labels)
        blobs_labels[blobs_labels>1]=1
        img=np.copy(blobs_labels)
    img=1-img
    return img

