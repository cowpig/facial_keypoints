import csv, random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from classifier import *
from scipy import ndimage

def str_to_float(string):
    if string == '':
        return None
    return float(string)

# loads the training set and processes it. Outputs two 2d arrays: 
#   the raw image data, and the facial feature coordinates. the third
#   array it outputs is the list of the names of the facial features
#   (e.g. 'left_eye_outer_corner_x') I generally call the values 'labels'
#   and that refers to the number, not the string
def load_train_set(filename):
    train_set = []
    labels = []

    with open(filename, 'rb') as f:
        r = csv.reader(f)
        label_names = r.next()[:-1]

        for line in r:
            try:
                labels.append([str_to_float(s) for s in line[:-1]])
                train_set.append([float(s) for s in line[-1].split(' ')])
            except:
                import pdb; pdb.set_trace() # loads up python debugger

    return (train_set, labels, label_names)

# takes a line containing raw image data, and reshapes it into a 96 row,
#   96-column matrix
def to_matrix(line):
    assert(len(line) == 96 * 96)
    return np.reshape(line, (96, 96))

# takes a matrix of pixel densities and outputs the integral image (as described
#   in the viola-jones paper)
def integral_matrix(m):
    l, w = m.shape
    out = np.zeros((l,w))
    for i in xrange(l):
        for j in xrange(w):
            left = 0 if i == 0 else out[i-1,j]
            up = 0 if j == 0 else out[i,j-1]
            if i==0 or j==0:
                up_and_left = 0
            else:
                up_and_left = out[i-1, j-1]
            
            out[i,j] = m[i,j] + left + up - up_and_left

    return out 

# takes an integral image matrix, and the top-left and bottom-right points,
#   and returns the sum of the sub-image's pixels
def get_rect(m, top_left, bot_right):
    top, left = top_left
    bot, right = bot_right
    # this is so that a rect from (0,0) to (1,1) is not zero.
    top -= 1
    left -= 1
    return mget(m, top, left) + mget(m, bot, right) - mget(m, top, right) - mget(m, bot, left)

# helper function for feature functions
def mget(m, row, col):
    if (row == -1) or (col == -1):
        return 0
    return m[row, col]
    
# takes an image and displays it
def display_image(img):
    if len(img) == 96*96:
        plt.imshow(to_matrix(img))
    else:
        plt.imshow(img)
    plt.gray()
    plt.show()

# calculates the some stats about a set (or list) of label(name)s
def stats(labels, label_names, labels_to_check):
    # can take either the names of features or their indices
    if type(labels_to_check[0]) != int:
        label_indices = [label_names.index(name) for name in labels_to_check]
    else:
        label_indices = labels_to_check

    good = []
    bad = 0
    n = len(label_indices)

    # count images that are missing the labels
    for line in labels:
        good_line = True
        for i in label_indices:
            if line[i] == None:
                bad += 1
                good_line = False
                break
        if good_line:
            good.append(line)

    # get some statistics on a feature
    counts = {}
    for index in label_indices:
        counts[index] = []

    for line in good:
        for index in label_indices:
            counts[index].append(line[index])

    stats = {}
    for index in label_indices:
        name = label_names[index]
        stats[name] = {}
        stats[name]["avg"] = sum(counts[index]) / float(len(counts[index]))
        stats[name]["min"] = min(counts[index])
        stats[name]["max"] = max(counts[index])
        

    return {
        'num_missing' : bad,
        'individual_stats' : stats
    }

def tests():
    test = np.ones((8,8))
    i_test = integral_matrix(test)

    assert(i_test[-1,-1] == 64)

    top_left = (random.choice(range(4)), random.choice(range(4)))
    bot_right = (random.choice(range(4)) + 4, random.choice(range(4)) + 4)
    dist = (bot_right[0] - top_left[0], bot_right[1] - top_left[1])

    try:
        # make sure the rectangle gets calculated correctly
        assert(get_rect(i_test, top_left, bot_right) == dist[0] * dist[1])
        # make sure the features work correctly
        assert(feature_a(i_test, top_left, bot_right) == 0)
        assert(feature_b(i_test, top_left, bot_right) == 0)
        assert(feature_c(i_test, top_left, bot_right) - (dist[0] * dist[1]/3.) <= 1)
        assert(feature_d(i_test, top_left, bot_right) == 0)
    except:
        import pdb; pdb.set_trace

    print "pass"

def resize(img, size):
    return ndimage.interpolation.zoom(img, size)

def get_subimage(img, top_left, bot_right):
    top, left = top_left
    bot_right = bot_right
    return img[top:bot, left:right]



# if __name__ == "__main__":
#     train_set, labels, label_names = load_train_set("data/training.csv")
#     display_image(random.choice(train_set))
#     indices = [
#         'left_eye_outer_corner_x',
#         'left_eye_outer_corner_y',
#         'left_eye_inner_corner_x',
#         'left_eye_inner_corner_y'
#     ]
#     import pprint
#     pprint.pprint(stats(labels, label_names, indices))
#     tests() 
