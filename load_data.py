import csv, random
import numpy as np
import matplotlib.pyplot as plt


def str_to_float(string):
    if string == '':
        return None
    return float(string)

def load_data(filename):
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

def to_matrix(line):
    return np.reshape(line, (96, 96))

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

# The right/left 2-rectangle feature. 
def feature_a(m, top_left, bot_right):
    top, left = top_left
    bot, right = bot_right

    mid_l = np.floor((left+right)/2.)
    mid_r = np.ceil((left+right)/2.)

    return get_rect(m, top_left, (bot, mid_l)) - get_rect(m, (top, mid_r), bot_right)

# helper function for feature functions
def mget(m, row, col):
    if (row == -1) or (col == -1):
        return 0
    return m[row, col]


def get_rect(m, top_left, bot_right):
    top, left = top_left
    bot, right = bot_right
    # this is so that a rect from (0,0) to (1,1) is not zero.
    top -= 1
    left -= 1
    return mget(m, top, left) + mget(m, bot, right) - mget(m, top, right) - mget(m, bot, left)


def display_image(line):
    plt.imshow(to_matrix(line))
    plt.gray()
    plt.show()

def stats(labels, label_names, labels_to_check):
    if type(names_to_check[0]) != int:
        label_indices = [label_names.index(name) for name in names_to_check]

    good = []
    bad = 0
    n = len(label_indices)

    # count images that are missing the labels
    for line in labels:
        for i in label_indices:
            if line[i] == None:
                bad += 1
                break
        good.append(line)

    # get some statistics on a feature
    counts = {}
    for index in label_indices:
        counts[index] = []

    for line in good:
        for index in label_indices:
            counts[index].append(line[index])

    stats = {}
    for index in indices:
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

    # make sure the rectangle gets calculated correctly
    assert(get_rect(i_test, top_left, bot_right) == dist[0] * dist[1])
    # make sure the feature works correctly
    assert(feature_a(i_test, top_left, bot_right) == 0)

    print "pass"

if __name__ == "__main__":
    train_set, labels, label_names = load_data("data/training.csv")
    display_image(random.choice(train_set))
    indices = [
        label_names.index('left_eye_outer_corner_x'),
        label_names.index('left_eye_outer_corner_y'),
        label_names.index('left_eye_inner_corner_x'),
        label_names.index('left_eye_inner_corner_y')
    ]
    print stats(labels, indices)
    tests() 