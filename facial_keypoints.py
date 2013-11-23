import csv, random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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

# The right/left 2-rectangle feature. 
def feature_a(m, top_left, bot_right):
    top, left = top_left
    bot, right = bot_right

    mid_l = np.floor((left+right)/2.)
    mid_r = np.ceil((left+right)/2.)

    return get_rect(m, top_left, (bot, mid_l)) - get_rect(m, (top, mid_r), bot_right)

# The top/bottom 2-rectangle feature. 
def feature_b(m, top_left, bot_right):
    top, left = top_left
    bot_right = bot_right

    mid_t = np.floor((top+bot)/2.)
    mid_b = np.ceil((top+bot)/2.)

    return get_rect(m, top_left, (mid_t, right)) - get_rect(m, (mid_b, left), bot_right)

# The 3-rectangle feature. 
def feature_c(m, top_left, bot_right):
    top, left = top_left
    bot_right = bot_right

    midleft_l = np.floor((left+right)/3.)
    midleft_r = np.ceil((left+right)/3.)
    midright_l = np.floor(2.*(left+right)/2.)
    midright_r = np.ceil((2.*left+right)/2.)

    left_rect = get_rect(m, top_left, (bot, midleft_l))
    mid_rect = get_rect(m, (top, midleft_r), (bot, midright_l))
    right_rect = get_rect(m, (top, midright_r), bot_right)

    return left_rect + right_rect - mid_rect

# The 4-rectangle feature. 
def feature_d(m, top_left, bot_right):
    top, left = top_left
    bot_right = bot_right

    mid_t = np.floor((top+bot)/2.)
    mid_b = np.ceil((top+bot)/2.)
    mid_l = np.floor((left+right)/2.)
    mid_r = np.ceil((left+right)/2.)

    rect_tl = get_rect(m, top_left, (mid_t, mid_l))
    rect_bl = get_rect(m, (mid_b, left), (bot, mid_l))
    rect_tr = get_rect(m, (top, mid_r), (mid_t, left))
    rect_br = get_rect(m, (mid_b, mid_r), bot_right)

    return rect_tl + rect_br - rect_tr - rect_bl
    
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

if __name__ == "__main__":
    train_set, labels, label_names = load_train_set("data/training.csv")
    display_image(random.choice(train_set))
    indices = [
        'left_eye_outer_corner_x',
        'left_eye_outer_corner_y',
        'left_eye_inner_corner_x',
        'left_eye_inner_corner_y'
    ]
    import pprint
    pprint.pprint(stats(labels, label_names, indices))
    tests() 

class Perceptron(Object):
    def __init__(self, ftype, top_left, bot_right):
        self.w = random.random()
        self.ftype = ftype
        self.top_left = top_left
        self.bot_right = bot_right

    def evaluate(img, answer):
        result = self.w * self.ftype(img, self.top_left, self.bot_right)
        if result > threshhold:
            evaluate = 1
        else:
            evaluate = 0
        if evaluate == answer:
            self.w += .05
        else:
            self.w -= .05

def boost_em_up(perceptrons, images, labels, T):
    perceptrons = deepcopy(perceptrons)

    labels = np.array([int(label) for label in labels])

    integrals = []
    for image in images:
        integrals.append(integral_matrix(image))

    positives = sum(labels)
    n = len(images)
    negatives = n - positives

    evals = {}
    for perceptron in perceptrons:
        evals[perceptron] = np.abs(labels - np.array([perceptron.eval(iimg) for iimg in integrals])) 

    weights = []
    for l in labels:
        if l:
            weights.append(1/(2*positives))
        else:
            weights.append(1/(2*negatives))

    weights = np.array(weights)

    boost_selection = []

    for t in xrange(T):
        best_percep = (perceptrons[0], np.dot(evals[perceptrons[0]] * weights))
        for perceptron in perceptrons[1:]:
            error = np.dot(evals[perceptron] * weights)
            if error < best_percep[1]:
                best_percep = (perceptron, error)

        percep, err = best_percep
        perceptrons.remove(percep)

        beta = (err/(1-err))
        weights = weights * beta ** 1 - evals[percep]
        
        boost_selection.append(percep, np.log(1/beta))

    return boost_selection