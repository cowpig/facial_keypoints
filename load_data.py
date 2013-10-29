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
                import pdb; pdb.set_trace()

    return (train_set, labels, label_names)

def to_matrix(line):
    return np.reshape(line, (96, 96))

def display_image(line):
    plt.imshow(to_matrix(line))
    plt.gray()
    plt.show()

def stats(labels, label_indices):
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

    # get average, min, and max size for frames
    sizes = []
    for line in good:
        # TODO: hmm, maybe we need a list of (x,y) index pairs rather than just indices
        pass 

    return {
        'num_missing' : bad,
        # 'minimum_size' : min(sizes),
        # 'maximum_size' : max(sizes),
        # 'average_size' : np.average(sizes)
    }


train_set, labels, label_names = load_data("data/training.csv")
display_image(random.choice(train_set))
indices = [
    label_names.index('left_eye_outer_corner_x'),
    label_names.index('left_eye_outer_corner_y'),
    label_names.index('left_eye_inner_corner_x'),
    label_names.index('left_eye_inner_corner_y')
]
print stats(labels, indices)