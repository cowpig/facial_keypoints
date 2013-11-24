import csv, random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from classifier import *
from scipy import ndimage

bad_images = [1747, 1908]

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
	
# takes an image and displays it
def display_image(img):
	if len(img) == 96*96:
		plt.imshow(to_matrix(img))
	else:
		plt.imshow(img)
	plt.gray()
	plt.show()


# takes an image and displays it
def save_image(img, fn):
	if len(img) == 96*96:
		plt.imshow(to_matrix(img))
	else:
		plt.imshow(img)
	plt.gray()
	plt.savefig(fn)

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
	if len(img) == 96*96:
		img = to_matrix(img)
	top, left = top_left
	bot, right = bot_right
	return img[top:bot+1, left:right+1]

def label_distance(label, indices_a, indices_b):
	point_a = [label[indices_a[0]], label[indices_a[1]]]
	if point_a[0] == '' or point_a[1] == '' or point_a[0] == None or point_a[1] == None:
		return 0
	point_b = [label[indices_b[0]], label[indices_b[1]]]
	if point_b[0] == '' or point_b[1] == '' or point_b[0] == None or point_b[1] == None:
		return 0

	try:
		point_a = np.array([float(x) for x in point_a])
		point_b = np.array([float(x) for x in point_b])
	except:
		import pdb;pdb.set_trace()

	return np.linalg.norm(point_a - point_b)

def build_eye_classifier(train_set, labels):
	max_dist_horizontal = 0
	max_dist_vertical = 0

	left_eye_inner = (4, 5)
	left_eye_outer = (6, 7)
	right_eye_inner = (8, 9)
	right_eye_outer = (10, 11)
	left_eyebrow_inner = (12, 13)
	left_eyebrow_outer = (14, 15)
	right_eyebrow_inner = (16, 17)
	right_eyebrow_outer = (18, 19)


	distances = []
	for i, label in enumerate(labels):
		dist_h_left_eye = label_distance(label, left_eye_inner, left_eye_outer)
		dist_h_right_eye = label_distance(label, right_eye_inner, right_eye_outer)

		if dist_h_left_eye != 0:
			left = label[4]
			right = label[6]
			# try:
			middle = np.average([label[5], label[7]])
			# except:
				# import pdb;pdb.set_trace()

			padding = (24 - (right - left))

			left = left - padding
			right = right + padding
			top = middle - 12
			bot = middle + 12

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			# try:
			subimg = get_subimage(train_set[i], (top, left), (bot, right))

			save_image(subimg, "images/{}_left_eye.png".format(i))
			# except:
			# 	import pdb;pdb.set_trace()

		if dist_h_right_eye != 0:
			left = label[10]
			right = label[8]
			middle = np.average([label[9], label[11]])

			padding = (24 - (right - left))

			left = left - padding
			right = right + padding
			top = middle - 12
			bot = middle + 12

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			subimg = get_subimage(train_set[i], (top, left), (bot, right))
			save_image(subimg, "images/{}_right_eye.png".format(i))

		# if dist_h_right_eye > 
		# distances.append(dist_h_right_eye)
		# distances.append(dist_h_)




		# dist_h_right_brow = label_distance(label, right_eyebrow_inner, right_eyebrow_outer)
		# dist_h_left_brow = label_distance(label, left_eyebrow_inner, left_eyebrow_outer)
		if dist_h_left_eye > 22 or dist_h_right_eye > 22:
			print "{}, {}, {}".format(i, dist_h_left_eye, dist_h_right_eye)

		max_dist_horizontal = max(max_dist_horizontal, 
									# dist_h_right_brow, 
									dist_h_right_eye, 
									# dist_h_left_brow, 
									dist_h_left_eye)

		dist_v_left_inner = label_distance(label, left_eye_inner, left_eyebrow_inner)
		dist_v_left_outer = label_distance(label, left_eye_outer, left_eyebrow_outer)
		dist_v_right_inner = label_distance(label, right_eye_inner, right_eyebrow_outer)
		dist_v_right_outer = label_distance(label, right_eye_outer, right_eyebrow_outer)

		max_dist_vertical = max(max_dist_vertical, 
									dist_v_left_inner, 
									dist_v_left_outer, 
									dist_v_right_inner, 
									dist_v_right_outer)

	coords = get_all_feature_coords((max_dist_vertical, max_dist_horizontal))

	for feature_type in coords:
		for coord in coords[feature_type]:
			pass

		

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
