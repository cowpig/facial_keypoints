import csv, random
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from classifier import *
from scipy import ndimage
import cPickle
from scipy.misc import imresize


# literals that are specific to our dataset
bad_images = [1747, 1908]

left_eye_center      	= (0, 1)
right_eye_center     	= (2, 3)
left_eye_inner       	= (4, 5)
left_eye_outer       	= (6, 7)
right_eye_inner      	= (8, 9)
right_eye_outer      	= (10, 11)
left_eyebrow_inner   	= (12, 13)
left_eyebrow_outer   	= (14, 15)
right_eyebrow_inner  	= (16, 17)
right_eyebrow_outer  	= (18, 19)
nose_tip             	= (20, 21)
mouth_left_corner    	= (22, 23)
mouth_right_corner   	= (24, 25)
mouth_center_top_lip 	= (26, 27)
mouth_center_bottom_lip = (28, 29)

feature_dict = {
	'a' : feature_a,
	'b' : feature_b,
	'c' : feature_c,
	'd' : feature_d
}

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

		for i, line in enumerate(r):
			try:
				if i not in bad_images:
					labels.append([str_to_float(s) for s in line[:-1]])
					train_set.append([float(s) for s in line[-1].split(' ')])
			except:
				import pdb; pdb.set_trace() # loads up python debugger
			# if i > 50:
			#     break

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
	# test feaures, integral matrix
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
		assert(feature_c(i_test, top_left, bot_right) - (dist[0] * dist[1]/3.) == 1)
		assert(feature_d(i_test, top_left, bot_right) == 0)
	except:
		import pdb; pdb.set_trace


	# test Classifiers

	fakemat_l = np.mat([[1, 1, 0, 0], [1, 1, 0, 0]])
	fakemat_r = np.mat([[0, 0, 1, 1], [0, 0, 1, 1]])
	dataset = [integral_matrix(fakemat_r)] * 25 + [integral_matrix(fakemat_l)] * 175
	labels = [1] * 25 + [0] * 175
	tl = (0,0)
	br = (1,3)
	weakie = WeakClass(feature_a, tl, br)
	weakie.train(zip(dataset, labels))

	assert(weakie.evaluate(integral_matrix(fakemat_r)) == 1)
	assert(weakie.evaluate(integral_matrix(fakemat_l)) == 0)

	print "pass"

def resize(img, size):
	return ndimage.interpolation.zoom(img, size)

def get_subimage(img, top_left, bot_right):
	if len(img) == 96*96:
		img = to_matrix(img)
	top, left = top_left
	bot, right = bot_right
	return img[top:bot+1, left:right+1]

def euclidean_distance(a, b):
	if type(a) == tuple or type(a) == list:
		a = np.array(a)
	if type(b) == tuple or type(b) == list:
		b = np.array(b)
	return np.linalg.norm(a - b)

def label_distance(label, indices_a, indices_b):
	point_a = [label[indices_a[0]], label[indices_a[1]]]
	if point_a[0] == '' or point_a[1] == '' or point_a[0] == None or point_a[1] == None:
		return None
	point_b = [label[indices_b[0]], label[indices_b[1]]]
	if point_b[0] == '' or point_b[1] == '' or point_b[0] == None or point_b[1] == None:
		return None

	try:
		point_a = np.array([float(x) for x in point_a])
		point_b = np.array([float(x) for x in point_b])
	except:
		import pdb;pdb.set_trace()

	return euclidean_distance(point_a, point_b)

def flip_horizontal(matrix):
	if type(matrix) == list:
		return [row[::-1] for row in matrix]

	return matrix[...,::-1]

def build_eye_trainset(train_set, labels):
	# to_shuffle = zip(train_set, labels)
	# np.random.shuffle(to_shuffle)
	# train_set, labels = zip(*to_shuffle)

	eyes = []

	for i, label in enumerate(labels):
		dist_h_left_eye = label_distance(label, left_eye_inner, left_eye_outer)
		dist_h_right_eye = label_distance(label, right_eye_inner, right_eye_outer)

		# add each eye image with a positive label
		if dist_h_left_eye != 0 and dist_h_left_eye != None:
			left = label[4]
			right = label[6]
			middle = np.average([label[5], label[7]])

			padding = (EYE_WIDTH - (right - left))

			left = left - padding/2.
			right = right + padding/2.
			top = middle - EYE_HEIGHT/2.
			bot = middle + EYE_HEIGHT/2.

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			if (top - bot) < 24:
				bot += 1

			subimg = get_subimage(train_set[i], (top, left), (bot, right))
			tl_l = (top, left)
			br_l = (bot, right)
			eyes.append((subimg, 1, i))

		if dist_h_right_eye != 0 and dist_h_right_eye != None:
			left = label[10]
			right = label[8]
			middle = np.average([label[9], label[11]])

			padding = (EYE_WIDTH - (right - left))

			left = left - padding/2.
			right = right + padding/2.
			top = middle - EYE_HEIGHT/2.
			bot = middle + EYE_HEIGHT/2.

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			subimg = get_subimage(train_set[i], (top, left), (bot, right))
			tl_r = (top, left)
			br_r = (bot, right)
			eyes.append((flip_horizontal(subimg), 1, i))

		def random(x):
			return int(np.random.random() * x)

		def too_close(new, *others):
			for other in others:
				if euclidean_distance(new, other) < TOO_CLOSE_VALUE:
					return True
			return False

		for _ in xrange(2):
			tl = (random(96 - EYE_HEIGHT), random(96 - EYE_WIDTH))
			br = (tl[0] + EYE_HEIGHT, tl[1] + EYE_WIDTH)

			while too_close(tl, tl_l, tl_r) or too_close(br, br_l, br_r):
				tl = (random(96 - EYE_HEIGHT), random(96 - EYE_WIDTH))
				br = (tl[0] + EYE_HEIGHT, tl[1] + EYE_WIDTH)

			eyes.append((get_subimage(train_set[i], tl, br), 0))

	return eyes

def build_mouth_trainset(train_set, labels):
	to_shuffle = zip(train_set, labels)
	np.random.shuffle(to_shuffle)
	train_set, labels = zip(*to_shuffle)

	mouth_left_corner = (22, 23)
	mouth_right_corner = (24, 25)
	mouth_center_top_lip = (26, 27)
	mouth_center_bottom_lip = (28, 29)

	mouths = []
	distances = []

	for i, label in enumerate(labels):
		dist_h = label_distance(label, mouth_left_corner, mouth_right_corner)
		dist_v = label_distance(label, mouth_center_top_lip, mouth_center_bottom_lip)

		# add each eye image with a positive label
		if dist_h != 0 and dist_h != None and dist_v != 0 and dist_v != None and dist_h < 40:
			left = label[24]
			right = label[22]
			top = label[27]
			bot = label[29]

			padding_h = (MOUTH_WIDTH - (right - left))
			padding_v = (MOUTH_HEIGHT - (bot - top))

			# import pdb; pdb.set_trace()

			left = left - padding_h/2.
			right = right + padding_h/2.
			top = top - padding_v/2.
			bot = bot + padding_v/2.

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			while (bot > 96):
				top -= 1
				bot -= 1

			subimg = get_subimage(train_set[i], (top, left), (bot, right))
			tl_m = (top, left)
			br_m = (bot, right)

			mouths.append((subimg, 1, i))

			def random(x):
				return int(np.random.random() * x)

			def too_close(new, *others):
				for other in others:
					if euclidean_distance(new, other) < TOO_CLOSE_VALUE:
						return True
				return False

			tl = (random(96 - MOUTH_HEIGHT), random(96 - MOUTH_WIDTH))
			br = (tl[0] + MOUTH_HEIGHT, tl[1] + MOUTH_WIDTH)

			while too_close(tl, tl_m, (0,0)) or too_close(br, br_m, (0,0)):
				tl = (random(96 - MOUTH_HEIGHT), random(96 - MOUTH_WIDTH))
				br = (tl[0] + MOUTH_HEIGHT, tl[1] + MOUTH_WIDTH)

			mouths.append((get_subimage(train_set[i], tl, br), 0))

	return mouths

def cross_sample(mouths, eyes):
	# the eyes sample is several times larger than the mouth
	num_mouths = len(mouths) * 0.5
	num_eyes = len(eyes) * 0.5

	eyeshape = np.shape(eyes[0][0])
	mouthshape = np.shape(mouths[0][0])

	for i in xrange(1, int(num_eyes*0.1), 2):
		try:
			eyes[i] = (imresize(mouths[i+1][0], eyeshape), 0)
		except Exception as e:
			print e.message
			import pdb; pdb.set_trace()

	for i in xrange(1, int(num_mouths*0.1), 2):
		mouths[i] = (imresize(eyes[i+1][0], mouthshape), 0)

	np.random.shuffle(mouths)
	np.random.shuffle(eyes)



def build_eye_classifier(train_set, labels):
	features_we_need = get_all_feature_coords((EYE_HEIGHT, EYE_WIDTH))

	classifiers = []

	for i, feature_coord in enumerate(features_we_need['d']):
		if i % 1000 == 0:
			print "{} weakies pumped out".format(i)

		weak = WeakClass(feature_d, *feature_coord)
		weak.train(eyeset)
		classifiers.append(weak)

	return classifiers

def eye_scores(img, strongclas):
	if  len(img) == 96*96:
		img = to_matrix(img)

	top = 0
	left = 0
	bot = EYE_HEIGHT
	right = EYE_WIDTH

	height, width = np.shape(img)

	precisions = [10, 50, 400]

	scores = {'left' : {}, 'right': {}}
	for side in scores:
		for p in precisions:
			scores[side][p] = []

	while True:
		while True:
			try:
				frame = get_subimage(img, (top, left), (bot, right))
				flipframe = flip_horizontal(frame)
				for p in precisions:
					score_l = strongclas.score(integral_matrix(frame), p)
					score_r = strongclas.score(integral_matrix(flipframe), p)

					scores['left'][p].append(((top, left), (bot, right), score_l))
					scores['right'][p].append(((top, left), (bot, right), score_r))
			except Exception as e:
				import pdb; pdb.set_trace()

			left += np.ceil(TOO_CLOSE_VALUE/2.)
			right += np.ceil(TOO_CLOSE_VALUE/2.)
			if right > width:
				# print "{}r > {}w".format(right, width)
				break
		top += np.ceil(TOO_CLOSE_VALUE/2.)
		bot += np.ceil(TOO_CLOSE_VALUE/2.)
		if bot > height:
			# print "{}b > {}h".format(bot, height)
			break
		left = 0
		right = EYE_WIDTH

	print "image scoring complete"
	return scores

def cascade(img, strongclas):
	if len(img) == 96*96:
		img = to_matrix(img)

	img = integral_matrix(img)

	top = 0
	left = 0
	bot = EYE_HEIGHT
	right = EYE_WIDTH

	height, width = np.shape(img)

	potential_r = []
	potential_l = []

	while True:
		while True:
			frame = get_subimage(img, (top, left), (bot, right))
			flipframe = flip_horizontal(frame)
			# print np.shape(frame)
			# print np.shape(flipframe)
			score_l = strongclas.score(frame, 10)
			score_r = strongclas.score(flipframe, 10)
			# print score_l
			# print score_r
			
			potential_l.append((score_l, (top, left), (bot, right)))

			potential_r.append((score_r, (top, left), (bot, right)))

			left += np.ceil(TOO_CLOSE_VALUE/3.)
			right += np.ceil(TOO_CLOSE_VALUE/3.)
			if right > width:
				break
		top += np.ceil(TOO_CLOSE_VALUE/3.)
		bot += np.ceil(TOO_CLOSE_VALUE/3.)
		if bot > height:
			break

		left = 0
		right = EYE_WIDTH

	potential_l = sorted(potential_l, key=lambda x: -x[0])[:50]
	potential_l = [(strongclas.score(get_subimage(img, tl, br), 200), tl, br) for score, tl, br in potential_l]
	potential_l = sorted(potential_l, key=lambda x: -x[0])[:10]


	potential_r = sorted(potential_r, key=lambda x: -x[0])[:50]
	potential_r = [(strongclas.score(get_subimage(img, tl, br), 200), tl, br) for score, tl, br in potential_r]
	potential_r = sorted(potential_r, key=lambda x: -x[0])[:10]


	# potential_l = [(strongclas.score(get_subimage(img, tl, br), 400), tl, br) for score, tl, br in potential_l]
	# potential_r = [(strongclas.score(get_subimage(img, tl, br), 400), tl, br) for score, tl, br in potential_r]

	# left = potential_l[0]
	# right = potential_r[0]

	# return ((left[1], left[2]), (right[1], right[2]), left[0] + right[0])
	pairs = []

	# import pdb; pdb.set_trace()

	for score, tl_l, br_l in potential_l:
		for score, tl_r, br_r in potential_r:
			if (tl_l[1] > tl_r[1]) and (euclidean_distance(tl_l, tl_r) > TOO_CLOSE_VALUE):
				frame_l = get_subimage(img, tl_l, br_l)
				frame_r = get_subimage(img, tl_r, br_r)
				pairs.append(((tl_l, br_l, frame_l), (tl_r, br_r, frame_r)))
			else:
				# import pdb; pdb.set_trace()
				pass

	print len(pairs)

	maximum = (None, None, 0.0)
	for pair in pairs:
		# import pdb; pdb.set_trace()
		pair_score = strongclas.score(pair[0][2]) + strongclas.score(pair[1][2])
		if pair_score > maximum[2]:
			maximum = (pair[0][:2], pair[1][:2], pair_score)

	return maximum

	# max_dist_horizontal = 0
	# max_dist_vertical = 0
	# for i, label in enumerate(labels):
	# 	dist_h_left_eye = label_distance(label, left_eye_inner, left_eye_outer)
	# 	dist_h_right_eye = label_distance(label, right_eye_inner, right_eye_outer)

	# 	if dist_h_right_eye < 24 and dist_h_left_eye > 24: 
	# 		# distances.append(dist_h_right_eye)
	# 		# distances.append(dist_h_)


	# 	# dist_h_right_brow = label_distance(label, right_eyebrow_inner, right_eyebrow_outer)
	# 	# dist_h_left_brow = label_distance(label, left_eyebrow_inner, left_eyebrow_outer)
	# 	if dist_h_left_eye > 22 or dist_h_right_eye > 22:
	# 		print "{}, {}, {}".format(i, dist_h_left_eye, dist_h_right_eye)

	# 	max_dist_horizontal = max(max_dist_horizontal, 
	# 								# dist_h_right_brow, 
	# 								dist_h_right_eye, 
	# 								# dist_h_left_brow, 
	# 								dist_h_left_eye)

	# 	dist_v_left_inner = label_distance(label, left_eye_inner, left_eyebrow_inner)
	# 	dist_v_left_outer = label_distance(label, left_eye_outer, left_eyebrow_outer)
	# 	dist_v_right_inner = label_distance(label, right_eye_inner, right_eyebrow_outer)
	# 	dist_v_right_outer = label_distance(label, right_eye_outer, right_eyebrow_outer)

	# 	max_dist_vertical = max(max_dist_vertical, 
	# 								dist_v_left_inner, 
	# 								dist_v_left_outer, 
	# 								dist_v_right_inner, 
	# 								dist_v_right_outer)

	# coords = get_all_feature_coords((max_dist_vertical, max_dist_horizontal))

	# for feature_type in coords:
	# 	for coord in coords[feature_type]:
	# 		pass

# if __name__ == "__main__":
# 	train_set, labels, label_names = load_train_set("data/training.csv")
# 	class1 = weakclass(feature_a, (20,20), (40,40))
# 	myset = []
# 	mylabels = []
# 	for i in range(50):
# 		mylabels.append(i % 2)

# 	for i in range(50):
# 		myset.append((to_matrix(train_set[i]), mylabels[i]))

# 	class1.train(myset)

# 	print class1.threshold
# 	print class1.evaluate(myset[15][0])

# 	display_image(random.choice(tr,in_set))
# 	indices = [
# 		'left_eye_outer_corner_x',
# 		'left_eye_outer_corner_y',
# 		'left_eye_inner_corner_x',
# 		'left_eye_inner_corner_y'
# 	]
# 	import pprint
# 	pprint.pprint(stats(labels, label_names, indices))
# 	tests() 

# if __name__ == "__main__":
# 	train_set, labels, label_names = load_train_set("data/training.csv")
# 	weakies = build_eye_classifier(train_set, labels)
# 	boooosted = boost_em_up(weakies, passable_eyes, 100)