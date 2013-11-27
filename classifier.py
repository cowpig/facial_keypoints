import numpy as np
from copy import deepcopy

class WeakClass(object):
	def __init__(self, ftype, top_left, bot_right):
		self.ftype = ftype
		self.top_left = top_left
		self.bot_right = bot_right
		self.threshold = None
		self.parity = None

	def evaluate(self, img):
		if self.parity * self.ftype(img, self.top_left, self.bot_right) < self.parity * self.threshold:
			return 1
		else:
			return 0

	def perbefore(self, ar, stop):
		positive = 0

		for i in xrange(stop):
			if ar[i][1] == 1:
				positive += 1

		return positive / (stop + 1.) *100

	def perafter(self, ar, start):
		positive = 0

		for i in xrange(start, len(ar)):
			if ar[i][1] == 1:
				positive += 1
		try:
			return float(positive) / (len(ar) - start) * 100
		except:
			import pdb; pdb.set_trace()
	
	#train the weak classifier      
	def train(self, imgs):
		result = []
		for pic in imgs:
			try:
				result.append((self.ftype(pic[0], self.top_left, self.bot_right), pic[1]))
			except Exception as e:
				import pdb;pdb.set_trace()


		result = sorted(result)

		summ = 0
		for i in result:
			summ += i[0]

		self.threshold = summ / len(result)


		for i, r in enumerate(result):
			if r[0] > self.threshold:
				threshindex = i
				break


		before = self.perbefore(result, threshindex)
		after = self.perafter(result, threshindex)

		if before > after:
			self.parity = 1
		else:
			self.parity = -1 

		# print result
		# print self.parity
		# print "average = " + str(self.threshold)
		# print "before = " + str(before)
		# print "after = " + str(after)



# takes a set/list of weak classifiers and applies boosting to them, as described in the
#   Viola-Jones paper (page 8)
def boost_em_up(weak_classifiers, trainset, T):
	imgs, lbls = zip(*eyeset2)

	perceptrons = set(weak_classifiers)

	lbls = np.array([int(label) for label in lbls])

	integrals = []
	for image in imgs:
		integrals.append(integral_matrix(image))

	positives = sum(lbls)
	n = len(imgs)
	negatives = n - positives

	# evals is a bit of a misnomer here
	#	it's really a vector of which examples the classifier got right
	evals = {}
	for perceptron in perceptrons:
		evals[perceptron] = np.abs(lbls - np.array([perceptron.evaluate(iimg) for iimg in integrals])) 

	weights = []
	for l in lbls:
		if l:
			weights.append(1./(2*positives))
		else:
			weights.append(1./(2*negatives))

	weights = np.array(weights)
	perceptrons = set(weak_classifiers)

	boost_selection = []
	# T = 20

	for t in xrange(T):
		print "getting classifier {}".format(t)
		weights = weights / sum(weights)
		best_percep = (None, 999999999999999999999999.9)
		for perceptron in perceptrons:
			error = np.dot(evals[perceptron], weights)
			if error < best_percep[1]:
				best_percep = (perceptron, error)

		percep, err = best_percep
		perceptrons.remove(percep)

		beta = (err/(1-err))
		weights = weights * beta ** (1 - evals[percep])
		boost_selection.append((percep, np.log(1./beta)))


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
	bot, right = bot_right

	mid_t = np.floor((top+bot)/2.)
	mid_b = np.ceil((top+bot)/2.)

	return get_rect(m, top_left, (mid_t, right)) - get_rect(m, (mid_b, left), bot_right)

# The 3-rectangle feature. 
def feature_c(m, top_left, bot_right):
	top, left = top_left
	bot, right = bot_right

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
	bot, right = bot_right

	mid_t = np.floor((top+bot)/2.)
	mid_b = np.ceil((top+bot)/2.)
	mid_l = np.floor((left+right)/2.)
	mid_r = np.ceil((left+right)/2.)

	rect_tl = get_rect(m, top_left, (mid_t, mid_l))
	rect_bl = get_rect(m, (mid_b, left), (bot, mid_l))
	rect_tr = get_rect(m, (top, mid_r), (mid_t, left))
	rect_br = get_rect(m, (mid_b, mid_r), bot_right)

	return rect_tl + rect_br - rect_tr - rect_bl

def get_all_feature_coords(matrix_size):
	feat_coords = {}

	feat_coords['a'] = set()
	min_size_a = (1,2)
	
	feat_coords['b'] = set()
	min_size_b = (2,1)
	
	feat_coords['c'] = set()
	min_size_c = (1,3)
	
	feat_coords['d'] = set()
	min_size_d = (2,2)

	for top in xrange(matrix_size[0]):
		for left in xrange(matrix_size[1]):
			for bot in xrange(top, matrix_size[0]):
				for right in xrange(left, matrix_size[1]):
					row_dist = bot - top + 1
					col_dist = right - left + 1

					if (min_size_a[0] <= row_dist 
								and min_size_a[1] <= col_dist
								and row_dist % min_size_a[0] == 0
								and col_dist % min_size_a[1] == 0):
						feat_coords['a'].add(((top, left), (bot, right)))

					if (min_size_b[0] <= row_dist 
								and min_size_b[1] <= col_dist
								and row_dist % min_size_b[0] == 0
								and col_dist % min_size_b[1] == 0):
						feat_coords['b'].add(((top, left), (bot, right)))

					if (min_size_c[0] <= row_dist 
								and min_size_c[1] <= col_dist
								and row_dist % min_size_c[0] == 0
								and col_dist % min_size_c[1] == 0):
						feat_coords['c'].add(((top, left), (bot, right)))

					if (min_size_d[0] <= row_dist 
								and min_size_d[1] <= col_dist
								and row_dist % min_size_d[0] == 0
								and col_dist % min_size_d[1] == 0):
						feat_coords['d'].add(((top, left), (bot, right)))

	return feat_coords

def feature_number(matrix_size):
	fc = get_all_feature_coords(matrix_size)
	return sum([len(fc[key]) for key in fc])


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
