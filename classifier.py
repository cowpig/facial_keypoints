import numpy as np

class weakclass(object):
	def __init__(self, ftype, top_left, bot_right):
		self.ftype = ftype
		self.top_left = top_left
		self.bot_right = bot_right
		self.threshhold = None
		self.parity = None

	def evaluate(self, img):
		if self.parity * self.ftype(img, self.top_left, self.bot_right) < self.parity * self.threshhold:
			return 1
		else:
			return 0

	def perbefore(self, ar, stop):
		positive = 0

		for i in xrange(stop + 1):
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
	def train(self, imgs):
		result = []
		for pic in imgs:
			result.append((self.ftype(pic[0], self.top_left, self.bot_right), pic[1]))


		result = sorted(result)

		ratios = []

		for i, r  in enumerate(result):
			ratios.append((self.perbefore(result, i), self.perafter(result, i)))


		maxratioindex = 0
		maxratio = np.abs(ratios[0][1] - ratios[0][0])
		for i, r in enumerate(ratios):
			current = np.abs(ratios[i][1]- ratios[i][0])
			if current > maxratio:
				maxratioindex = i
				maxratio = current

		if ratios[maxratioindex][0] > ratios[maxratioindex][1]:
			self.threshhold = result[maxratioindex + 1][0]
			self.parity = 1
		else:
			self.threshhold = result[maxratioindex][0]
			self.parity = -1

	def perbefore(ar, stop):
		positive = 0

		for i in xrange(stop):
			if ar[i][1] == 1:
				positive += 1

		return positive / (stop + 1.) *100

	def perafter(ar, stop):
		positive = 0

		for i in xrange((len(ar) - 1) - stop):
			if ar[i + stop][1] == 1:
				positive += 1

		return positive / ((len(ar) - 1.) - stop) * 100


# takes a set/list of weak classifiers and applies boosting to them, as described in the
#	Viola-Jones paper (page 8)
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
