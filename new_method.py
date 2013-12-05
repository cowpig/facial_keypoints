execfile("facial_keypoints.py")

BOX_RADIUS = 8
BOX_DIAM = 2 * BOX_RADIUS + 1
FEATCOORD_DICT = get_all_feature_coords((BOX_DIAM, BOX_DIAM))

def box_around_coords(center, boxsize):
	row, col = center
	if type(boxsize) == tuple:
		d_row, d_col = boxsize
	else:
		d_row = boxsize
		d_col = boxsize
	top_left = (row - d_row, col - d_col)
	bot_right = (row + d_row, col + d_col)

	for coord in top_left:
		if coord < 0:
			# print "Warning: box goes < zero"
			return None
		if coord > 95:
			# print "Warning: box goes > 95"
			return None

	for coord in bot_right:
		if coord < 0:
			# print "Warning: box goes < zero"
			return None
		if coord > 95:
			# print "Warning: box goes > 95"
			return None

	return (top_left, bot_right)

def jiggle_within_bounds(center, min_jig, max_jig):
	c_row, c_col = center

	if c_row - max_jig < 0 + BOX_RADIUS + 1:
		c_row += np.random.randint(min_jig, max_jig)
	elif c_row + max_jig > 95 - BOX_RADIUS - 1:
		c_row -= np.random.randint(min_jig, max_jig)
	else:
		if np.random.randint(2):
			c_row += np.random.randint(min_jig, max_jig)
		else:
			c_row -= np.random.randint(min_jig, max_jig)

	if c_col - max_jig < 0 + BOX_RADIUS + 1:
		c_col += np.random.randint(min_jig, max_jig)
	elif c_col + max_jig > 95 - BOX_RADIUS - 1:
		c_col -= np.random.randint(min_jig, max_jig)
	else:
		if np.random.randint(2):
			c_col += np.random.randint(min_jig, max_jig)
		else:
			c_col -= np.random.randint(min_jig, max_jig)

	return (c_row, c_col)

def create_dataset(trainset, keypoint_name):
	i, j = KEYPOINT_DICT[keypoint_name]
	output = []
	
	nose_idxs = set([(21,20)])
	mouth_idxs = set([(23,22),(25,24),(27,26),(29,28)])
	eye_idxs = set(KEYPOINT_DICT.values()) - nose_idxs - mouth_idxs

	if "eye" in keypoint_name:
		other_idxs = list(mouth_idxs.union(nose_idxs))
	elif "mouth" in keypoint_name:
		other_idxs = list(eye_idxs.union(nose_idxs))
	else:
		other_idxs = list(mouth_idxs.union(eye_idxs))

	it = -1
	for img, lbl in trainset:
		it += 1
		if lbl[i] == None or lbl[j] == None:
			print "{} : skipping {} (unlabeled)".format(keypoint_name, it)
			continue

		r = lbl[i]
		c = lbl[j]

		box = box_around_coords((r,c), BOX_RADIUS)

		if box == None:
			print "{} : skipping {} (bad box - center at [{}, {}])".format(keypoint_name, it, r, c)
			continue

		pos = get_subimage(img, *box)

		rand = np.random.randint(3)
		if rand == 0:
			# get something nearby as a negative example
			center_nearby = jiggle_within_bounds((r, c), BOX_RADIUS, BOX_RADIUS*2)
			negbox = box_around_coords(center_nearby, BOX_RADIUS)
			if negbox == None:
				raise Exception("Bad box!")
			neg = get_subimage(img, *negbox)
		if rand == 1:
			# get a different keypoint at random
			done = False
			tries = 0
			while not done:
				k, l = random.choice(other_idxs)
				if lbl[k] == None or lbl[l] == None:
					continue
				negbox = box_around_coords((lbl[k], lbl[l]), BOX_RADIUS)
				if negbox != None:
					neg = get_subimage(img, *negbox)
					done = True

				tries += 1
				if tries > 10:
					print "Warning: out of tries, going for random."
					rand = 2
					done = True

		if rand == 2:
			# totally random
			done = False
			while not done:
				r2 = np.random.randint(1+BOX_RADIUS, 95-BOX_RADIUS)
				c2 = np.random.randint(1+BOX_RADIUS, 95-BOX_RADIUS)

				if np.abs(r-r2) > BOX_RADIUS or np.abs(c-c2) > BOX_RADIUS:
					negbox = box_around_coords((r2, c2), BOX_RADIUS)
					neg = get_subimage(img, *negbox)
					done = True

		# print "appending neg and pos for {}".format(it)
		output.append((pos, 1))
		output.append((neg, 0))

	return output

def get_weak_classifiers(trainset, keypoint_name):
	weak_classifiers = []
	for featname in ['a', 'b', 'c', 'd']:
		for i, feature_coord in enumerate(FEATCOORD_DICT[featname]):
			if i % 1000 == 0:
				print "{} - {} weakies pumped out ({})".format(featname, i, keypoint_name)

			weak = WeakClass(FEATURE_DICT[featname], *feature_coord)
			weak.train(trainset)
			weak_classifiers.append(weak)

	return weak_classifiers

def boost(trainset, keypoint_name, weak_classifiers, T):
	imgs, lbls = zip(*trainset)

	perceptrons = set(weak_classifiers)

	lbls = np.array([int(label) for label in lbls])

	integrals = []
	for image in imgs:
		integrals.append(integral_matrix(image))

	positives = sum(lbls)
	n = len(imgs)
	negatives = n - positives

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

	boost_selection = []

	for t in xrange(T):
		print "getting classifier {} for {}".format(t, keypoint_name)
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

if __name__ == "__main__":
	from sys import argv
	if len(argv) != 2:
		print "give me a keypoint plz"
	else:
		keypoint_name = argv[1]
		with open("data/{}_dataset.pkl".format(keypoint_name), 'rb') as f:
			fullset = cPickle.load(f)

		train = fullset[:int(len(fullset)*0.05)]

		imgs, lbls = zip(*train)
		iimgs = [integral_matrix(img) for img in imgs]

		trainset = zip(iimgs, lbls)

		weakies = get_weak_classifiers(train, keypoint_name)
		boosted = boost(train, keypoint_name, weakies, 500)

		with open("data/{}_boost.pkl", "wb") as f:
			cPickle.dump(boosted, f)

		test = fullset[int(len(fullset)*0.05):]

		imgs, lbls = zip(*test)
		iimgs = [integral_matrix(img) for img in imgs]

		classifier = StrongClassifier(boosted)

		results = np.array([StrongClassifier.evaluate(iimg) for iimg in imgs])
		lbls = np.array(lbls)

		score = sum(np.logical_not(np.abs(results - lbls))) / float(len(lbls))

		print "SCORE FOR {}: {}".format(keypoint_name, score)