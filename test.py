execfile("facial_keypoints.py")

train_set, labels, label_names = load_train_set("data/training.csv")

mouths = build_mouth_trainset(train_set, labels)
eyes = build_eye_trainset(train_set, labels)

def sanitize(set):
	sanitized = False
	while not sanitized:
		for i, x in enumerate(set):
			s = np.shape(x[0])
			if s[0] == 0 or s[1] == 0:
				set.pop(i)
				break

			if i == len(set) - 1:
				sanitized = True 

sanitize(mouths)
sanitize(eyes)

cross_sample(mouths, eyes)

mouthtrain_cutoff = int(len(mouths) * 0.7)
eyetrain_cutoff = int(len(eyes) * 0.7)

## EYES

with open("eye_dataset.pkl", "wb") as f:
	cPickle.dump(eyes, f)

weak_classifiers = []
features_we_need = get_all_feature_coords((EYE_HEIGHT, EYE_WIDTH))

eyeset = eyes[:eyetrain_cutoff]

for featname in ['a', 'b', 'c', 'd']:
	for i, feature_coord in enumerate(features_we_need[featname]):
		if i % 1000 == 0:
			print "{} weakies pumped out (eyes)".format(i)

		weak = WeakClass(FEATURE_DICT[featname], *feature_coord)
		weak.train(eyeset)
		weak_classifiers.append(weak)

with open("weaktrain_eyes.pkl", "wb") as f:
	cPickle.dump(weak_classifiers, f)

# reload (damn)

execfile("facial_keypoints.py")

with open("weaktrain_eyes.pkl", "rb") as f:
	weak_classifiers = cPickle.load(f)

with open("eye_dataset.pkl", "rb") as f:
	eyes = cPickle.load(f)

eyetrain_cutoff = int(len(eyes) * 0.7)
eyeset = eyes[:eyetrain_cutoff]

imgs, lbls = zip(*eyeset)

perceptrons = set(weak_classifiers)

lbls = np.array([int(label) for label in lbls])

positives = sum(lbls)
n = len(imgs)
negatives = n - positives

# evals is a bit of a misnomer here
#	it's really a vector of which examples the classifier got right
evals = {}
for perceptron in perceptrons:
	evals[perceptron] = np.abs(lbls - np.array([perceptron.evaluate(img) for img in imgs])) 

weights = []
for l in lbls:
	if l:
		weights.append(1./(2*positives))
	else:
		weights.append(1./(2*negatives))

weights = np.array(weights)
perceptrons = set(weak_classifiers)

boost_selection = []
T = 500

for t in xrange(T):
	if t % 10 == 0:
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

import cPickle
with open("boosted_eyes.pkl", "wb") as f:
	cPickle.dump(boost_selection, f)


## MOUTH

with open("mouth_dataset.pkl", "wb") as f:
	cPickle.dump(mouths, f)

weak_classifiers = []
features_we_need = get_all_feature_coords((MOUTH_HEIGHT, MOUTH_WIDTH))

mouthset = mouths[:mouthtrain_cutoff]

for featname in ['a', 'b', 'c', 'd']:
	for i, feature_coord in enumerate(features_we_need[featname]):
		if i % 1000 == 0:
			print "{} weakies pumped out (mouths)".format(i)

		weak = WeakClass(FEATURE_DICT[featname], *feature_coord)
		weak.train(mouthset)
		weak_classifiers.append(weak)

with open("weaktrain_mouths.pkl", "wb") as f:
	cPickle.dump(weak_classifiers, f)

# reload (damn)

execfile("facial_keypoints.py")

with open("weaktrain_mouths.pkl", "rb") as f:
	weak_classifiers = cPickle.load(f)

with open("mouth_dataset.pkl", "rb") as f:
	mouths = cPickle.load(f)

mouthtrain_cutoff = int(len(mouths) * 0.7)
mouthset = mouths[:mouthtrain_cutoff]

imgs, lbls = zip(*mouthset)

perceptrons = set(weak_classifiers)

lbls = np.array([int(label) for label in lbls])

positives = sum(lbls)
n = len(imgs)
negatives = n - positives

# evals is a bit of a misnomer here
#	it's really a vector of which examples the classifier got right
evals = {}
for perceptron in perceptrons:
	evals[perceptron] = np.abs(lbls - np.array([perceptron.evaluate(img) for img in imgs])) 

weights = []
for l in lbls:
	if l:
		weights.append(1./(2*positives))
	else:
		weights.append(1./(2*negatives))

weights = np.array(weights)
perceptrons = set(weak_classifiers)

boost_selection = []
T = 500

for t in xrange(T):
	if t % 10 == 0:
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

import cPickle
with open("boosted_mouths.pkl", "wb") as f:
	cPickle.dump(boost_selection, f)


######
# Testing


import cPickle
execfile("facial_keypoints.py")

with open("mouth_dataset.pkl", "rb") as f:
	mouths = cPickle.load(f)

with open("eye_dataset.pkl", "rb") as f:
	eyes = cPickle.load(f)

with open("boosted_mouths.pkl", "rb") as f:
	boosted_mouths = cPickle.load(f)

with open("boosted_eyes.pkl", "rb") as f:
	boosted_eyes = cPickle.load(f)

mouthtrain_cutoff = int(len(mouths) * 0.7)
eyetrain_cutoff = int(len(eyes) * 0.7)

test_m_imgs, test_m_lbls = zip(*mouths[mouthtrain_cutoff:])
test_e_imgs, test_e_lbls = zip(*eyes[eyetrain_cutoff:])

strong_m = StrongClassifier(boosted_mouths)
strong_e = StrongClassifier(boosted_eyes)

m_results = [strong_m.evaluate(img) for img in test_m_imgs]
e_results = []
for img in test_e_imgs:
	try:
		e_results.append(strong_e.evaluate(img))
	except Exception as e:
		print e.message
		e_results.append(0)

m_score = sum(np.logical_not(np.abs(np.array(m_results) - np.array(test_m_lbls)))) / float(len(test_m_lbls))
e_score = sum(np.logical_not(np.abs(np.array(e_results) - np.array(test_e_lbls)))) / float(len(test_e_lbls))


# with open("eyes_feat_coords.pkl", "wb") as f:
# 	cPickle.dump(features_we_need, f)

# with open("weak_classes_1ktrain.pkl", "wb") as f:
# 	cPickle.dump(weak_classifiers, f)

# with open("weak_classes_1ktrain.pkl", "rb") as f:
# 	weak_classifiers = cPickle.load(f)

# with open("eyes_dataset.pkl", "rb") as f:
# 	eyes = cPickle.load(f)

# eyeset2 = []
# for eye in eyes[1000:]:
# 	if np.shape(eye[0]) == (19,25):
# 		eyeset2.append((eye[0], eye[1]))
# 	if len(eyeset2) == 1000:
# 		break

# # eyeset2 = [(integral_matrix(eye[0]), eye[1]) for eye in eyes[1000:2000]]
# imgs, lbls = zip(*eyeset2)

# perceptrons = set(weak_classifiers)

# lbls = np.array([int(label) for label in lbls])

# positives = sum(lbls)
# n = len(imgs)
# negatives = n - positives

# # evals is a bit of a misnomer here
# #	it's really a vector of which examples the classifier got right
# evals = {}
# for perceptron in perceptrons:
# 	evals[perceptron] = np.abs(lbls - np.array([perceptron.evaluate(img) for img in imgs])) 

# weights = []
# for l in lbls:
# 	if l:
# 		weights.append(1./(2*positives))
# 	else:
# 		weights.append(1./(2*negatives))

# weights = np.array(weights)
# perceptrons = set(weak_classifiers)

# boost_selection = []
# T = 500

# for t in xrange(T):
# 	if t % 10 == 0:
# 		print "getting classifier {}".format(t)
# 	weights = weights / sum(weights)
# 	best_percep = (None, 999999999999999999999999.9)
# 	for perceptron in perceptrons:
# 		error = np.dot(evals[perceptron], weights)
# 		if error < best_percep[1]:
# 			best_percep = (perceptron, error)

# 	percep, err = best_percep
# 	perceptrons.remove(percep)

# 	beta = (err/(1-err))
# 	weights = weights * beta ** (1 - evals[percep])
# 	boost_selection.append((percep, np.log(1./beta)))

# import cPickle
# with open("classifiers.pkl", "wb") as f:
# 	cPickle.dump(boost_selection, f)



# test_set = eyes[2000:3000]
# classifier = StrongClassifier(boost_selection)

# scored_set = []

# for img, lbl, i in test_set:
# 	score = classifier.score(integral_matrix(img))
# 	scored_set.append((img, lbl, score, (score >= 0.5)))


# ############################################
# with open("eyes_feat_coords.pkl", "rb") as f:
# 	features_we_need = cPickle.load(f)

# with open("weak_classes_1ktrain.pkl", "rb") as f:
# 	weak_classifiers = cPickle.load(f)

# with open("classifiers.pkl", "wb") as f:
# 	boost_selection = cPickle.dump(f)


# execfile("facial_keypoints.py")

# # test_scores = [eye_scores(to_matrix(img), classifier) for img, lbl in test_imgs]

# ############################################
# # PRESENT
# execfile("facial_keypoints.py")

# train_set, labels, label_names = load_train_set("data/training.csv")

# train_w_labels = []
# for img, lbl in zip(train_set, labels):
# 	good = True
# 	for i in lbl[:12]:
# 		if i == None:
# 			good = False

# 	if good:
# 		train_w_labels.append((img, lbl))

# test_imgs = random.sample(train_w_labels, 20)

# with open("classifiers.pkl", "rb") as f:
# 	boost_selection = cPickle.load(f)

# classifier = StrongClassifier(boost_selection)

# show = []
# for i in [1,2,3,7,8,9]:
# 	show.append(test_imgs[i])

# with open("show.pkl", "wb") as f:
#     cPickle.dump(show, f)

# with open("show.pkl", "rb") as f:
#     show = cPickle.load(f)

# def show_img(img):
# 	img, lbl = img
# 	eyes = cascade(img, classifier)
# 	print "img {}: {}".format(i, eyes)
# 	display_image(img)
# 	display_image(get_subimage(img, eyes[1][0], eyes[1][1]))
# 	display_image(get_subimage(img, eyes[0][0], eyes[0][1]))

# ############################################




# # eyes = cascade(test_imgs[1][0], classifier)

# for clas in weak_classifiers[:2]:
