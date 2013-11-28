execfile("facial_keypoints.py")

train_set, labels, label_names = load_train_set("data/training.csv")

eyes = build_eye_trainset(train_set, labels)

with open("eyes_dataset.pkl", "wb") as f:
	cPickle.dump(eyes, f)

eyeset = [(integral_matrix(eye[0]), eye[1]) for eye in eyes[:1000]]

weak_classifiers = []
features_we_need = get_all_feature_coords((18, 24))

for featname in ['a', 'b', 'c', 'd']:
	for i, feature_coord in enumerate(features_we_need[featname]):
		if i % 1000 == 0:
			print "{} weakies pumped out".format(i)

		weak = WeakClass(feature_dict[featname], *feature_coord)
		weak.train(eyeset)
		weak_classifiers.append(weak)


with open("eyes_feat_coords.pkl", "wb") as f:
	cPickle.dump(features_we_need, f)

with open("weak_classes_1ktrain.pkl", "wb") as f:
	cPickle.dump(weak_classifiers, f)

eyeset2 = [(integral_matrix(eye[0]), eye[1]) for eye in eyes[1000:2000]]
imgs, lbls = zip(*eyeset2)

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
T = 400

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
with open("classifiers.pkl", "wb") as f:
	cPickle.dump(boost_selection, f)

test_set = eyes[2000:3000]
classifier = StrongClassifier(boost_selection)

scored_set = []

for img, lbl, i in test_set:
	score = classifier.score(integral_matrix(img))
	scored_set.append(img, lbl, score, (score >= 0.5))

