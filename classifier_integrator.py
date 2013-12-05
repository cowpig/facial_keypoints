execfile("new_method.py")

classifiers = {}

for name in KEYPOINT_DICT:
	with open("data/{}_boost.pkl", "wb") as f:
		classifiers[name] = StrongClassifier(cPickle.load(f))

