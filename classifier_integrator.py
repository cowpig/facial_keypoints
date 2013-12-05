execfile("new_method.py")
from logreg import *

classifiers = {}

for name in KEYPOINT_DICT:
	with open("data/{}_boost.pkl", "wb") as f:
		classifiers[name] = StrongClassifier(cPickle.load(f))

def check_correct(featname, label, topleft, botright):
	row, col = KEYPOINT_DICT[featname]
	good_r = label[row]
	good_c = label[col]
	return (topleft[0] < good_r < botright[0]) and (topleft[1] < good_c < topleft[1])

scorevec = []
for i, img in enumerate(images):
	tops = cascade_scan(classifiers, img)
	found = False
	for out in tops['right_mouth_corner']:
		if check_correct('right_mouth_corner', labels[i], out[1], out[2]):
			scorevec.append(1)
			found = True
	if not found:
		scorevec.append(0)


def get_supertrainset(images, classifiers):
	superset = {}
	for img in images:
		tops = cascade_scan(classifiers, img)