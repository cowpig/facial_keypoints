execfile("new_method.py")

train_set, labels, label_names = load_train_set("data/training.csv")

full_set = zip(train_set, labels)

for k in KEYPOINT_DICT:
	dataset = create_dataset(full_set, k)
	np.random.shuffle(dataset)
	with open("data/{}_dataset.pkl".format(k), "wb") as f:
		cPickle.dump(dataset, f)