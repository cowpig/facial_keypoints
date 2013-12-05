execfile("new_method.py")

train_set, labels, label_names = load_train_set("data/training.csv")

full_set = zip(train_set, labels)

for k in KEYPOINT_DICT:
	dataset = create_dataset(train_set, k)
	with open("data/{}_dataset.pkl", "wb") as f:
		cPickle.dump(f, dataset)