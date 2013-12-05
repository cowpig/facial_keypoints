execfile("facial_keypoints.py")

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
			print "Warning: box goes < zero"
		if coord > 95:
			print "Warning: box goes > 95"

		return None

	for coord in bot_right:
		if coord < 0:
			print "Warning: box goes < zero"
		if coord > 95:
			print "Warning: box goes > 95"

		return None

	return (top_left, bot_right)

def create_dataset(trainset, keypoint_name):
	row, col = keypoints[keypoint_name]
	output = []
	for img, lbl in trainset:
		if lbl[row] == None or lbl[col] == None:
			continue

		r = lbl[row]
		c = lbl[col]

		box = box_around_coords((r,c), 8)

		if box == None:
			continue

		pos = get_subimage(img, *box)

		# TODO produce a negative
		neg = None

		output.append((pos, 1))
		output.append((neg, 0))

	return output