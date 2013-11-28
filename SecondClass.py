import numpy as np

class SecondClass(object):

	def __init__(self):
		None

	def train_all(self, train_set, labels):

		self.train_eye_inner(train_set, labels)
		self.train_eye_center(train_set, labels)
		self.train_eye_outer(train_set, labels)
		self.train_eyebrow_outer(train_set, labels)
		self.train_eyebrow_inner(train_set, labels)
		self.train_nose_tip(train_set, labels)
		self.train_mouth_right_corner(train_set, labels)
		self.train_mouth_left_corner(train_set, labels)
		self.train_mouth_center_top(train_set, labels)
		self.train_mouth_center_bottom(train_set, labels)

	def train_eye_inner(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][4])
			y = np.round(labels[i][5])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)

		self.eye_inner = np.average(result, axis = 0)


	def train_eye_center(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][0])
			y = np.round(labels[i][1])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.eye_center = np.average(result, axis = 0)

	def train_eye_outer(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][6])
			y = np.round(labels[i][7])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.eye_outer = np.average(result, axis = 0)

	def train_eyebrow_outer(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][14])
			y = np.round(labels[i][15])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.eyebrow_outer = np.average(result, axis = 0)

	def train_eyebrow_inner(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][12])
			y = np.round(labels[i][13])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.eyebrow_inner = np.average(result, axis = 0)

	def train_nose_tip(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][20])
			y = np.round(labels[i][21])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.nose_tip = np.average(result, axis = 0)

	def train_mouth_right_corner(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][24])
			y = np.round(labels[i][25])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.mouth_right_corner = np.average(result, axis = 0)

	def train_mouth_left_corner(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][22])
			y = np.round(labels[i][23])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.mouth_left_corner = np.average(result, axis = 0)

	def train_mouth_center_top(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][26])
			y = np.round(labels[i][27])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.mouth_cener_top = np.average(result, axis = 0)

	def train_mouth_center_bottom(self, train_set, labels):

		result = []
		for i, pic in enumerate(train_set):
			x = np.round(labels[i][28])
			y = np.round(labels[i][29])

			r = self.get_eight_neighbor(pic, x, y)
			if r == None:
				continue
			else:
				result.append(r)
		self.mouth_center_bottom = np.average(result, axis = 0)

	def find_eye_inner(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.eye_inner)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.eye_inner)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_eye_center(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.eye_center)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.eye_center)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_eye_outer(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.eye_outer)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.eye_outer)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_eyebrow_outer(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.eyebrow_outer)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.eyebrow_outer)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_eyebrow_inner(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.eyebrow_inner)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.eyebrow_inner)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_nose_tip(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.nose_tip)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.nose_tip)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_mouth_right_corner(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.mouth_right_corner)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.mouth_right_corner)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_mouth_left_corner(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.mouth_left_corner)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.mouth_left_corner)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_mouth_center_top(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.mouth_cener_top)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.mouth_cener_top)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def find_mouth_center_bottom(self, img):

		min_dist_coords = (1,1)
		min_dist = np.linalg.norm(self.get_eight_neighbor(img, 1, 1) - self.mouth_center_bottom)
		for i in range(1, len(img) - 1):
			for j in range(1, len(img[i]) - 1):
				new_dist = np.linalg.norm(self.get_eight_neighbor(img, j, i) - self.mouth_center_bottom)
				if new_dist < min_dist:
					min_dist = new_dist
					min_dist_coords = (j,i)

		return min_dist_coords

	def get_eight_neighbor(self, img, x, y):
		if y-1 >= 0 and y+1 < len(img) and x-1 >= 0 and x+1 < len(img[y]):
			return (img[y + 1][x - 1], img[y + 1][x], img[y + 1][x + 1], img[y][x - 1], img[y][x], img[y][x + 1], img[y - 1][x - 1], img[y + 1][x], img[y - 1][x + 1])
		else:
			return None