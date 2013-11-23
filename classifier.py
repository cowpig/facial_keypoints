from facial_keypoints import feature_a, feature_b, feature_c, feature_d

class weakclass(object):
	
	def __init__(self, ftype, top_left, bot_right):
		self.ftype = ftype
		self.top_left = top_left
		self.bot_right = bot_right
		self.threshhold = None
		self.parity = None

	def evaluate(img):
		if parity * ftype(img, top_left, bot_right) < parity * threshhold:
			return 1
		else:
			return 0

	def train(imgs):
		result = []
		for pic in imgs:
			result.append((ftype(pic[0], top_left, bot_right)), pic[1])

		result = sorted(result)

		ratios = []

		for i, r  in enumerate(result):
			ratios.append((perbefore(result, i), perafter(result, i)))

		maxratioindex = 0
		maxratio = math.fabs(ratios[0][1] - ratios[0][0])
		for i, r in enumerate(ratios):
			current = math.fabs(ratios[i][1]- ratios[i][0])
			if current > maxratio:
				maxratioindex = i
				maxratio = current

		self.threshhold = result[maxratio][0]

		if ratios[maxratio][0] < ratios[maxratio][1]:
			parity = 1
		else:
			parity = -1



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

