__author__ = 'Andrei Eliseev'





# Necessary imports
import perspective_control.verbosity
from skimage import color, filters, measure
import numpy as np





def hough_kb(image):
	"""
	Hough transform in (k, b) parameter space
	
	Parameters
	==========
	image : NumPy (N x M) array
	        Unichomic (single-channel) image (width = M, height = N).
	
	Return
	======
	hough_h : NumPy ((2M + 1) x (N + 2M)) array
	          Hough space of predominantly horizontal lines.
	          Axis 0 represents values of k that change uniformly from -1 to 1.
	          Axis 1 represents values of b that change uniformly from -M to N + M.
	hough_v : NumPy ((2N + 1) x (2N + M)) array
	          Hough space of predominantly horizontal lines.
	          Axis 0 represents values of k that change uniformly from -1 to 1.
	          Axis 1 represents values of b that change uniformly from -N to N + M.
	"""
	# 1. Initialise horizontal Hough space
	hough_h = np.zeros((2 * image.shape[1] + 1, image.shape[0] + 2 * image.shape[1]), dtype='int')
	k_h = np.arange(hough_h.shape[0])
	k_true_h = np.linspace(-1.0, 1.0, hough_h.shape[0])

	# 2. Initialise vertical Hough space
	hough_v = np.zeros((2 * image.shape[0] + 1, 2 * image.shape[0] + image.shape[1]), dtype='int')
	k_v = np.arange(hough_v.shape[0])
	k_true_v = np.linspace(-1.0, 1.0, hough_v.shape[0])

	# 3. Saturate Hough spaces
	for point in np.array(np.where(image != 0)).T:
		b = (point[0] - k_true_h * point[1] + image.shape[1]).astype('int')
		hough_h[k_h, b] += image[point[0]][point[1]]
		b = (point[1] - k_true_v * point[0] + image.shape[0]).astype('int')
		hough_v[k_v, b] += image[point[0]][point[1]]
	
	return hough_h, hough_v





def hough_kb_peaks(hough_space):
	"""
	Identification of bundles in Hough space
	
	Nullifies all values that are less than max - s * std such that
	more than 99.99% of values are nullified.
	
	Parameters
	==========
	hough_space : NumPy (N x M) array
	              A Hough space.
	
	Returns
	=======
	hough_peaks : NumPy (N x M) array
	              A thresholded Hough space.
	"""
	# 1. We don't want to spoil the original Hough space
	hough_copy = hough_space.copy()

	# 2. Get rid of non-peak values
	softness = 1.6
	while hough_copy[hough_copy == 0].size / hough_copy.size < 0.9999:
		softness -= 0.1
		hough_copy[hough_space < np.max(hough_space) - softness * np.std(hough_space)] = 0
	while hough_copy[hough_copy == 0].size / hough_copy.size > 0.99999:
		softness += 0.01
		hough_copy[hough_space >= np.max(hough_space) - softness * np.std(hough_space)] = hough_space[hough_space >= np.max(hough_space) - softness * np.std(hough_space)]
	
	return hough_copy





def clusterise(image):
	"""
	Replace bundles with single pixels
	
	Finds connected regions in binarised copy of image and replaces
	each region with a single pixel that has cumulative value of all
	pixels from the corresponding region and is located in a centre
	of a bounding box of the region.
	
	Parameters
	==========
	image : NumPy (N x M) array
	        Unichomic (single-channel) image (width = M, height = N).
	
	Return
	======
	clusters : NumPy (N x M) array
	           A copy of image with all clusters replaced with a single
	           pixel.
	"""
	answer = np.zeros(image.shape)

	# 1. Create binary mask of non-zero pixels
	image_bin = image.copy()
	image_bin[image_bin != 0] = 1

	# 2. Form a list of connected regions
	labels, labels_count = measure.label(image_bin, background=0, return_num=True)
	clusters = tuple(np.where(labels == label) for label in range(1, labels_count + 1))
	
	# 3. Replace each cluster with a single representative
	for cluster in clusters:
		cluster_centre = (np.mean(cluster[0]).astype('int'), np.mean(cluster[1]).astype('int'))
		answer[cluster_centre[0]][cluster_centre[1]] = np.sum(image[cluster])
	
	return answer.astype('int')





def get_vanishing_points(image, verbosity):
	"""
	Calculate coordinates of vanishing points
	
	Calculates coordinates of dominant horizontal and dominant vertical
	vanishing points using the original image, its Hough transform and
	double Hough transform applied to each of the original Hough spaces.
	
	Parameters
	==========
	image     : NumPy (N x M x ...) array
	            Original image.
	verbosity : bool
	            Toggles verbosity.
	
	Return
	======
	(x_h, y_h) : tuple of int
	             Coordinates (in image coordinate system) of dominant
	             horizontal vanishing point.
	(x_v, y_v) : tuple of int
	             Coordinates (in image coordinate system) of dominant
	             vertical vanishing point.
	"""
	verbosity_label_capacity = 51
	# 1. Identify edges with 3 x 3 Scharr operator and Gaussian smoothing,
	#    binarise
	perspective_control.verbosity.start_progress(verbosity, 'Retrieving vanishing points', verbosity_label_capacity)
	edges = filters.scharr(filters.gaussian(color.rgb2gray(image), sigma=1))
	edges[np.where(edges > filters.threshold_otsu(edges))] = 1
	edges[np.where(edges != 1)] = 0
	edges = edges.astype(int)

	# 2. Apply Hough transform and get the peaks
	hough_h, hough_v = hough_kb(edges)
	perspective_control.verbosity.update_progress(verbosity, 3)
	peaks_h = hough_kb_peaks(hough_h)
	peaks_v = hough_kb_peaks(hough_v)
	perspective_control.verbosity.update_progress(verbosity, 3)

	# 3. Identify clusters in Hough spaces
	clusters_h = clusterise(peaks_h)
	perspective_control.verbosity.update_progress(verbosity, 3)
	clusters_v = clusterise(peaks_v)
	perspective_control.verbosity.update_progress(verbosity, 3)

	# 4. Apply Hough transform to Hough spaces
	hough_h_hough_h, hough_h_hough_v = hough_kb(clusters_h)
	perspective_control.verbosity.update_progress(verbosity, 3)
	hough_v_hough_h, hough_v_hough_v = hough_kb(clusters_v)
	perspective_control.verbosity.update_progress(verbosity, 3)

	# 5. Identify horizontal vanishing point
	if (np.max(hough_h_hough_h) > np.max(hough_h_hough_v)):
		k_, b_ = np.unravel_index(np.argmax(hough_h_hough_h), hough_h_hough_h.shape)
		k_ = -1 + k_ * 2 / (hough_h_hough_h.shape[0] - 1)
		b_ = b_ - hough_h.shape[1]
		k0 = int(hough_h.shape[0] / 2) # k = 0
		k1 = hough_h.shape[0] - 1 # k = 1
		b0 = int((k0 - b_) / k_)
		b1 = int((k1 - b_) / k_)
	else:
		k_, b_ = np.unravel_index(np.argmax(hough_h_hough_v), hough_h_hough_v.shape)
		k_ = -1 + k_ * 2 / (hough_h_hough_v.shape[0] - 1)
		b_ = b_ - hough_h.shape[0]
		k0 = int(hough_h.shape[0] / 2) # k = 0
		k1 = hough_h.shape[0] - 1 # k = 1
		b0 = int(k_ * k0 + b_)
		b1 = int(k_ * k1 + b_)
	x_h = b0 - image.shape[1]
	y_h = x_h - b1 + image.shape[1]

	# 6. Identify vertical vanishing point
	if (np.max(hough_v_hough_h) > np.max(hough_v_hough_v)):
		k_, b_ = np.unravel_index(np.argmax(hough_v_hough_h), hough_v_hough_h.shape)
		k_ = -1 + k_ * 2 / (hough_v_hough_h.shape[0] - 1)
		b_ = b_ - hough_v.shape[1]
		k0 = int(hough_v.shape[0] / 2) # k = 0
		k1 = hough_v.shape[0] - 1 # k = 1
		b0 = int((k0 - b_) / k_)
		b1 = int((k1 - b_) / k_)
	else:
		k_, b_ = np.unravel_index(np.argmax(hough_v_hough_v), hough_v_hough_v.shape)
		k_ = -1 + k_ * 2 / (hough_v_hough_v.shape[0] - 1)
		b_ = b_ - hough_v.shape[0]
		k0 = int(hough_v.shape[0] / 2) # k = 0
		k1 = hough_v.shape[0] - 1 # k = 1
		b0 = int(k_ * k0 + b_)
		b1 = int(k_ * k1 + b_)
	y_v = b0 - image.shape[0]
	x_v = y_v - b1 + image.shape[0]
	perspective_control.verbosity.update_progress(verbosity, 2)

	return (x_h, y_h), (x_v, y_v)
