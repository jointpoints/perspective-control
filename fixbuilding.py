__author__ = 'Andrei Eliseev'





import sys
import perspective_control.verbosity





if __name__ == '__main__':
	# 1. Process arguments
	helpline = 'USAGE:\n\tfixbuilding <image_path> [-ov]\nWHERE:\n\t<image_path> : a path to image of a building;\n\t-o           : show original image in final result demonstration.\n\t-v           : switch on verbosity.\n'
	verbosity_label_capacity = 51
	args = set(sys.argv[1:])
	if len(args) == 0:
		print(helpline)
		exit()
	show_original = '-o' in args
	verbosity = '-v' in args
	args = args - {'-o', '-v'}
	if len(args) != 1:
		print(helpline)
		exit()
	image_path = tuple(args)[0]
	if verbosity:
		print(f'Perspective fix for "{image_path}".\n')
		print(' ' * 23 + 'Step' + ' ' * 23 + '|' + ' ' * 6 + 'Progress' + ' ' * 6 + '|', end='')
	
	# 2. Necessary imports
	perspective_control.verbosity.start_progress(verbosity, 'Initialisation', verbosity_label_capacity)
	import perspective_control.vanishing_points
	perspective_control.verbosity.update_progress(verbosity, 5)
	import perspective_control.transform
	perspective_control.verbosity.update_progress(verbosity, 5)
	from skimage import io
	perspective_control.verbosity.update_progress(verbosity, 5)
	from matplotlib import pyplot as plt
	perspective_control.verbosity.update_progress(verbosity, 5)

	# 3. Open the image
	perspective_control.verbosity.start_progress(verbosity, 'Image loading', verbosity_label_capacity)
	try:
		image = io.imread(image_path)
		perspective_control.verbosity.update_progress(verbosity, 20)
	except:
		perspective_control.verbosity.update_progress(verbosity, 20)
		print('\n\nERROR. Image was not found.')
		exit()

	# 4. Find vanishing points
	v_h, v_v = perspective_control.vanishing_points.get_vanishing_points(image, verbosity)

	# 5. Get rid of vanishing points
	corrected_image = perspective_control.transform.remove_perspective(image, v_h, v_v, verbosity)

	# 6. Show the result
	if not show_original:
		plt.imshow(corrected_image)
		plt.xticks([])
		plt.yticks([])
	else:
		plt.subplot(1, 2, 1)
		plt.imshow(image)
		plt.xticks([])
		plt.yticks([])
		plt.subplot(1, 2, 2)
		plt.imshow(corrected_image)
		plt.xticks([])
		plt.yticks([])
	plt.show()
