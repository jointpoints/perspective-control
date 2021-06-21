__author__ = 'Andrei Eliseev'





# Necessary imports
import perspective_control.verbosity
from skimage import transform
import numpy as np
from math import cos, sin, asin





def remove_perspective(image, v_h, v_v, verbosity):
	verbosity_label_capacity = 51
	# 1. Estimate focal length
	perspective_control.verbosity.start_progress(verbosity, 'Removing perspective', verbosity_label_capacity)
	focal_length = 1#int((image.shape[0]**2 + image.shape[1]**2)**.5)

	# 2. Estimate rotation angles
	# 2.1. Find radius-vectors of vanishing points in 3D
	v_v_3d = np.array([[v_v[0] - v_h[0]], [0], [focal_length]])
	v_h_3d = np.array([[0], [v_h[1] - v_v[1]], [focal_length]])
	# 2.2. Normalise radius-vectors to only have directions
	v_v_3d_n = v_v_3d / np.linalg.norm(v_v_3d)
	v_h_3d_n = v_h_3d / np.linalg.norm(v_h_3d)
	# 2.3. Specify t_x and phi_x right away
	cos_x = v_v_3d_n[2]
	t_x = v_v[1]
	phi_x = asin(cos_x)
	# 2.4. We need v_h_3d to be a radius-vector of v_h after rotation about X axis.
	#      Let us apply the inverse rotation to see where the initial vector must be
	inverse_spin_x = np.array([[ cos(-phi_x), 0, sin(-phi_x)],
	                           [      0     , 1,      0     ],
	                           [-sin(-phi_x), 0, cos(-phi_x)]])
	v_h_3d_original = inverse_spin_x @ v_h_3d
	# 2.5. Specify t_y and phi_y
	cos_y = v_h_3d_n[2]
	t_y = v_h[0] + int(v_h_3d_original[0])
	phi_y = asin(cos_y) if v_h[1] > v_v[1] else -asin(cos_y)
	perspective_control.verbosity.update_progress(verbosity, 7)

	# 3. Apply rotation
	# 3.1. Shift the origin of the image
	#      [3 x 3]
	origin_shift = np.array([[1, 0, -t_x],
	                         [0, 1, -t_y],
	                         [0, 0,   1 ]])
	# 3.2. Go from uniform 2D to uniform 3D: (x, y, 1) -> (x, y, focal length, 1)
	#      [4 x 3]
	dim_shift_2to3 = np.array([[1, 0,      0      ],
	                           [0, 1,      0      ],
	                           [0, 0, focal_length],
	                           [0, 0,      1      ]])
	# 3.3. Spin the world about the X axis by phi_x radians
	#      [4 x 4]
	spin_x = np.array([[1,     0     ,      0     , 0],
	                   [0, cos(phi_x), -sin(phi_x), 0],
	                   [0, sin(phi_x),  cos(phi_x), 0],
	                   [0,     0     ,      0     , 1]])
	# 3.4. Spin the world about the Y axis by phi_y radians
	#      [4 x 4]
	spin_y = np.array([[ cos(phi_y), 0, sin(phi_y), 0],
	                   [     0     , 1,     0     , 0],
	                   [-sin(phi_y), 0, cos(phi_y), 0],
	                   [     0     , 0,     0     , 1]])
	# 3.5. Go from uniform 3D to uniform 2D: (x, y, z, 1) -> (x, y, 1)
	#      [3 x 4]
	dim_shift_3to2 = np.array([[1, 0,        0        , 0],
	                           [0, 1,        0        , 0],
	                           [0, 0, focal_length**-1, 0]])
	# 3.6. Shift the origin back
	#      [3 x 3]
	inv_origin_shift = np.array([[1, 0, t_x],
	                             [0, 1, t_y],
	                             [0, 0,  1 ]])
	# 3.7. Resulting rotation transform
	#      [3 x 3] x [3 x 4] x [4 x 4] x [4 x 4] x [4 x 3] x [3 x 3] = [3 x 3]
	transform_matrix = inv_origin_shift @ dim_shift_3to2 @ spin_y @ spin_x @ dim_shift_2to3 @ origin_shift
	major_transform = transform.EuclideanTransform(matrix=transform_matrix)
	perspective_control.verbosity.update_progress(verbosity, 7)

	# 4. Make the resulting image fit into the borders
	# 4.1. Initial corner points of the image
	corner_tl_orig = np.array([[0], [0], [1]])
	corner_tr_orig = np.array([[image.shape[1] - 1], [0], [1]])
	corner_bl_orig = np.array([[0], [image.shape[0] - 1], [1]])
	corner_br_orig = np.array([[image.shape[1] - 1], [image.shape[0] - 1], [1]])
	# 4.2. Transformed points
	corner_tl_trans = transform_matrix @ corner_tl_orig
	corner_tr_trans = transform_matrix @ corner_tr_orig
	corner_bl_trans = transform_matrix @ corner_bl_orig
	corner_br_trans = transform_matrix @ corner_br_orig
	corners_trans = np.hstack([corner_tl_trans, corner_tr_trans, corner_bl_trans, corner_br_trans])
	# 4.3. Inscribed box
	ibox_tl = np.array([[np.max(corners_trans[0, [0,2]] / corners_trans[2, [0,2]])], [np.max(corners_trans[1, :2] / corners_trans[2, :2])]])
	ibox_tr = np.array([[np.min(corners_trans[0, [1,3]] / corners_trans[2, [1,3]])], [np.max(corners_trans[1, :2] / corners_trans[2, :2])]])
	ibox_bl = np.array([[np.max(corners_trans[0, [0,2]] / corners_trans[2, [0,2]])], [np.min(corners_trans[1, 2:] / corners_trans[2, 2:])]])
	ibox_br = np.array([[np.min(corners_trans[0, [1,3]] / corners_trans[2, [1,3]])], [np.min(corners_trans[1, 2:] / corners_trans[2, 2:])]])
	ibox_width = ibox_tr[0][0] - ibox_tl[0][0]
	ibox_height = ibox_br[1][0] - ibox_tr[1][0]
	# 4.4. Translation
	fit_border_translation = np.array([[1, 0, -ibox_tl[0][0]],
	                                   [0, 1, -ibox_tl[1][0]],
	                                   [0, 0,        1     ]])
	# 4.5. Scale
	fit_border_scale = max((image.shape[1] / ibox_width, image.shape[0] / ibox_height))
	fit_border_scale = np.array([[fit_border_scale,        0        , 0],
	                             [       0        , fit_border_scale, 0],
	                             [       0        ,        0        , 1]])
	# 4.6. Resulting fit transform
	fit_border_matrix = fit_border_scale @ fit_border_translation
	fit_border_transform = transform.EuclideanTransform(matrix=fit_border_matrix)
	perspective_control.verbosity.update_progress(verbosity, 6)

	return transform.warp(image, (major_transform + fit_border_transform).inverse)
