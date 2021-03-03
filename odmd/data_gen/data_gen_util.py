# File: data_gen_util.py

import os, IPython, numpy as np

def add_perturbations(bb_3D, bb, odmd_data):
	if odmd_data.perturb:
		if odmd_data.std_dev > 0:
			bb = bounding_box_perturbation(bb, odmd_data.std_dev)
		if odmd_data.shuffle > 0:
			bb = bounding_box_shuffle(bb, odmd_data.shuffle)
		if odmd_data.cam_dev > 0:
			bb_3D = camera_move_perturbation(bb_3D, odmd_data.cam_dev)
	return bb_3D, bb

def bounding_box_perturbation(bb, std_dev):
	# Add random noise to bounding box.
	dev = np.random.normal(scale=std_dev,size=(bb["n_positions"],4,bb["n_ex"]))
	bb["bboxes"][:,:4,:] += dev
	return bb

def bounding_box_shuffle(bb, shuffle):
	# Randomly shuffle a percentage of bounding boxes to learn data selection.
	dim = bb["bboxes"].shape
	n_shuffle = int(dim[2] * shuffle)
	change_idx = np.random.choice(range(dim[2]), n_shuffle, replace=False)
	replace_idx = np.random.choice(range(dim[2]), n_shuffle)
	position_idx = np.random.choice(range(dim[0]), n_shuffle)
	bb["bboxes"][position_idx,:4,change_idx] = \
									bb["bboxes"][position_idx,:4,replace_idx]
	return bb

def camera_move_perturbation(bb_3D, cam_dev):
	dev = np.random.normal(scale=cam_dev,size=(bb_3D["n_positions"] - 1, 3,
															bb_3D["n_ex"]))
	bb_3D["camera_movement"] += dev
	return bb_3D

def bb_to_inputs(bb_3D, bb, n_obs=10):
	# Prepare input data based on the number of observations used.
	idx = np.round(np.linspace(0, bb_3D["n_positions"]-1, n_obs)).astype("int") 
	input_bb = np.array(bb["bboxes"])[idx,:-1]	
	camera_move = np.array(bb_3D["camera_movement"])[idx[:-1]]
	labels = np.array(bb["bboxes"])[-1][-1]	
	# Can also use more labels if training for intermediate depth:
	#	labels = np.array(bb["bboxes"])[:,-1]	
	return input_bb, camera_move, labels
