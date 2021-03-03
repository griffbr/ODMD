# File: analytic_model.py

import os, IPython, numpy as np, yaml
from copy import deepcopy

def Box_LS(input_bb, camera_move, n_obs=10):
	input_bb = missing_detection_filter(input_bb)
	# Find Ax = b least-squares solution (see equation sheet for details).
	n_examples = len(input_bb[0,0])
	predictions = np.zeros(n_examples)
	A = np.zeros(shape=(2*n_obs, 3))
	A[:n_obs,1]=1; A[n_obs:,2] = 1
	b = np.zeros(2*n_obs)
	z = np.zeros(n_obs)
	for i in range(n_examples):
		w = input_bb[:,2,i]
		h = input_bb[:,3,i]
		z[:-1] = camera_move[:,2,i]
		b = np.concatenate((w*z, h*z))
		A[:,0] = np.concatenate((w, h))
		try:
			x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T),b)
		except:
			print("Warning! Matrix A.T A is not invertable! x is not solved.")
			x = np.zeros(3)
		predictions[i] = x[0]
	return predictions

def missing_detection_filter(input_bb):
	miss_idx = np.argwhere(input_bb[:,2,:] == 0) 
	n_miss = len(miss_idx)
	if n_miss > 0:
		print("\nMissing %i observations! Using nearest valid.\n" % n_miss)
		bb_init = deepcopy(input_bb)
		for idx in miss_idx:
			# Replace missing detection with closest valid obsesrvation.
			obs_idx = np.argwhere((bb_init[:,2,idx[1]]==0)==False)
			try:
				replace_idx = obs_idx[np.argmin(abs(obs_idx-idx[0]))][0]
			except:
				print("No observation at all!")
				replace_idx = 0
			input_bb[idx[0],:,idx[1]] = input_bb[replace_idx,:,idx[1]]
	return input_bb
	
def bb_m_parallax(input_bb, camera_move):
	# NOTE: Be sure that input number of observations is two!
	in_bb = missing_detection_filter(input_bb)
	# Using deltaZ neq 0 method from derivation.
	# Load camera parameters. Normalize cx, cy, fx, fy by image size.
	cam_file = "../config/camera/hsr_grasp_camera.yaml"
	params = yaml.full_load(open(cam_file))
	dim = params["image_dim"]
	cx, cy = params["cx"]/dim[1], params["cy"]/dim[0]
	fx, fy = params["fx"]/dim[1], params["fy"]/dim[0]
	# Find x0, xf, y0, yf, w0/wf, h0/hf, and Delta X, Y from two observations. 
	dX, dY = camera_move[0, 0, :], camera_move[0, 1, :]
	x0, xf, y0, yf = in_bb[0,0,:], in_bb[1,0,:], in_bb[0,1,:], in_bb[1,1,:]
	w0, wf, h0, hf = in_bb[0,2,:], in_bb[1,2,:], in_bb[0,3,:], in_bb[1,3,:]
	# Solve for depth using x and y motion parallax then average result.
	depth_mpx = dX*fx / ((xf-cx) - (x0-cx)*(wf/w0))
	depth_mpy = dY*fy / ((yf-cy) - (y0-cy)*(hf/h0))
	# Replace Inf. values with one.
	replace_idx = np.argwhere(np.isfinite(depth_mpx)==False)
	for idx in replace_idx:
		depth_mpx[idx] = 1
		depth_mpy[idx] = 1
	"""
	# No deltaZ version:
	depth_mpx = dX*fx / (xf - x0)
	depth_mpy = dY*fy / (yf - y0)
	"""
	predictions = (depth_mpx + depth_mpy) / 2
	return predictions

def bb_opt_expansion(input_bb, camera_move):
	# NOTE: Be sure that input number of observations is two!
	in_bb = missing_detection_filter(input_bb)
	# Load camera parameters. Normalize cx, cy, fx, fy by image size.
	cam_file = "../config/camera/hsr_grasp_camera.yaml"
	params = yaml.full_load(open(cam_file))
	dim = params["image_dim"]
	cx, cy = params["cx"]/dim[1], params["cy"]/dim[0]
	fx, fy = params["fx"]/dim[1], params["fy"]/dim[0]
	# Find x0, xf, y0, yf, w0/wf, h0/hf, and Delta X, Y from two observations. 
	dZ = camera_move[0, 2, :]
	x0, xf, y0, yf = in_bb[0,0,:], in_bb[1,0,:], in_bb[0,1,:], in_bb[1,1,:]
	w0, wf, h0, hf = in_bb[0,2,:], in_bb[1,2,:], in_bb[0,3,:], in_bb[1,3,:]
	# Solve for depth using w and h optical expansion then average result.
	depth_oew = dZ / (1 - (wf/w0))
	depth_oeh = dZ / (1 - (hf/h0))
	# Replace Inf. values with one.
	replace_idx = np.argwhere(np.isfinite(depth_oew)==False)
	for idx in replace_idx:
		depth_oew[idx] = 1
		depth_oeh[idx] = 1
	predictions = (depth_oew + depth_oeh) / 2
	return predictions

