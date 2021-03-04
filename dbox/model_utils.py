import torch, os, IPython, numpy as np, yaml
from copy import deepcopy

from .dbox import *

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(model_config, n_pos=10):
	m_param = yaml.full_load(open(model_config))
	m_param["params"]["n_obs"] = n_pos
	# Take model configuration parameters and set up network architecture.
	net = pass_arg_for_model(m_param["model_name"], **m_param["params"])
	# Randomly select between both GPUs (distribute training).
	gpus = ["cuda:0", "cuda:1"]
	idx = np.random.randint(0,2)
	try:
		device = torch.device(gpus[idx])
		net.to(device)
	except:
		try:
			device = torch.device(gpus[idx-1])
			net.to(device)
		except:
			device = torch.device("cpu")
			net.to(device)
	m_param["n_predict"] = net.n_predict
	return net, device, m_param

def load_training_params(train_config):
	train = yaml.full_load(open(train_config))
	train["display_iter"] = int(train["train_iter"] / train["n_display"])
	train["save_iter"] = np.linspace(0, train["train_iter"], 
								train["n_train_model_saves"] + 1).astype(int)
	return train

def load_weights(net, weight_file):
	# Load weights onto already initialized network model.
	try:
		net.load_state_dict(torch.load(weight_file))
	except:
		net.load_state_dict(torch.load(weight_file, map_location=lambda 
														storage, loc: storage))
	return net

class BoundingBoxToNetwork:
	"""
	BoudingBoxToNetwork converts bounding boxes to network input.
	"""

	def __init__(self, m_params, n_bat=1):
		"""
		Args:
			n_obs (int): Number of bounding box and movement inputs to network.
			n_bat (int): Number of input sets batched together.
		"""
		self.n_obs = m_params["params"]["n_obs"]
		self.bb_in = self.n_obs * 4
		#self.cam_in = (self.n_obs - 1) * 3
		self.cam_in = self.n_obs * 3
		self.in_sz = self.bb_in + self.cam_in
		self.prediction = m_params["prediction"]
		self.sequence = m_params["sequence_in"]
		if self.sequence:
			self.seq_dist = m_params["sequence_dist"]
		self.n_predict = m_params["n_predict"]
		self.set_batch(n_bat)

	def set_batch(self, n_bat):
		# Change network input batch size.
		self.n_bat = n_bat
		self.tmp_in = np.zeros(shape=(n_bat, self.in_sz), dtype="float32")
		self.labels = torch.zeros(n_bat, self.n_predict, dtype=torch.float)
		if self.sequence:
			self.inputs =torch.zeros(self.n_bat,self.n_obs,7,dtype=torch.float)
			self.tmp2=np.zeros(shape=(self.n_bat,self.n_obs,7),dtype="float32")
		else:
			self.inputs = torch.zeros(self.n_bat,self.in_sz,dtype=torch.float)

	def bb_to_labels(self, bb_3D, bb):
		# Convert bounding box data to network input and labels.
		idx = np.linspace(0, bb_3D["n_positions"] - 1, self.n_obs, dtype=int)
		cam_idx = np.linspace(0,bb_3D["n_positions"]-2,self.n_obs-1,dtype=int)
		bb = missing_detection_network_filter(bb, idx)
		self.tmp_in[:,:self.bb_in] = bb["bboxes"][idx,:-1].reshape(
													self.bb_in, self.n_bat).T
		self.tmp_in[:,-self.cam_in:-3] = bb_3D["camera_movement"][
								cam_idx].reshape(self.cam_in-3, self.n_bat).T

		"""
		# Temp change for recovering results from old networks.
		dim = bb_3D["camera_movement"].shape
		temp = np.zeros(shape=(dim[0]+1,dim[1],dim[2]))
		temp[:-1] = bb_3D["camera_movement"]
		temp -= temp[0]
		bb_3D["camera_movement"] = temp[1:]
		self.tmp_in[:,-self.cam_in:] = np.array(bb_3D["camera_movement"])[
								idx[1:]-1].reshape(self.cam_in, self.n_bat).T
		"""

		# Convert input and labels for prediction type.
		self.labels[:] = torch.from_numpy(bb["bboxes"][-self.n_predict:,-1].T)
		#self.labels[:] = torch.from_numpy(bb["bboxes"][-1][-1])
		if self.prediction == "normalized":
			self.norm = np.linalg.norm(self.tmp_in[:,
										-self.cam_in:-self.cam_in+3], axis=1)
			self.tmp_in[:,-self.cam_in:] = np.multiply(
					self.tmp_in[:,-self.cam_in:],(1/self.norm)[:, np.newaxis])
			self.labels[:] /= np.tile(self.norm, (self.n_predict,1)).T
			#self.labels[:] /=  self.norm
		
		# Sequence separates each observation for LSTM input.
		if self.sequence:
			self.tmp2[:,:,:4] = self.tmp_in[:,:self.bb_in].reshape(
													self.n_bat,self.n_obs,4)
			self.tmp2[:,:,-3:] = self.tmp_in[:,-self.cam_in:].reshape(
												self.n_bat, self.n_obs, 3)
			# Sequential has each movement relative to previous observation.
			if self.seq_dist:	
				self.tmp2[:,1:,-3:] = np.diff(self.tmp2[:,:,-3:], axis=1)
				self.tmp2[:,0,-3:] = 0
			self.inputs[:] = torch.from_numpy(self.tmp2)
		else:
			self.inputs[:] = torch.from_numpy(self.tmp_in)

def missing_detection_network_filter(bb, idx):
	miss_idx = np.argwhere(bb["bboxes"][idx,2]==0)
	n_miss = len(miss_idx)
	if n_miss > 0:
		print("\nMissing %i observations! Using nearest valid.\n" % n_miss)
		bb_init = deepcopy(bb["bboxes"])
		for idx in miss_idx:
			# Replace misssing detection with closest valid observation.
			obs_idx = np.argwhere((bb_init[:,2,idx[1]]==0)==False)
			try:
				replace_i = obs_idx[np.argmin(abs(obs_idx-idx[0]))][0]
			except:
				print("No replacement!")
				replace_i = 0
			bb["bboxes"][idx[0],:-1,idx[1]] =bb["bboxes"][replace_i,:-1,idx[1]]
	return bb
