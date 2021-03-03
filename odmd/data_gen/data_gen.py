# File: data_gen.py

import IPython, os, numpy as np, yaml

class DataGenerator:
	"""
	DataGenerator generates random data for depth boxes.
	"""

	def __init__(self, data_params):
		"""
		Args:
			data_params (dict): dictionary with the following keys and values:
				z_lim ([float,float]): minimum and maximum object start depth.
				move_max (list[float]): maximum position change for X, Y, Z
				size_range (list[[float,float]]): minimum and maximum width and
					height of object in world coordinates.
				num_pos (int): number of positions that object is viewed from.
		"""
		self.set_yaml_params(data_params)
		self.move_range = np.array(self.move_max) - self.move_min
		self.z_range = self.z_lim[1] - self.z_lim[0]
		self.size_range = self.size_lim[1] - self.size_lim[0]

	def set_yaml_params(self, yaml_file):
		self.set_params(yaml.full_load(open(yaml_file)))

	def set_params(self, params):
		for _, key in enumerate(params.keys()):
			setattr(self, key, params[key])

	def initialize_data_gen(self, camera_config):
		self.set_yaml_params(camera_config)
		# See derivation 200520 page 33.
		cfx = self.cx / self.fx; self.xmina = -cfx
		self.xminb = self.move_max[0] +self.size_lim[1]/2 +self.move_max[2]*cfx 
		wfx = (self.image_dim[1] - self.cx) / self.fx; self.xmaxa = wfx
		self.xmaxb = -self.move_max[0] -self.size_lim[1]/2-self.move_max[2]*wfx
		cfy = self.cy / self.fy; self.ymina = -cfy
		self.yminb = self.move_max[1] +self.size_lim[1]/2 +self.move_max[2]*cfy 
		wfy = (self.image_dim[0] - self.cy) / self.fy; self.ymaxa = wfy
		self.ymaxb = -self.move_max[1] -self.size_lim[1]/2-self.move_max[2]*wfy
		self.xa = self.xmaxa - self.xmina; self.xb = self.xmaxb - self.xminb
		self.ya = self.ymaxa - self.ymina; self.yb = self.ymaxb - self.yminb
		self.fx_norm = self.fx / self.image_dim[1] 
		self.cx_norm = self.cx / self.image_dim[1]
		self.fy_norm = self.fy / self.image_dim[0]
		self.cy_norm = self.cy / self.image_dim[0]
		# Check that bounds are valid.
		if (self.z_lim[0] * self.xmina + self.xminb > 0) or (self.z_lim[0] *
												self.ymina + self.yminb > 0):
			print("\n\nData gen. bounds are not valid! Try increasing z.\n\n")
			IPython.embed()	

	def generate_object_examples(self, n_ex):
		# Find random initial and final object positions, then intermediate.
		rdm = np.random.rand(n_ex * 8)
		sign = np.random.randint(0,2, size=n_ex*3)*2 - 1
		p = np.zeros(shape=(2,3,n_ex))
		# Find initial Z position, then X(Z) and Y(Z) within the field of view.
		p[0][2] = self.z_lim[0] + self.z_range * rdm[:n_ex]
		p[0][0] = p[0][2]*self.xmina + self.xminb + \
					rdm[n_ex:n_ex*2]*( p[0][2]*self.xa + self.xb )
		p[0][1] = p[0][2]*self.ymina + self.yminb + \
					rdm[n_ex*2:n_ex*3]*( p[0][2]*self.ya + self.yb )
		for i in range(3):
			p[1][i] = p[0][i] + sign[n_ex*i:n_ex*(i+1)] * (self.move_min[i] + \
								self.move_range[i] *rdm[n_ex*(i+3):n_ex*(i+4)])
		if self.end_swap:
			# For greater sample diversity, switch start / end points randomly.
			swap = np.argwhere(np.random.randint(0,2, size=n_ex))
			p[[1,0],:,swap] = p[[0,1],:,swap]
		if self.num_pos>2: p = self.add_intermediate_positions(p)
		# Determine camera movement for each position.
		# Note: camera movement is opposite (-1) of object movement p.
		movement = (p[-1] - p)[:-1]
		# Find random height and width of objects.
		s = [self.size_lim[0] + self.size_range * rdm[n_ex*(i+6):n_ex*(i+7)] 
				for i in range(2)]
		obj_examples = {"positions": p, "camera_movement": np.array(movement), 
						"sizes": s, "n_ex": n_ex, "n_positions": self.num_pos}
		bb = self.find_image_bb_from_objects(obj_examples)
		return obj_examples, bb

	def add_intermediate_positions(self, p_init):
		# Add intermediate object positions between initial and final pose.
		n_ex = len(p_init[0][0])
		rdm = np.random.rand(n_ex * 3 * (self.num_pos - 2))
		dp = p_init[1] - p_init[0]
		p = np.zeros(shape=(self.num_pos-2,3,n_ex))
		for i in range(3):
			for j in range(0,self.num_pos-2):
				p[j][i] = p_init[0][i] + dp[i] \
											* rdm[n_ex*(i+j*3):n_ex*(i+1+j*3)]	
		# Sort intermediate points to be monotonicly increasing or decreasing.
		p.sort(axis=0)
		descend = dp<0
		p[:,descend] = p[::-1][:,descend]
		p_out = np.zeros(shape=(self.num_pos,3,n_ex))
		p_out [[0,-1]] = [p_init[0], p_init[-1]]
		p_out[1:-1] = p
		return p_out

	def find_image_bb_from_objects(self, objects):
		# Find image bounding boxes for each object and position.
		s = objects['sizes']
		bboxes = [[] for i in range(objects['n_positions'])]
		for i, p in enumerate(objects['positions']):
			xc = p[0]*self.fx_norm/p[2] + self.cx_norm
			yc = p[1]*self.fy_norm/p[2] + self.cy_norm
			w = s[0]*self.fx_norm/p[2]
			h = s[1]*self.fy_norm/p[2]
			z = p[2]
			bboxes[i] = [xc,yc,w,h,z]
		image_bb = {"bboxes": np.array(bboxes), 
					"n_positions": objects["n_positions"],
					"n_ex": objects["n_ex"], "image_dim": self.image_dim,
					"fx_norm": self.fx_norm, "fy_norm": self.fy_norm,
					"bbox_format": "[xc_norm; yc_norm; w_norm; h_norm; Z]"}
		return image_bb
