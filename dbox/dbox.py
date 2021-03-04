import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython, math, os

# LSTM design inspired by https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091

def pass_arg_for_model(model_name, **kwargs):
	if model_name == "DBox":
		model = DBox(**kwargs)
	return model

class DBox(nn.Module):
	def __init__(self,input_obs,hidden_sz,n_obs,fcs=[1],fc_drop=0,peep=False,
															all_loss=False):
		super(DBox, self).__init__()
		# LSTM.
		input_sz = input_obs*7
		self.idx = [[i+j for j in range(input_obs)] 
											for i in range(n_obs-input_obs+1)]
		self.n_in_obs = input_obs
		self.input_sz = input_sz
		self.hidden_sz = hidden_sz
		self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))	
		self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
		self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
		self.peephole = peep
		self.all_loss = all_loss
		if self.all_loss: self.n_predict = len(self.idx)
		else: self.n_predict = 1
		if peep:
			self.V = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))	
		self.cat_sz = n_obs * 7
		self.init_weights()
		# Linear layers.
		self.n_fc = len(fcs)
		self.relu = nn.ReLU(inplace=True)
		self.linears = nn.ModuleList([nn.Linear(hidden_sz+self.cat_sz,fcs[0])])
		for i in range(self.n_fc-1):
			self.linears.append(nn.Linear(fcs[i] + self.cat_sz, fcs[i+1]))
		self.linears.append(nn.Linear(fcs[-1],1))
		# Dropout.
		if fc_drop > 0:
			self.drop = True
			self.dropout = nn.Dropout(p=fc_drop)
		else: self.drop = False
		self.return_info = False

	def init_weights(self):
		stdv = 1.0 / math.sqrt(self.hidden_sz)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)

	def forward(self, x, init_state=None):
		"""Assumes x is of shape (batch, sequence, feature)"""
		bs, seq_sz, _ = x.size()
		hidden_seq = []
		if init_state is None:
			h_t, c_t = (torch.zeros(bs, self.hidden_sz).to(x.device),
						torch.zeros(bs, self.hidden_sz).to(x.device))
		else:
			h_t, c_t = init_state

		HS = self.hidden_sz
		#for t in range(seq_sz):
		#	x_t = x[:, t, :]
		for idx in self.idx:
			x_t = x[:, idx, :].reshape(bs, 1, self.input_sz)[:,0,:]
			# Batch computations into single matrix multiplication.
			if self.peephole:
				gates = x_t @ self.W + h_t @ self.U + self.bias + c_t @ self.V
			else:
				gates = x_t @ self.W + h_t @ self.U + self.bias
				o_t = torch.sigmoid(gates[:, HS*3:]) # output
			i_t, f_t, g_t = (
				torch.sigmoid(gates[:, :HS]), # input
				torch.sigmoid(gates[:, HS:HS*2]), # forget
				torch.tanh(gates[:, HS*2:HS*3]),
			)
			c_t = f_t * c_t + i_t * g_t
			if self.peephole:
				gates = x_t @ self.W + h_t @ self.U + self.bias + c_t @ self.V
				o_t = torch.sigmoid(gates[:, HS*3:]) # output
			h_t = o_t * torch.tanh(c_t)
			if self.return_info or self.all_loss:
				hidden_seq.append(h_t.unsqueeze(0))

		# Linear layers.
		x_cat = x.reshape(1, bs, self.cat_sz)
		d_set = torch.zeros(bs, self.n_predict)
		if self.all_loss: h_t_set = hidden_seq
		else: h_t_set = [h_t.unsqueeze(0)]
		for j, d in enumerate(h_t_set):
			for i in range(self.n_fc):
				d = torch.cat((d, x_cat), 2)
				d = self.linears[i](d)
				d = self.relu(d)
				if self.drop:
					d = self.dropout(d)
			d = self.linears[-1](d)
			d_set[:,j] = d.reshape(-1)

		if self.return_info:
			hidden_seq = torch.cat(hidden_seq, dim=0)
			# Reshape from (sequence, batch, feature) to (batch, seq, feature)
			hidden_seq = hidden_seq.transpose(0, 1).contiguous()
			return d_set, hidden_seq, (h_t, c_t)
		else:
			return d_set
