import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.mps import MPS, ReLUMPS
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = 1e-6

class MERAnet_clean(nn.Module):
	def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
				 kernel=[2, 2, 2], virtual_dim=1,
				 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
				 label_site=None, path=None, init_std=1e-9, use_bias=True,
				 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
		#bond_dim parameter is non-sense
		super().__init__()
		self.input_dim = input_dim
		self.virtual_dim = bond_dim

		### Squeezing of spatial dimension in first step
		#self.kScale = 4  # what is this?
		#nCh = self.kScale ** 2 * nCh
		# self.input_dim = self.input_dim / self.kScale

		# print(nCh)
		self.nCh = nCh
		if isinstance(kernel, int):
			kernel = 3 * [kernel]
		self.ker = kernel

		num_layers = np.int(np.log2(input_dim[0]) - 2)
		self.num_layers = num_layers
		self.disentangler_list = []
		self.isometry_list = []
		iDim = (self.input_dim - 2) / 2

		for ii in range(num_layers):
			#feature_dim = 2 * nCh
			feature_dim = 2 * nCh
			# print(feature_dim)
			#First level disentanglers
			# First level isometries

			### First level MERA blocks
			self.disentangler_list.append(nn.ModuleList([ReLUMPS(input_dim=4,
											  output_dim=4,
											  nCh=nCh, bond_dim=4,
											  feature_dim=feature_dim, parallel_eval=parallel_eval,
											  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
										  for i in range(torch.prod(iDim))]))

			iDim = iDim + 1

			self.isometry_list.append(nn.ModuleList([ReLUMPS(input_dim=4,
													output_dim=1,
													nCh=nCh, bond_dim=bond_dim,
													feature_dim=feature_dim, parallel_eval=parallel_eval,
													adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
												for i in range(torch.prod(iDim))]))
			iDim = (iDim-2) / 2


		### Final MPS block
		self.mpsFinal = ReLUMPS(input_dim=49,
							output_dim=output_dim, nCh=1,
							bond_dim=bond_dim, feature_dim=feature_dim,
							adaptive_mode=adaptive_mode, periodic_bc=periodic_bc,
							parallel_eval=parallel_eval)

	def forward(self, x):
		iDim = self.input_dim
		b = x.shape[0]  # Batch size
		# Disentangler layer
		x_in = x

		for jj in range(self.num_layers):
			iDim = iDim // 2
			x_ent = x_in[:, :, 1:-1, 1:-1]
			# print('---------------')
			# print(x_ent.shape)
			#print('x_ent unfold shape: ',  x_ent.unfold(2, 2, 2).unfold(3, 2, 2).shape)
			x_ent = x_ent.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1,  4)
			#print('x_ent unfold->reshape shape: ',  x_ent.shape)
			#print('single x_ent unfold->reshape shape: ',  x_ent[:, :, 0].shape)
			# print(x_ent.shape)
			# print(len(self.disentangler_list[jj]))
			y_ent = [self.disentangler_list[jj][i](x_ent[:, :, i]) for i in range(len(self.disentangler_list[jj]))]
			y_ent = torch.stack(y_ent, dim=1) # 512, 3969, 4
			y_ent = y_ent.view(y_ent.shape[0], y_ent.shape[1], 2, 2)
			y_ent = y_ent.view(y_ent.shape[0], iDim[0]-1, iDim[1]-1, 2, 2)
			y_ent_list = []
			#print('y_ent shape: ', y_ent.shape)
			#print('torch cat col shape: ', torch.cat([y_ent[:, 0, i, :, :] for i in range(y_ent.shape[2])], dim=2).shape)
			for j in range(y_ent.shape[1]):
				y_ent_list.append(torch.cat([y_ent[:, j, i, :, :] for i in range(y_ent.shape[2])], dim=2))
			y_ent = torch.cat(y_ent_list, dim=1)

			#print('y_ent shape: ', y_ent.shape)
			x_iso = x_in
			#print(x_iso.shape)
			x_iso[:, :, 1:-1, 1:-1] = y_ent.view(b, self.nCh, y_ent.shape[1], y_ent.shape[2])
			#print('x_iso shape: ', x_iso.shape)
			x_iso = x_iso.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b,  self.nCh, -1, 4)
			y_iso = [self.isometry_list[jj][i](x_iso[:, :, i]) for i in range(len(self.isometry_list[jj]))]
			y_iso = torch.stack(y_iso, dim=1) # 512, 4096, 1

			x_in = y_iso.view(y_iso.shape[0], self.nCh, iDim[0], iDim[1])

		#print('x6 shape: ', x6.shape) # 512, 1, 2, 2

		y = x_in.view(b, self.nCh, iDim[0] * iDim[1])
		#print('LoTe y shape before mpsfinal ', y.shape)
		y = self.mpsFinal(y)
		return y.squeeze()

class MERAnet(nn.Module):
	def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
				 kernel=[2, 2, 2], virtual_dim=1,
				 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
				 label_site=None, path=None, init_std=1e-9, use_bias=True,
				 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
		#bond_dim parameter is non-sense
		super().__init__()
		self.input_dim = input_dim
		self.virtual_dim = bond_dim

		### Squeezing of spatial dimension in first step
		#self.kScale = 4  # what is this?
		#nCh = self.kScale ** 2 * nCh
		# self.input_dim = self.input_dim / self.kScale

		# print(nCh)
		self.nCh = nCh
		if isinstance(kernel, int):
			kernel = 3 * [kernel]
		self.ker = kernel


		iDim = (self.input_dim-2) / 2

		#feature_dim = 2 * nCh
		feature_dim = 2 * nCh
		# print(feature_dim)
		#First level disentanglers
		# First level isometries

		### First level MERA blocks
		self.Disentangler_1 = nn.ModuleList([ReLUMPS(input_dim=4,
										  output_dim=4,
										  nCh=nCh, bond_dim=4,
										  feature_dim=feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(iDim))])

		iDim = iDim + 1

		self.Isometry_1 = nn.ModuleList([ReLUMPS(input_dim=4,
												output_dim=1,
												nCh=nCh, bond_dim=bond_dim,
												feature_dim=feature_dim, parallel_eval=parallel_eval,
												adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
											for i in range(torch.prod(iDim))])


		iDim = (iDim-2) / 2

		### Second level MERA blocks
		self.Disentangler_2 = nn.ModuleList([ReLUMPS(input_dim=4,
												 output_dim=4,
												 nCh=nCh, bond_dim=4,
												 feature_dim=feature_dim, parallel_eval=parallel_eval,
												 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
											 for i in range(torch.prod(iDim))])

		iDim = iDim + 1

		self.Isometry_2 = nn.ModuleList([ReLUMPS(input_dim=4,
											 output_dim=1,
											 nCh=nCh, bond_dim=bond_dim,
											 feature_dim=feature_dim, parallel_eval=parallel_eval,
											 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
										 for i in range(torch.prod(iDim))])

		iDim = (iDim - 2) / 2

		### 3rd level MERA blocks
		self.Disentangler_3 = nn.ModuleList([ReLUMPS(input_dim=4,
												 output_dim=4,
												 nCh=nCh, bond_dim=4,
												 feature_dim=feature_dim, parallel_eval=parallel_eval,
												 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
											 for i in range(torch.prod(iDim))])

		iDim = iDim + 1

		self.Isometry_3 = nn.ModuleList([ReLUMPS(input_dim=4,
											 output_dim=1,
											 nCh=nCh, bond_dim=bond_dim,
											 feature_dim=feature_dim, parallel_eval=parallel_eval,
											 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
										 for i in range(torch.prod(iDim))])
		iDim = (iDim - 2) / 2

		### 4th level MERA blocks
		self.Disentangler_4 = nn.ModuleList([ReLUMPS(input_dim=4,
												 output_dim=4,
												 nCh=nCh, bond_dim=4,
												 feature_dim=feature_dim, parallel_eval=parallel_eval,
												 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
											 for i in range(torch.prod(iDim))])

		iDim = iDim + 1

		self.Isometry_4 = nn.ModuleList([ReLUMPS(input_dim=4,
											 output_dim=1,
											 nCh=nCh, bond_dim=bond_dim,
											 feature_dim=feature_dim, parallel_eval=parallel_eval,
											 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
										 for i in range(torch.prod(iDim))])
		iDim = (iDim - 2) / 2

		### 5th level MERA blocks
		self.Disentangler_5 = nn.ModuleList([ReLUMPS(input_dim=4,
												 output_dim=4,
												 nCh=nCh, bond_dim=4,
												 feature_dim=feature_dim, parallel_eval=parallel_eval,
												 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
											 for i in range(torch.prod(iDim))])

		iDim = iDim + 1

		self.Isometry_5 = nn.ModuleList([ReLUMPS(input_dim=4,
											 output_dim=1,
											 nCh=nCh, bond_dim=bond_dim,
											 feature_dim=feature_dim, parallel_eval=parallel_eval,
											 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
										 for i in range(torch.prod(iDim))])
		iDim = (iDim - 2) / 2
		### 6th level MERA blocks
		self.Disentangler_6 = nn.ModuleList([ReLUMPS(input_dim=4,
												 output_dim=4,
												 nCh=nCh, bond_dim=4,
												 feature_dim=feature_dim, parallel_eval=parallel_eval,
												 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
											 for i in range(torch.prod(iDim))])

		iDim = iDim + 1

		self.Isometry_6 = nn.ModuleList([ReLUMPS(input_dim=4,
											 output_dim=1,
											 nCh=nCh, bond_dim=bond_dim,
											 feature_dim=feature_dim, parallel_eval=parallel_eval,
											 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
										 for i in range(torch.prod(iDim))])


		iDim = (iDim - 2) / 2

		### Final MPS block
		self.mpsFinal = ReLUMPS(input_dim=4,
							output_dim=output_dim, nCh=1,
							bond_dim=bond_dim, feature_dim=feature_dim,
							adaptive_mode=adaptive_mode, periodic_bc=periodic_bc,
							parallel_eval=parallel_eval)

	def forward(self, x):
		iDim = self.input_dim // 2
		b = x.shape[0]  # Batch size
		# Disentangler layer
		x_ent = x[:, :, 1:-1, 1:-1]
		#print('x_ent unfold shape: ',  x_ent.unfold(2, 2, 2).unfold(3, 2, 2).shape)
		x_ent = x_ent.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1,  4)
		#print('x_ent unfold->reshape shape: ',  x_ent.shape)
		#print('single x_ent unfold->reshape shape: ',  x_ent[:, :, 0].shape)

		y_ent = [self.Disentangler_1[i](x_ent[:, :, i]) for i in range(len(self.Disentangler_1))]
		y_ent = torch.stack(y_ent, dim=1) # 512, 3969, 4
		y_ent = y_ent.view(y_ent.shape[0], y_ent.shape[1], 2, 2)
		y_ent = y_ent.view(y_ent.shape[0], iDim[0]-1, iDim[1]-1, 2, 2)
		y_ent_list = []
		#print('y_ent shape: ', y_ent.shape)
		#print('torch cat col shape: ', torch.cat([y_ent[:, 0, i, :, :] for i in range(y_ent.shape[2])], dim=2).shape)
		for j in range(y_ent.shape[1]):
			y_ent_list.append(torch.cat([y_ent[:, j, i, :, :] for i in range(y_ent.shape[2])], dim=2))
		y_ent = torch.cat(y_ent_list, dim=1)

		#print('y_ent shape: ', y_ent.shape)
		x_iso = x
		x_iso[:, :, 1:-1, 1:-1] = y_ent.view(b, self.nCh, y_ent.shape[1], y_ent.shape[2])
		#print('x_iso shape: ', x_iso.shape)
		x_iso = x_iso.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b,  self.nCh, -1, 4)
		y_iso = [self.Isometry_1[i](x_iso[:, :, i]) for i in range(len(self.Isometry_1))]
		y_iso = torch.stack(y_iso, dim=1) # 512, 4096, 1

		x1 = y_iso.view(y_iso.shape[0], self.nCh, iDim[0], iDim[1])

		#print('x1 shape: ', x1.shape)

		iDim = iDim // 2
		x_ent = x1[:, :, 1:-1, 1:-1]
		x_ent = x_ent.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1,  4)
		y_ent = [self.Disentangler_2[i](x_ent[:, :, i]) for i in range(len(self.Disentangler_2))]
		y_ent = torch.stack(y_ent, dim=1)  # 512, 3969, 4
		y_ent = y_ent.view(y_ent.shape[0], y_ent.shape[1], 2, 2)
		y_ent = y_ent.view(y_ent.shape[0], iDim[0] - 1, iDim[1] - 1, 2, 2)
		y_ent_list = []
		for j in range(iDim[0] - 1):
			y_ent_list.append(torch.cat([y_ent[:, j, i, :, :] for i in range(iDim[1] - 1)], dim=2))
		y_ent = torch.cat(y_ent_list, dim=1)

		x_iso = x1
		x_iso[:, :, 1:-1, 1:-1] = y_ent.view(b, self.nCh, y_ent.shape[1], y_ent.shape[2])
		x_iso = x_iso.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b,  self.nCh, -1, 4)
		y_iso = [self.Isometry_2[i](x_iso[:, :, i]) for i in range(len(self.Isometry_2))]
		y_iso = torch.stack(y_iso, dim=1) # 512, 4096, 1

		x2 = y_iso.view(y_iso.shape[0], self.nCh, iDim[0], iDim[1])

		#print('x2 shape: ', x2.shape)

		iDim = iDim // 2
		x_ent = x2[:, :, 1:-1, 1:-1]
		x_ent = x_ent.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
		y_ent = [self.Disentangler_3[i](x_ent[:, :, i]) for i in range(len(self.Disentangler_3))]
		y_ent = torch.stack(y_ent, dim=1)  # 512, 3969, 4
		y_ent = y_ent.view(y_ent.shape[0], y_ent.shape[1], 2, 2)
		y_ent = y_ent.view(y_ent.shape[0], iDim[0] - 1, iDim[1] - 1, 2, 2)
		y_ent_list = []
		for j in range(iDim[0] - 1):
			y_ent_list.append(torch.cat([y_ent[:, j, i, :, :] for i in range(iDim[1] - 1)], dim=2))
		y_ent = torch.cat(y_ent_list, dim=1)

		x_iso = x2
		x_iso[:, :, 1:-1, 1:-1] = y_ent.view(b, self.nCh, y_ent.shape[1], y_ent.shape[2])
		x_iso = x_iso.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
		y_iso = [self.Isometry_3[i](x_iso[:, :, i]) for i in range(len(self.Isometry_3))]
		y_iso = torch.stack(y_iso, dim=1)  # 512, 4096, 1

		x3 = y_iso.view(y_iso.shape[0], self.nCh, iDim[0], iDim[1])

		#print('x3 shape: ', x3.shape)

		iDim = iDim // 2
		x_ent = x3[:, :, 1:-1, 1:-1]
		x_ent = x_ent.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
		y_ent = [self.Disentangler_4[i](x_ent[:, :, i]) for i in range(len(self.Disentangler_4))]
		y_ent = torch.stack(y_ent, dim=1)  # 512, 3969, 4
		y_ent = y_ent.view(y_ent.shape[0], y_ent.shape[1], 2, 2)
		y_ent = y_ent.view(y_ent.shape[0], iDim[0] - 1, iDim[1] - 1, 2, 2)
		y_ent_list = []
		for j in range(iDim[0] - 1):
			y_ent_list.append(torch.cat([y_ent[:, j, i, :, :] for i in range(iDim[1] - 1)], dim=2))
		y_ent = torch.cat(y_ent_list, dim=1)

		x_iso = x3
		x_iso[:, :, 1:-1, 1:-1] = y_ent.view(b, self.nCh, y_ent.shape[1], y_ent.shape[2])
		x_iso = x_iso.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
		y_iso = [self.Isometry_4[i](x_iso[:, :, i]) for i in range(len(self.Isometry_4))]
		y_iso = torch.stack(y_iso, dim=1)  # 512, 4096, 1

		x4 = y_iso.view(y_iso.shape[0], self.nCh, iDim[0], iDim[1])

		#print('x4 shape: ', x4.shape)

		iDim = iDim // 2
		x_ent = x4[:, :, 1:-1, 1:-1]
		x_ent = x_ent.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
		y_ent = [self.Disentangler_5[i](x_ent[:, :, i]) for i in range(len(self.Disentangler_5))]
		y_ent = torch.stack(y_ent, dim=1)  # 512, 3969, 4
		y_ent = y_ent.view(y_ent.shape[0], y_ent.shape[1], 2, 2)
		y_ent = y_ent.view(y_ent.shape[0], iDim[0] - 1, iDim[1] - 1, 2, 2)
		y_ent_list = []
		for j in range(iDim[0] - 1):
			y_ent_list.append(torch.cat([y_ent[:, j, i, :, :] for i in range(iDim[1] - 1)], dim=2))
		y_ent = torch.cat(y_ent_list, dim=1)

		x_iso = x4
		x_iso[:, :, 1:-1, 1:-1] = y_ent.view(b, self.nCh, y_ent.shape[1], y_ent.shape[2])
		x_iso = x_iso.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
		y_iso = [self.Isometry_5[i](x_iso[:, :, i]) for i in range(len(self.Isometry_5))]
		y_iso = torch.stack(y_iso, dim=1)  # 512, 4096, 1

		x5 = y_iso.view(y_iso.shape[0], self.nCh, iDim[0], iDim[1])

		#print('x5 shape: ', x5.shape)

		iDim = iDim // 2
		x_ent = x5[:, :, 1:-1, 1:-1]
		x_ent = x_ent.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
		#print(x_ent.shape)
		#print(len(self.Disentangler_6))
		y_ent = [self.Disentangler_6[i](x_ent[:, :, i]) for i in range(len(self.Disentangler_6))]
		y_ent = torch.stack(y_ent, dim=1)  # 512, 3969, 4
		y_ent = y_ent.view(y_ent.shape[0], y_ent.shape[1], 2, 2)
		y_ent = y_ent.view(y_ent.shape[0], iDim[0] - 1, iDim[1] - 1, 2, 2)
		y_ent_list = []
		for j in range(iDim[0] - 1):
			y_ent_list.append(torch.cat([y_ent[:, j, i, :, :] for i in range(iDim[1] - 1)], dim=2))
		y_ent = torch.cat(y_ent_list, dim=1)

		x_iso = x5
		x_iso[:, :, 1:-1, 1:-1] = y_ent.view(b, self.nCh, y_ent.shape[1], y_ent.shape[2])
		x_iso = x_iso.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
		y_iso = [self.Isometry_6[i](x_iso[:, :, i]) for i in range(len(self.Isometry_6))]
		y_iso = torch.stack(y_iso, dim=1)  # 512, 4096, 1

		x6 = y_iso.view(y_iso.shape[0], self.nCh, iDim[0], iDim[1])

		#print('x6 shape: ', x6.shape) # 512, 1, 2, 2

		y = x6.view(b, self.nCh, iDim[0] * iDim[1])
		# print('LoTe y shape before mpsfinal ', y.shape)
		y = self.mpsFinal(y)
		return y.squeeze()


class loTeNet(nn.Module):
	def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
				 kernel=[2, 2, 2], virtual_dim=1,
				 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
				 label_site=None, path=None, init_std=1e-9, use_bias=True,
				 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
		super().__init__()
		self.input_dim = input_dim
		self.virtual_dim = bond_dim

		### Squeezing of spatial dimension in first step
		self.kScale = 4 # what is this?
		nCh =  self.kScale**2 * nCh
		self.input_dim = self.input_dim/self.kScale

		#print(nCh)
		self.nCh = nCh
		if  isinstance(kernel, int):
			kernel = 3 * [kernel]
		self.ker = kernel
		iDim = (self.input_dim/(self.ker[0]))

		feature_dim = 2*nCh 
		#print(feature_dim)
		### First level MPS blocks
		self.module1 = nn.ModuleList([ MPS(input_dim=(self.ker[0])**2,
			output_dim=self.virtual_dim, 
			nCh=nCh, bond_dim=bond_dim, 
			feature_dim=feature_dim, parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc) 
			for i in range(torch.prod(iDim))])

		self.BN1 = nn.BatchNorm1d(torch.prod(iDim).numpy(),affine=True)

		
		iDim = iDim/self.ker[1]
		feature_dim = 2*self.virtual_dim
		
		### Second level MPS blocks
		self.module2 = nn.ModuleList([ MPS(input_dim=self.ker[1]**2,
			output_dim=self.virtual_dim, 
			nCh=self.virtual_dim, bond_dim=bond_dim,
			feature_dim=feature_dim,  parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
			for i in range(torch.prod(iDim))])

		self.BN2 = nn.BatchNorm1d(torch.prod(iDim).numpy(),affine=True)

		iDim = iDim/self.ker[2]

		### Third level MPS blocks
		self.module3 = nn.ModuleList([ MPS(input_dim=self.ker[2]**2,
			output_dim=self.virtual_dim, 
			nCh=self.virtual_dim, bond_dim=bond_dim,  
			feature_dim=feature_dim, parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc) 
			for i in range(torch.prod(iDim))])

		self.BN3 = nn.BatchNorm1d(torch.prod(iDim).numpy(),affine=True)

		### Final MPS block
		self.mpsFinal = MPS(input_dim=len(self.module3), 
				output_dim=output_dim, nCh=1,
				bond_dim=bond_dim, feature_dim=feature_dim, 
				adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, 
				parallel_eval=parallel_eval)
		
	def forward(self,x):

		b = x.shape[0] #Batch size
		iDim = self.input_dim/(self.ker[0])
		#print(self.input_dim)
		#print(x.shape)
		#print(torch.prod(iDim))
		#print(self.nCh)
		# Level 1 contraction
		#print(x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],iDim[1]).shape)
		x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],iDim[1]).reshape(b,
					self.nCh,-1,(self.ker[0])**2)
		# print(x.shape)
		#print('x[:, :, 0].shape', x[:, :, 0].shape)
		#assert False
		#print('LoTe x total shape layer 1: ', x.shape)
		#print('LoTe x shape layer 1: ', x[:, :, 0].shape)
		y = [ self.module1[i](x[:,:,i]) for i in range(len(self.module1))]
		y = torch.stack(y,dim=1)
		#print(y.shape)
		#assert False
		y = self.BN1(y).unsqueeze(1)

		# Level 2 contraction

		y = y.view(b,self.virtual_dim,iDim[0],iDim[1])
		iDim = (iDim/self.ker[1])
		y = y.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],
				iDim[1]).reshape(b,self.virtual_dim,-1,self.ker[1]**2)
		#print('LoTe x total shape layer 2: ', y.shape)
		#print('LoTe x shape layer 2: ', y[:, :, 0].shape)
		x = [ self.module2[i](y[:,:,i]) for i in range(len(self.module2))]
		x = torch.stack(x,dim=1)
		x = self.BN2(x).unsqueeze(1)


		# Level 3 contraction
		x = x.view(b,self.virtual_dim,iDim[0],iDim[1])
		iDim = (iDim/self.ker[2])
		x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],
				iDim[1]).reshape(b,self.virtual_dim,-1,self.ker[2]**2)
		#print('LoTe x total shape layer 3: ', x.shape)
		#print('LoTe x shape layer 3: ', x[:, :, 0].shape)
		y = [self.module3[i](x[:,:,i]) for i in range(len(self.module3))]

		y = torch.stack(y,dim=1)
		y = self.BN3(y)

		if self.virtual_dim == 1:
			y = y.unsqueeze(2)
		if y.shape[1] > 1:
		# Final layer
			y = y.permute(0,2,1)
			#print('LoTe y shape before mpsfinal ', y.shape)
			y = self.mpsFinal(y)
		return y.squeeze()


class ConvTeNet(nn.Module):
	def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
				 kernel=[2, 2, 2], virtual_dim=1,
				 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
				 label_site=None, path=None, init_std=1e-9, use_bias=True,
				 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
		super().__init__()
		self.input_dim = input_dim
		self.virtual_dim = bond_dim

		### Squeezing of spatial dimension in first step
		#self.kScale = 4
		#nCh = self.kScale ** 2 * nCh
		self.input_dim = self.input_dim

		self.nCh = nCh
		if isinstance(kernel, int):
			kernel = 3 * [kernel]
		self.ker = kernel
		iDim = (self.input_dim / (self.ker[0]))
		#feature_dim = 2 * nCh
		feature_dim = 2 * self.ker[0] ** 2
		#print(feature_dim)
		### First level MPS blocks
		#(self.ker[0]) ** 2,
		self.module1 = nn.ModuleList([MPS(input_dim=nCh,
										  output_dim=self.virtual_dim,
										  nCh=nCh, bond_dim=bond_dim,
										  feature_dim=feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(iDim))])

		self.BN1 = nn.BatchNorm1d(torch.prod(iDim).numpy(), affine=True)

		iDim = iDim / self.ker[1]
		feature_dim = 2 * self.ker[1] ** 2

		### Second level MPS blocks
		self.module2 = nn.ModuleList([MPS(input_dim=self.virtual_dim,
										  output_dim=self.virtual_dim,
										  nCh=self.virtual_dim, bond_dim=bond_dim,
										  feature_dim=feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(iDim))])

		self.BN2 = nn.BatchNorm1d(torch.prod(iDim).numpy(), affine=True)

		iDim = iDim / self.ker[2]
		feature_dim = 2 * self.ker[2] ** 2
		### Third level MPS blocks
		self.module3 = nn.ModuleList([MPS(input_dim=self.virtual_dim,
										  output_dim=self.virtual_dim,
										  nCh=self.virtual_dim, bond_dim=bond_dim,
										  feature_dim=feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(iDim))])

		self.BN3 = nn.BatchNorm1d(torch.prod(iDim).numpy(), affine=True)
		feature_dim = 2 * self.virtual_dim
		### Final MPS block
		self.mpsFinal = MPS(input_dim=len(self.module3),
							output_dim=output_dim, nCh=1,
							bond_dim=bond_dim, feature_dim=feature_dim,
							adaptive_mode=adaptive_mode, periodic_bc=periodic_bc,
							parallel_eval=parallel_eval)

	def forward(self, x):

		b = x.shape[0]  # Batch size
		H, W = x.shape[2], x.shape[3]
		iDim = self.input_dim / (self.ker[0])
		#print('iDim: ', iDim)
		#print('x.shape: ',x.shape)
		# Level 1 contraction
		x_org = x
		x = x.unfold(2, self.ker[0], self.ker[0]).unfold(3, self.ker[0], self.ker[0]).reshape(b,  (self.ker[0]) ** 2, -1,  self.nCh)
		###
		x_unfold = x_org.unfold(2, self.ker[0], self.ker[0]).unfold(3, self.ker[0], self.ker[0]).reshape(b,  (self.ker[0]) ** 2, -1,  self.nCh)
		x_unfold = x_unfold.view(x_unfold.shape[0], x_unfold.shape[2], x_unfold.shape[1])
		print('unfolded x shape: ', x_unfold.shape)
		x_fold = F.fold(x_unfold, output_size=(H, W), kernel_size=(self.ker[0], self.ker[0]))
		print('x_fold shape: ', x_fold)
		assert False
		#print('x.shape: ',x.shape)
		#print(x.)
		#assert False
		# print(iDim)
		#x = torch.cat([x, x], dim=3)
		#print('x[:, :, 0].shape', x[:, :, 0].shape)
		print('Conv x total shape layer 1: ', x.shape)
		print('Conv x shape layer 1: ', x[:, :, 0].shape)
		print('After contraction x_i: ', self.module1[0](x[:, :, 0]).shape)
		y = [self.module1[i](x[:, :, i]) for i in range(len(self.module1))]
		y = torch.stack(y, dim=1)
		print(y.shape)
		assert False
		y = self.BN1(y).unsqueeze(1)

		# Level 2 contraction
		#print(y.shape)
		#iDim = (iDim / self.ker[1])
		y = y.view(b, self.virtual_dim, iDim[0], iDim[1])
		iDim = (iDim / self.ker[1])
		y = y.unfold(2, self.ker[1], self.ker[1]).unfold(3, self.ker[1], self.ker[1]).reshape(b, self.ker[1] ** 2, -1, self.virtual_dim)
		#print(y.shape)
		#print(y[:, :, 0].shape)
		#print('Conv x total shape layer 2: ', y.shape)
		#print('Conv x shape layer 2: ', y[:, :, 0].shape)

		x = [self.module2[i](y[:, :, i]) for i in range(len(self.module2))]
		#assert False

		x = torch.stack(x, dim=1)
		#print(x.shape)

		x = self.BN2(x).unsqueeze(1)

		# Level 3 contraction
		x = x.view(b, self.virtual_dim, iDim[0], iDim[1])
		iDim = (iDim / self.ker[2])
		x = x.unfold(2, self.ker[2], self.ker[2]).unfold(3, self.ker[2], self.ker[2]).reshape(b, self.ker[2] ** 2, -1, self.virtual_dim)
		#print('x[:, :, 0].shape before module3: ', x[:, :, 0].shape)
		#print('Conv x total shape layer 3: ', x.shape)
		#print('Conv x shape layer 3: ', x[:, :, 0].shape)
		y = [self.module3[i](x[:, :, i]) for i in range(len(self.module3))]

		y = torch.stack(y, dim=1)
		#print(y.shape)
		y = self.BN3(y)

		if self.virtual_dim == 1:
			y = y.unsqueeze(2)
		if y.shape[1] > 1:
			# Final layer
			y = y.permute(0, 2, 1)
			#print('Conv y shape before mpsfinal ', y.shape)
			y = self.mpsFinal(y)
		return y.squeeze()


class Combined_LoTeConv(nn.Module):
	def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
				 kernel1=[2, 2, 2], kernel2=[2, 2, 2],virtual_dim=1,
				 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
				 label_site=None, path=None, init_std=1e-9, use_bias=True,
				 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
		super().__init__()
		self.input_dim = input_dim
		self.virtual_dim = bond_dim

		### Squeezing of spatial dimension in first step
		self.LoTe_kScale = 4  # what is this?
		LoTe_nCh = self.LoTe_kScale ** 2 * nCh
		self.LoTe_input_dim = self.input_dim / self.LoTe_kScale

		# print(nCh)
		self.LoTe_nCh = LoTe_nCh
		if isinstance(kernel1, int):
			LoTe_kernel = 3 * [kernel1]
		else:
			LoTe_kernel = kernel1
		self.LoTe_ker = LoTe_kernel
		LoTe_iDim = (self.LoTe_input_dim / (self.LoTe_ker[0]))

		LoTe_feature_dim = 2 * LoTe_nCh

		## Parameters for Conv
		self.Conv_input_dim = self.input_dim

		self.Conv_nCh = nCh
		if isinstance(kernel2, int):
			Conv_kernel = 3 * [kernel2]
		else:
			Conv_kernel = kernel2
		self.Conv_ker = Conv_kernel
		Conv_iDim = (self.Conv_input_dim / (self.Conv_ker[0]))
		Conv_feature_dim = 2 * self.Conv_ker[0] ** 2

		### First level MPS blocks
		self.LoTe_module1 = nn.ModuleList([MPS(input_dim=(self.LoTe_ker[0]) ** 2,
										  output_dim=self.virtual_dim,
										  nCh=nCh, bond_dim=bond_dim,
										  feature_dim=LoTe_feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(LoTe_iDim))])

		self.LoTe_BN1 = nn.BatchNorm1d(torch.prod(LoTe_iDim).numpy(), affine=True)

		LoTe_iDim = LoTe_iDim / self.LoTe_ker[1]
		LoTe_feature_dim = 2 * self.virtual_dim

		### Second level MPS blocks
		self.LoTe_module2 = nn.ModuleList([MPS(input_dim=self.LoTe_ker[1] ** 2 + self.Conv_ker[1] ** 2,
										  output_dim=self.virtual_dim,
										  nCh=self.virtual_dim, bond_dim=bond_dim,
										  feature_dim=LoTe_feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(LoTe_iDim))])

		self.LoTe_BN2 = nn.BatchNorm1d(torch.prod(LoTe_iDim).numpy(), affine=True)
		LoTe_iDim = LoTe_iDim / self.LoTe_ker[2]

		### Third level MPS blocks
		self.LoTe_module3 = nn.ModuleList([MPS(input_dim=self.LoTe_ker[2] ** 2,
										  output_dim=self.virtual_dim,
										  nCh=self.virtual_dim, bond_dim=bond_dim,
										  feature_dim=2 * LoTe_feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(LoTe_iDim))])

		self.LoTe_BN3 = nn.BatchNorm1d(torch.prod(LoTe_iDim).numpy(), affine=True)

		### Final MPS block
		# self.LoTe_mpsFinal = MPS(input_dim=len(self.LoTe_module3),
		# 					output_dim=output_dim, nCh=1,
		# 					bond_dim=bond_dim, feature_dim=LoTe_feature_dim,
		# 					adaptive_mode=adaptive_mode, periodic_bc=periodic_bc,
		# 					parallel_eval=parallel_eval)

		##############################################################################
		##############################################################################
		############################ #  #######   ##### ###### #######################
		########################### ########## ### ##### #### ########################
		########################### ########## ### ###### ## #########################
		############################# # #######   ######## ###########################
		#############################################################################

		## Parameters for Conv TN
		### First level MPS blocks
		self.Conv_module1 = nn.ModuleList([MPS(input_dim=self.Conv_nCh,
										  output_dim=self.virtual_dim,
										  nCh=nCh, bond_dim=bond_dim,
										  feature_dim=Conv_feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(Conv_iDim))])

		self.Conv_BN1 = nn.BatchNorm1d(torch.prod(Conv_iDim).numpy(), affine=True)

		Conv_iDim = Conv_iDim / self.Conv_ker[1]
		Conv_feature_dim = 2 * self.Conv_ker[1] ** 2

		### Second level MPS blocks
		self.Conv_module2 = nn.ModuleList([MPS(input_dim=self.virtual_dim,
										  output_dim=self.virtual_dim,
										  nCh=self.virtual_dim, bond_dim=bond_dim,
										  feature_dim=2 * self.LoTe_kScale + Conv_feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(Conv_iDim))])

		self.Conv_BN2 = nn.BatchNorm1d(torch.prod(Conv_iDim).numpy(), affine=True)
		self.BN2 = nn.BatchNorm1d(torch.prod(Conv_iDim).numpy(), affine=True)

		Conv_iDim = Conv_iDim / self.Conv_ker[2]
		Conv_feature_dim = 2 * self.Conv_ker[2] ** 2
		### Third level MPS blocks
		self.Conv_module3 = nn.ModuleList([MPS(input_dim=2 * self.virtual_dim,
										  output_dim=self.virtual_dim,
										  nCh=self.virtual_dim, bond_dim=bond_dim,
										  feature_dim=Conv_feature_dim, parallel_eval=parallel_eval,
										  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
									  for i in range(torch.prod(Conv_iDim))])

		self.Conv_BN3 = nn.BatchNorm1d(torch.prod(Conv_iDim).numpy(), affine=True)
		self.BN3 = nn.BatchNorm1d(torch.prod(Conv_iDim).numpy(), affine=True)

		Conv_feature_dim = 2 * self.virtual_dim
		### Final MPS block
		# self.Conv_mpsFinal = MPS(input_dim=len(self.Conv_module3),
		# 					output_dim=output_dim, nCh=1,
		# 					bond_dim=bond_dim, feature_dim=Conv_feature_dim,
		# 					adaptive_mode=adaptive_mode, periodic_bc=periodic_bc,
		# 					parallel_eval=parallel_eval)

		self.mpsFinal = MPS(input_dim=len(self.Conv_module3),
								 output_dim=output_dim, nCh=1,
								 bond_dim=bond_dim, feature_dim=2 * Conv_feature_dim,
								 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc,
								 parallel_eval=parallel_eval)
	def forward(self, x):

		b = x.shape[0]  # Batch size
		LoTe_iDim = self.LoTe_input_dim / (self.LoTe_ker[0])
		Conv_iDim = self.Conv_input_dim / (self.Conv_ker[0])

		LoTe_x = x.unfold(2, LoTe_iDim[0], LoTe_iDim[0]).unfold(3, LoTe_iDim[1], LoTe_iDim[1]).reshape(b,
																			  self.LoTe_nCh, -1, (self.LoTe_ker[0]) ** 2)
		Conv_x = x.unfold(2, self.Conv_ker[0], self.Conv_ker[0]).unfold(3, self.Conv_ker[0],
																		self.Conv_ker[0]).reshape(b, (self.Conv_ker[0]) ** 2, -1,
																							  self.Conv_nCh)
		LoTe_y = [self.LoTe_module1[i](LoTe_x[:, :, i]) for i in range(len(self.LoTe_module1))]
		Conv_y = [self.Conv_module1[i](Conv_x[:, :, i]) for i in range(len(self.Conv_module1))]

		LoTe_y= torch.stack(LoTe_y, dim=1)
		Conv_y= torch.stack(Conv_y, dim=1)

		LoTe_y = self.LoTe_BN1(LoTe_y).unsqueeze(1)
		Conv_y = self.Conv_BN1(Conv_y).unsqueeze(1)

		# Level 2 contraction

		LoTe_y = LoTe_y.view(b, self.virtual_dim, LoTe_iDim[0], LoTe_iDim[1])
		LoTe_iDim = (LoTe_iDim / self.LoTe_ker[1])
		LoTe_y = LoTe_y.unfold(2, LoTe_iDim[0], LoTe_iDim[0]).unfold(3, LoTe_iDim[1],
														   LoTe_iDim[1]).reshape(b, self.virtual_dim, -1, self.LoTe_ker[1] ** 2)

		Conv_y = Conv_y.view(b, self.virtual_dim, Conv_iDim[0], Conv_iDim[1])
		Conv_iDim = (Conv_iDim / self.Conv_ker[1])
		Conv_y = Conv_y.unfold(2, self.Conv_ker[1], self.Conv_ker[1]).unfold(3, self.Conv_ker[1],
																			 self.Conv_ker[1]).reshape(b, self.Conv_ker[1] ** 2, -1,
																							  self.virtual_dim)

		#print(LoTe_y.shape)
		#print(Conv_y.permute(0, 3, 2, 1).shape)
		Combined_Feature_1 = torch.cat([LoTe_y, Conv_y.permute(0, 3, 2, 1)], dim=3)
		Lote_x2 = Combined_Feature_1
		Conv_x2 = Combined_Feature_1.permute(0, 3, 2, 1)

		# print('Lote_x2', Lote_x2.shape)
		# print(len(self.LoTe_module2))
		# for i in range(len(self.LoTe_module2)):
		# 	print('Now loop: ', i)
		# 	print(Lote_x2[:, :, i].shape)
		# 	self.LoTe_module2[i](Lote_x2[:, :, i])
		# assert False, 'stop here'
		LoTe_x2 = [self.LoTe_module2[i](Lote_x2[:, :, i]) for i in range(len(self.LoTe_module2))]
		Conv_x2 = [self.Conv_module2[i](Conv_x2[:, :, i]) for i in range(len(self.Conv_module2))]

		LoTe_x2 = torch.stack(LoTe_x2, dim=1)
		Conv_x2 = torch.stack(Conv_x2, dim=1)

		cat_x2 = torch.cat([LoTe_x2, Conv_x2], dim=2)
		bn_x2 = self.BN2(cat_x2).unsqueeze(1)
		#print("LoTe_x2", LoTe_x2.shape)
		#print("Conv_x2", Conv_x2.shape)

		#
		# # Level 3 contraction
		Lote_x3 = bn_x2.view(b, 2 * self.virtual_dim, LoTe_iDim[0], LoTe_iDim[1])
		LoTe_iDim = (LoTe_iDim / self.LoTe_ker[2])
		Lote_x3 = Lote_x3.unfold(2, LoTe_iDim[0], LoTe_iDim[0]).unfold(3, LoTe_iDim[1],
														   LoTe_iDim[1]).reshape(b, 2 * self.virtual_dim, -1, self.LoTe_ker[2] ** 2)


		#print('LoTe x total shape layer 3: ', Lote_x3.shape)
		#print('LoTe x shape layer 3: ', Lote_x3[:, :, 0].shape)

		Conv_x3 = bn_x2.view(b, 2 * self.virtual_dim, Conv_iDim[0], Conv_iDim[1])
		Conv_iDim = (Conv_iDim / self.Conv_ker[2])
		Conv_x3 = Conv_x3.unfold(2, self.Conv_ker[2], self.Conv_ker[2]).unfold(3, self.Conv_ker[2],
																			   self.Conv_ker[2]).reshape(b, self.Conv_ker[2] ** 2, -1,
																										 2 * self.virtual_dim)
		# print('x[:, :, 0].shape before module3: ', x[:, :, 0].shape)
		#print('Conv x total shape layer 3: ', x.shape)
		#print('Conv x shape layer 3: ', x[:, :, 0].shape)

		Lote_x3 = [self.LoTe_module3[i](Lote_x3[:, :, i]) for i in range(len(self.LoTe_module3))]
		Conv_x3 = [self.Conv_module3[i](Conv_x3[:, :, i]) for i in range(len(self.Conv_module3))]

		Lote_y3 = torch.stack(Lote_x3, dim=1)
		Conv_y3 = torch.stack(Conv_x3, dim=1)

		#print("Lote_y3", Lote_y3.shape)
		#print("Conv_y3", Conv_y3.shape)

		cat_x3 = torch.cat([Lote_y3, Conv_y3], dim=2)

		bn_x3 = self.BN3(cat_x3)

		# Final layer
		y3 = bn_x3.permute(0, 2, 1)
		#print('LoTe y shape before mpsfinal ', y3.shape)
		y3 = self.mpsFinal(y3)
		return y3.squeeze()