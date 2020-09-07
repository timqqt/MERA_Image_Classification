import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.mps import MPS
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = 1e-6


class MERAnet_clean(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
                 kernel=[2, 2, 2], virtual_dim=1,
                 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
                 label_site=None, path=None, init_std=1e-9, use_bias=True,
                 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
        # bond_dim parameter is non-sense
        super().__init__()
        self.input_dim = input_dim
        self.virtual_dim = bond_dim

        ### Squeezing of spatial dimension in first step
        # self.kScale = 4  # what is this?
        # nCh = self.kScale ** 2 * nCh
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
            # feature_dim = 2 * nCh
            feature_dim = 2 * nCh
            # print(feature_dim)
            # First level disentanglers
            # First level isometries

            ### First level MERA blocks
            self.disentangler_list.append(nn.ModuleList([MPS(input_dim=4,
                                                             output_dim=4,
                                                             nCh=nCh, bond_dim=4,
                                                             feature_dim=feature_dim, parallel_eval=parallel_eval,
                                                             adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
                                                         for i in range(torch.prod(iDim))]))

            iDim = iDim + 1

            self.isometry_list.append(nn.ModuleList([MPS(input_dim=4,
                                                         output_dim=1,
                                                         nCh=nCh, bond_dim=bond_dim,
                                                         feature_dim=feature_dim, parallel_eval=parallel_eval,
                                                         adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
                                                     for i in range(torch.prod(iDim))]))
            iDim = (iDim - 2) / 2

        ### Final MPS block
        self.mpsFinal = MPS(input_dim=49,
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
            # print('x_ent unfold shape: ',  x_ent.unfold(2, 2, 2).unfold(3, 2, 2).shape)
            x_ent = x_ent.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
            # print('x_ent unfold->reshape shape: ',  x_ent.shape)
            # print('single x_ent unfold->reshape shape: ',  x_ent[:, :, 0].shape)
            # print(x_ent.shape)
            # print(len(self.disentangler_list[jj]))
            y_ent = [self.disentangler_list[jj][i](x_ent[:, :, i]) for i in range(len(self.disentangler_list[jj]))]
            y_ent = torch.stack(y_ent, dim=1)  # 512, 3969, 4
            y_ent = y_ent.view(y_ent.shape[0], y_ent.shape[1], 2, 2)
            y_ent = y_ent.view(y_ent.shape[0], iDim[0] - 1, iDim[1] - 1, 2, 2)
            y_ent_list = []
            # print('y_ent shape: ', y_ent.shape)
            # print('torch cat col shape: ', torch.cat([y_ent[:, 0, i, :, :] for i in range(y_ent.shape[2])], dim=2).shape)
            for j in range(y_ent.shape[1]):
                y_ent_list.append(torch.cat([y_ent[:, j, i, :, :] for i in range(y_ent.shape[2])], dim=2))
            y_ent = torch.cat(y_ent_list, dim=1)

            # print('y_ent shape: ', y_ent.shape)
            x_iso = x_in
            # print(x_iso.shape)
            x_iso[:, :, 1:-1, 1:-1] = y_ent.view(b, self.nCh, y_ent.shape[1], y_ent.shape[2])
            # print('x_iso shape: ', x_iso.shape)
            x_iso = x_iso.unfold(2, 2, 2).unfold(3, 2, 2).reshape(b, self.nCh, -1, 4)
            y_iso = [self.isometry_list[jj][i](x_iso[:, :, i]) for i in range(len(self.isometry_list[jj]))]
            y_iso = torch.stack(y_iso, dim=1)  # 512, 4096, 1

            x_in = y_iso.view(y_iso.shape[0], self.nCh, iDim[0], iDim[1])

        # print('x6 shape: ', x6.shape) # 512, 1, 2, 2

        y = x_in.view(b, self.nCh, iDim[0] * iDim[1])
        # print('LoTe y shape before mpsfinal ', y.shape)
        y = self.mpsFinal(y)
        return y.squeeze()


