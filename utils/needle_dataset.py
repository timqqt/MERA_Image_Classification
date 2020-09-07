# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import torch
import argparse
import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import numpy as np
import os
from torch.utils.data.dataset import Dataset
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from scipy.ndimage import zoom
import pandas as pd
import numpy as np
import os
from torch.utils.data.dataset import Dataset

class ClutteredMNISTDataset(Dataset):
    reg_dataset_size = 11276

    def __init__(self, base_path, csv_path, data_scaling=1., num_examples=None, balance=None):
        self.base_path = base_path
        self.csv_path = csv_path
        self.csv = pd.read_csv(csv_path)
        self.data_scaling = data_scaling
        self.num_examples = num_examples
        self.balance = balance

        self.img_paths = self.csv['img_path'].values
        self.lbls = self.csv['label'].values.astype(np.int32)
        self.weights = np.ones([len(self.img_paths), ])

        if self.num_examples is not None and self.balance is not None:
            # only rebalance if examples and balance is given
            assert self.num_examples <= len(self.img_paths), \
                'not enough examples in dataset {} - {}'.format(self.num_examples, len(self.img_paths))

            pos_num = int(self.balance * self.num_examples)
            neg_num = self.num_examples - pos_num

            pos_mask = (self.lbls == 1)
            pos_paths = self.img_paths[pos_mask][:pos_num]

            neg_mask = (self.lbls == 0)
            neg_paths = self.img_paths[neg_mask][:neg_num]

            self.img_paths = np.concatenate([pos_paths, neg_paths], 0)
            self.lbls = np.concatenate([np.ones([pos_num, ]), np.zeros([neg_num, ])], 0)
            self.weights = np.ones([self.num_examples, ])
            self.weights[:pos_num] /= pos_num
            self.weights[pos_num:] /= neg_num

        self.shrinkage = self.reg_dataset_size // len(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = os.path.join(self.base_path, self.img_paths[index])
        lbl = self.lbls[index].astype(np.int64)

        img = np.load(path)
        if isinstance(img, np.lib.npyio.NpzFile):
            img = img['arr_0']

        if self.data_scaling != 1.:
            img = zoom(img, self.data_scaling)
            img = img.clip(0., 1.)

        img = img[np.newaxis].astype(np.float32)

        return img, lbl

def NeedleMNIST(data_path, data_scaling, data_balance, batch_size, num_examples):
    dataset = ClutteredMNISTDataset

    train_dataset = dataset(data_path, os.path.join(data_path, 'train.csv'),
                            data_scaling, num_examples, data_balance)

    weights = torch.Tensor(train_dataset.weights)
    # over and undersamples..
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, sampler=sampler, pin_memory=True)

    val_dataset = dataset(data_path, os.path.join(data_path, 'val.csv'), data_scaling)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    test_dataset = dataset(data_path, os.path.join(data_path, 'test.csv'), data_scaling)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    return train_loader, val_loader, test_loader






