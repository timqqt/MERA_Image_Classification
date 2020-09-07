### Adapted from https://github.com/bamos/densenet.pytorch

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import truncnorm

import torchvision.models as models

import sys
import math
import pdb

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class FrBottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, dup=2):
        super(FrBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.Fr1 = frTNLayer2(nChannels, dup=dup)
        interChannels = nChannels // dup
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.Fr1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


def get_truncated_normal(mean=0, sd=1, low=-2, upp=2):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class frTNLayer(nn.Module):
    def __init__(self, nChannels, V=False, P=True, H=True, S=True,W=True, dup=2, ratio=6, z_bias=1.0):
        super(frTNLayer, self).__init__()
        self.flag_V = V
        self.flag_P = P
        self.flag_H = H
        self.flag_S = S
        self.flag_W = W
        self.nChannels = nChannels
        self.dup = dup
        self.ratio = 6

        self.z_bias = z_bias

    def forward(self, x):
        #print('x_shape', x.shape)
        if self.flag_V:
            mean, variance = x.view(x.shape[0], x.shape[1], -1).mean(2), x.view(x.shape[0], x.shape[1], -1).std(2)
            mv = torch.cat([mean, variance], 1)  # B x 2C
            coff = 2
        else:
            mv = x.view(x.shape[0], x.shape[1], -1).mean(2)
            coff = 1
        α = Variable(torch.zeros(1), requires_grad=True)
        α = torch.clamp(α, -1.0, 5.0)
        α = α.cuda()
        if self.flag_P:
            x = torch.nn.functional.relu(x) + self.z_bias
            z = torch.pow(x, α + 1)
        else:
            y = torch.abs(x)
            z = torch.pow(y, α + 1) * torch.sign(x)
        if self.flag_W:
            size_wa =  self.nChannels * coff *  self.nChannels * coff // self.ratio
            wx = get_truncated_normal()
            fcwa = wx.rvs(size_wa)
            fcwa = np.reshape(fcwa, [self.nChannels * coff, self.nChannels * coff // self.ratio])
            fc_weights_a = Variable(torch.from_numpy(fcwa), requires_grad=True)

            size_wb = self.nChannels * coff * self.nChannels * coff // self.ratio
            wx = get_truncated_normal()
            fcwb = wx.rvs(size_wb)
            fcwb = np.reshape(fcwb, [self.nChannels * coff, self.nChannels * coff // self.ratio])
            fc_weights_b = Variable(torch.from_numpy(fcwb), requires_grad=True)

            fc_weights_a = fc_weights_a.cuda().float()
            fc_weights_b = fc_weights_b.cuda().float()
            mv = mv.cuda().float()
            # print(fc_weights_a.shape)
            # print(fc_weights_b.shape)
            # print(mv.shape)

            ω = torch.nn.functional.sigmoid(torch.matmul(torch.nn.functional.relu(torch.matmul(mv, fc_weights_a)), fc_weights_b.transpose(0,1)))
            ω = ω.view(-1, self.nChannels, 1, 1)
            #print(z.shape)
            #print(ω.shape)
            z = z * ω
        z = z.view(-1, self.dup, self.nChannels // self.dup, x.shape[2], x.shape[3])
        z = z.sum(1)

        return z

class frTNLayer2(nn.Module):
    def __init__(self, nChannels, V=True, P=True, H=True, S=True,W=True, dup=2, ratio=6, z_bias=1.0):
        super(frTNLayer2, self).__init__()
        self.flag_V = V
        self.flag_P = P
        self.flag_H = H
        self.flag_S = S
        self.flag_W = W
        self.nChannels = nChannels
        self.dup = dup
        self.ratio = 6

        self.z_bias = z_bias

    def forward(self, x):
        #print('x_shape', x.shape)
        if self.flag_V:
            mean, variance = x.view(x.shape[0], x.shape[1], -1).mean(2), x.view(x.shape[0], x.shape[1], -1).std(2)
            mv = torch.cat([mean, variance], 1)  # B x 2C
            coff = 2
        else:
            mv = x.view(x.shape[0], x.shape[1], -1).mean(2)
            coff = 1

        if self.flag_W:
            size_wa = self.nChannels * coff * self.nChannels * coff // self.ratio
            wx = get_truncated_normal(mean=0, sd=0.05)
            fcwa = wx.rvs(size_wa)
            fcwa = np.reshape(fcwa, [self.nChannels * coff, self.nChannels * coff // self.ratio])
            fc_weights_a = Variable(torch.from_numpy(fcwa), requires_grad=True)

            size_wb = self.nChannels * coff * self.nChannels * coff // self.ratio
            wx = get_truncated_normal(sd=0.002)
            fcwb = wx.rvs(size_wb)
            fcwb = np.reshape(fcwb, [self.nChannels * coff, self.nChannels * coff // self.ratio])
            fc_weights_b = Variable(torch.from_numpy(fcwb), requires_grad=True)
            fc_weights_a = fc_weights_a.cuda().float()
            fc_weights_b = fc_weights_b.cuda().float()
            mv = mv.cuda().float()
            if self.flag_S:
                # print(mv.shape)
                # print(fc_weights_a.shape)
                # print(fc_weights_b.shape)
                η =torch.nn.functional.sigmoid(
                torch.matmul(torch.nn.functional.relu(torch.matmul(mv, fc_weights_a)), fc_weights_b.transpose(0, 1)))
                #print(η.shape)
                α, ω = torch.split(η, η.shape[1]//2, dim=1)
                α, ω = α, torch.nn.functional.sigmoid(ω)
                # print(self.nChannels)
                # print(α.shape)
                # print(ω.shape)
                α = α.view(-1, self.nChannels, 1, 1)
                ω = ω.view(-1, self.nChannels, 1, 1)
                α, ω = α.view(-1, self.nChannels, 1, 1), ω.view(-1, self.nChannels, 1, 1)
                α = torch.clamp(α, -0.5, 2.0)
                α = α.cuda().float()
                ω = ω.cuda().float()



        if self.flag_P:
            x = torch.nn.functional.relu(x) + self.z_bias
            z = torch.pow(x, α + 1)
        else:
            y = torch.abs(x)
            z = torch.pow(y, α + 1) * torch.sign(x)
        z = z * ω
        z = z.view(-1, self.dup, self.nChannels // self.dup, x.shape[2], x.shape[3])
        z = z.sum(1)

        return z

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        self.nChannels = nChannels
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
#        pdb.set_trace()
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = out.view(x.shape[0],self.nChannels,-1).mean(2)
        out = torch.sigmoid(self.fc(out))
        return out.squeeze()

class FrDenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(FrDenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        self.nChannels = nChannels
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(FrBottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
#        pdb.set_trace()
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = out.view(x.shape[0],self.nChannels,-1).mean(2)
        out = torch.sigmoid(self.fc(out))
        return out.squeeze()
