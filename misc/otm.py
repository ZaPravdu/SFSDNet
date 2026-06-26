# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as tF
import numpy as np
# from simple_ot import SampleOT
EPS =  1e-12

def max_diameter(x, y):
    mins = torch.stack((x.min(dim=1)[0], y.min(dim=1)[0]), dim=1).min(dim=1)[0] # B 2
    maxs = torch.stack((x.max(dim=1)[0], y.max(dim=1)[0]), dim=1).max(dim=1)[0] # B 2
    diameter = (maxs-mins).norm(dim=1).max().item()
    if diameter == 0:
        diameter = 16
    return diameter

def epsilon_schedule(diameter, blur, scaling, fixed_epsilon=False):
    # print("[EPS]:", np.log(diameter), np.log(blur), np.log(scaling))
    schedule = np.arange(np.log(diameter), np.log(blur), np.log(scaling))
    if fixed_epsilon:
        epsilon_s = [ blur ] + [ blur for _ in  schedule] + [ blur ]
    else:
        epsilon_s = [ diameter ] + [ np.exp(e) for e in schedule ] + [ blur ]
    return epsilon_s

def dampening(epsilon, reach):
    return 1 if reach is None else 1 / ( 1 + epsilon / reach )

def softmin(logB, G, C, epsilon):
    B = C.shape[0]
    x = logB.view(B, 1, -1) + (G.view(B, 1, -1) - C) / epsilon
    x = - epsilon * x.logsumexp(2).view(B, -1, 1)
    return x

class SampleOT:
    def __init__(self, blur=0.01, scaling=0.5, reach=None, fixed_epsilon=False) -> None:
        self.blur = blur
        self.scaling = scaling
        self.fixed_epsilon = fixed_epsilon
        self.reach = reach

    @torch.no_grad()        
    def __call__(self, A, B, cost, F=None, G=None, diameter=None):
        '''
        A.shape = B H 1
        B.shape = B W 1
        cost.shape = B H W
        '''
        
        bsize, H, W = cost.shape
        
        fixed_epsilon = (F is not None and G is not None) or self.fixed_epsilon
        diameter = diameter if diameter is not None else cost.max().item()
        diameter = max(8, diameter)
        epsilons = epsilon_schedule(diameter, self.blur, self.scaling, fixed_epsilon)
        
        logA, logB = A.log(), B.log()
        Cab, Cba = cost, cost.permute(0, 2, 1)
        factor = dampening(epsilons[0], self.reach)
        if F is None:
            F = factor * softmin(logB, torch.zeros_like(B), Cab, epsilons[0])
        if G is None:
            G = factor * softmin(logA, torch.zeros_like(A), Cba, epsilons[0])
            
        for i, epsilon in enumerate(epsilons):

            factor = dampening(epsilon, self.reach)
            tF = factor * softmin(logB, G, Cab, epsilon)
            tG = factor * softmin(logA, F, Cba, epsilon)
            F, G = (F + tF) / 2, (G + tG) / 2

        factor = dampening(self.blur, self.reach)
        F, G = factor * softmin(logB, G, Cab, self.blur), factor * softmin(logA, F, Cba, self.blur)
        

        return F.detach(), G.detach()

    def loss(self, A, B, F, G):
        if self.reach is not None:
            F = self.weightfunc(1 - (- F / self.reach).exp())
            G = self.weightfunc(1 - (- G / self.reach).exp())
        return torch.mean( (A * F).flatten(1).sum(dim=1) + (B * G).flatten(1).sum(dim=1) )


    def plan(self, A, B, F, G, cost):
        PI1 = torch.exp((F + G.permute(0, 2, 1) - cost) / self.blur)
        PI2 = A * B.permute(0, 2, 1)
        PI = PI1 * PI2
        return PI

eps = 1e-12
class L2_DIS:
    factor = 1 / 32
    @staticmethod
    def __call__(X, Y):
        '''
        X.shape = (batch, M, D)
        Y.shape = (batch, N, D)
        returned cost matrix's shape is ()
        '''
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        C = ((x_col - y_row) ** 2).sum(dim=-1) / 2
        return C * L2_DIS.factor

    @staticmethod
    def barycenter(weight, coord):
        '''
        weight.shape = (batch, M, N)
        coord.shape = (batch, M, D)
        returned coord's shape is (batch, N D)
        '''
        weight = weight / (weight.sum(dim=1, keepdim=True) + eps)
        # coord = coord.to(weight.dtype)
        return weight.permute(0, 2, 1) @ coord

blur = 0.01
per_cost = L2_DIS()
ot = SampleOT(blur=blur, scaling=0.9, reach=None, fixed_epsilon=False)

def den2coord(denmap, scale_factor=8):
    coord = torch.nonzero(denmap > eps)
    denval = denmap[coord[:, 0], coord[:, 1]]
    if scale_factor != 1:
        coord = coord.float() * scale_factor + scale_factor / 2
    return denval.reshape(1, -1, 1), coord.reshape(1, -1, 2).float()

def init_dot(denmap, n, scale_factor=8):

    norm_den = denmap[None, None, ...]
    norm_den = tF.interpolate(norm_den, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    norm_den = norm_den[0, 0]

    d_coord = torch.nonzero(norm_den > eps)
    norm_den = norm_den[d_coord[:, 0], d_coord[:, 1]]

    cidx = torch.multinomial(norm_den, num_samples=n, replacement=False)
    coord = d_coord[cidx]
    
    B = torch.ones(1, n, 1).to(denmap)
    B_coord = coord.reshape(1, n, 2)
    return B, B_coord

@torch.no_grad()
def OT_M(A, A_coord, B, B_coord, scale_factor=8, max_itern=8):
    for iter in range(max_itern):
        # OT-step
        C = per_cost(A_coord, B_coord)
        F, G = ot(A, B, C)
        PI = ot.plan(A, B, F, G, C)
        # M-step
        nB_coord = per_cost.barycenter(PI, A_coord)
        move = torch.norm(nB_coord - B_coord, p=2, dim=-1)
        if move.mean().item() < 1 and move.max().item() < scale_factor:
            break
        B_coord = nB_coord
    print(iter)
    return (nB_coord).reshape(-1, 2)

@torch.no_grad()
def den2seq(denmap, scale_factor=1, max_itern=16, ot_scaling=0.75):
    ot.scaling = ot_scaling
    assert denmap.dim() == 2, f"the shape of density map should be [H, W], but the given one is {denmap.shape}"
    
    num = int(denmap.sum().item() + 0.5)
    if num < 0.5:
        return torch.zeros((0, 2)).to(denmap)

    # normalize density map
    denmap = denmap * num / denmap.sum()
    
    A, A_coord = den2coord(denmap, scale_factor)
    B, B_coord = init_dot(denmap, num, scale_factor)

    flocs = OT_M(A, A_coord, B, B_coord, scale_factor, max_itern=max_itern)
    return flocs.long()

@torch.no_grad()
def main():
    import cv2
    import os
    import matplotlib.pyplot as plt
    datadir = 'samples'
    imlist = [14, 77]
    for idx in imlist:
        img = cv2.imread(os.path.join(datadir, f'IMG_{idx}.jpg'))
        imh, imw = img.shape[:2]
        denmap = torch.load(os.path.join(datadir, f"{idx}.pth"))
        dh, dw = denmap.shape
        scale_factor = imw / dw
        print(img.shape, denmap.shape, scale_factor)
        plt.imsave(f"denmap{idx}.png", denmap.cpu(), cmap='jet')
        dot = den2seq(denmap, scale_factor)
        
        # the output's axis is (h, w)
        dot_coord = dot.long().cpu()
        dotmap = torch.zeros((imh, imw))
        dotmap[dot_coord[:, 0], dot_coord[:, 1]] = 1
        dotmap = tF.conv2d(dotmap[None, None, ...], torch.ones((1, 1, 5, 5)), padding=2)[0, 0]
        
        
        plt.imsave(f"dotmap{idx}.png", dotmap)
    
if __name__ == '__main__':
    main()