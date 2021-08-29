import sys
sys.path.append('../')

from quats import rand_quats, fz_reduce, rot_dist
from symmetries import hcp_syms
import matplotlib.pyplot as plt
from time import time
import numpy as np
from losses import l1
    

N = 10**6
n_bins = 500
theta_max = 3.5
#theta_max = 6

def l1_dist(x):
    delta = x.clone()
    delta[:,0] -= 1
    return abs(delta).sum(-1)

bins = np.linspace(0,theta_max,n_bins+1)

t1 = time()

q1 = rand_quats(N)

func_list = [
        lambda x: rot_dist(fz_reduce(x, hcp_syms)),
        lambda x: l1_dist(x),
        lambda x: l1_dist(fz_reduce(x, hcp_syms)),
]


bin_centers = (bins[:-1] + bins[1:])/2

width = theta_max/n_bins

fig, axes = plt.subplots(3,1, figsize=(4,6))

titles = [
        'Rot dist with Symmetries',
        'L1 Dist',
        'L1 with Symmetries',
]

for i in range(3):
    print(i)
    d = func_list[i](q1)
    h,_ = np.histogram(d.numpy(),bins=bins)
    h = h.astype('float')/h.sum()/width
    axes[i].bar(bin_centers,h,width=width)
    axes[i].set_title(titles[i])

    #axes[i].set_xlabel('Misorientation (Radians)')
    #axes[i].set_ylabel('PDF')



fig.tight_layout()

plt.show()


