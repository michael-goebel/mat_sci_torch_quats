import sys
sys.path.append('../')

from quats import rand_quats, fz_reduce, rot_dist
from symmetries import hcp_syms
import matplotlib.pyplot as plt
from time import time
import numpy as np

N = 10**8
n_bins = 500
theta_max = 1.7
#theta_max = 6

bins = np.linspace(0,theta_max,n_bins+1)

t1 = time()

q1 = rand_quats(N)

q1_fz = fz_reduce(q1,hcp_syms)
#q1_fz = q1

d = rot_dist(q1_fz)

h,_ = np.histogram(d.numpy(),bins=bins)

bin_centers = (bins[:-1] + bins[1:])/2

width = theta_max/n_bins

h = h.astype('float')/h.sum()/width

print(time()-t1)

fig, axes = plt.subplots(1,1, figsize=(4,4))

axes.bar(bin_centers,h,width=width)


axes.set_xlabel('Misorientation (Radians)')
axes.set_ylabel('Density Distribution Function')
fig.tight_layout()

plt.show()


