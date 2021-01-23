from quats import Quat, rot_dist_w_syms, rand_quats
from symmetries import hcp_syms, fcc_syms

import matplotlib.pyplot as plt
from time import time

N = 3000000
n_bins = 100


use_torch = False


q1 = rand_quats(N)
q2 = rand_quats(N)

t1 = time()
dists_hcp = rot_dist_w_syms(q1,q2,hcp_syms)
dists_fcc = rot_dist_w_syms(q1,q2,fcc_syms)
print(f'{time()-t1:0.5f} seconds to compute {2*N} misorientations')

fig,axes = plt.subplots(1,2)

axes[0].hist(dists_hcp,bins=n_bins)
axes[1].hist(dists_fcc,bins=n_bins)

axes[0].set_title('HCP')
axes[1].set_title('FCC')

for a in axes: a.set_xlabel('radians')


#plt.hist(dists_hcp,bins=n_bins)

plt.show()










