import torch
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
from quats import approx_rot_dist


def rot_dist(q1,q2):
    d_euclid = torch.norm(q1-q2,dim=-1)
    d_rot = 4*torch.asin(d_euclid/2)
    return d_rot


N = 1000

q1 = torch.zeros(N,4)
q2 = torch.zeros(N,4)

q1[:,0] = 1

x_range = torch.linspace(0,2,N+2)[1:-1]
x_range.requires_grad = True

q2[:,0] = 1 + x_range

d = rot_dist(q1,q2)
d.sum().backward(retain_graph=True)


fig, axes = plt.subplots(1,2, figsize=(4,4))

axes[0].plot(x_range.detach().numpy(),d.detach().numpy())
axes[1].plot(x_range.detach().numpy(), x_range.grad.detach().numpy().copy(),label='Rot distance')

q1 = torch.zeros(N,4)
q2 = torch.zeros(N,4)

q1[:,0] = 1

x_range = torch.linspace(0,2,N+1)[:-1]
x_range.requires_grad = True

q2[:,0] = 1 + x_range

d2 = approx_rot_dist(q1,q2)

d2.sum().backward()


axes[0].plot(x_range.detach().numpy(),d2.detach().numpy())
axes[1].plot(x_range.detach().numpy(), x_range.grad.detach().numpy(),label='Approx Dist')


axes[1].set_ylim(0,10)

axes[0].set_xlabel('Euclidean Distance\n(' + r'$d_{euclid}$' + ')')
axes[1].set_xlabel('Euclidean Distance\n(' + r'$d_{euclid}$' + ')')


#axes[0].set_ylabel('Angle (Radians)')
#axes[1].set_ylabel('Angle (Radians)')

axes[0].set_ylabel(r'$\theta$' + ' (Radians)')
axes[1].set_ylabel(r'$\frac{\partial \theta}{\partial d_{euclid}}$' + ' (Radians)')

axes[0].set_title('Function')
axes[1].set_title('Derivative')


plt.legend(bbox_to_anchor=(0.00,1.1), loc='lower left')

#plt.legend(bbox_to_anchor=(0,0), loc='upper left')


#plt.tight_layout(rect=[0,0,0.75,1])
plt.tight_layout()

plt.show()
plt.savefig('foo.png')



