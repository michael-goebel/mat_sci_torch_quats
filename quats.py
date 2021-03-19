from math import pi

import torch
import numpy as np

# Defines mapping from quat vector to matrix. Though there are many
# possible matrix representations, this one is selected since the
# first row, X[...,0], is the vector form.
# https://en.wikipedia.org/wiki/Quaternion#Matrix_representations
q1 = np.diag([1,1,1,1])
qj = np.roll(np.diag([-1,1,1,-1]),-2,axis=1)
qk = np.diag([-1,-1,1,1])[:,::-1]
qi = np.matmul(qj,qk)
Q_arr = torch.Tensor([q1,qi,qj,qk])
Q_arr_flat = Q_arr.reshape((4,16)).float()



# Checks if 2 arrays can be broadcast together
def _broadcastable(s1,s2):
    if len(s1) != len(s2): return False
    else: return all((i==j) or (i==1) or (j==1) for i,j in zip(s1,s2))

# Converts an array of quats as vectors to matrices. Generally
# used to facilitate quat multiplication.
def vec2mat(X):
    assert X.shape[-1] == 4, 'Last dimension must be of size 4'
    new_shape = X.shape[:-1] + (4,4)
    device = X.device
    Q = Q_arr_flat.to(device)
    return torch.matmul(X,Q).reshape(new_shape)


# Performs element-wise multiplication, like the standard multiply in
# numpy. Equivalent to q1 * q2.
def hadamard_prod(q1,q2):
    assert _broadcastable(q1.shape,q2.shape), 'Inputs of shapes ' \
            f'{q1.shape}, {q2.shape} could not be broadcast together'
    X1 = vec2mat(q1)
    X_out = (X1 * q2[...,None,:]).sum(-1)
    return X_out



# Performs outer product on ndarrays of quats
# Ex if X1.shape = (s1,s2,4) and X2.shape = (s3,s4,s5,4),
# output will be of size (s1,s2,s3,s4,s5,4)
def outer_prod(q1,q2):
    X1 = vec2mat(q1)
    X2 = torch.movedim(q2,-1,0)
    X1_flat = X1.reshape((-1,4))
    X2_flat = X2.reshape((4,-1))
    X_out = torch.matmul(X1_flat,X2_flat)
    X_out = X_out.reshape(q1.shape + q2.shape[:-1])
    X_out = torch.movedim(X_out,len(q1.shape)-1,-1)
    return X_out


# Utilities to create random vectors on the L2 sphere. First produces
# random samples from a rotationally invariantt distibution (i.e. Gaussian)
# and then normalizes onto the unit sphere

# Produces random array of the same size as shape.
def rand_arr(shape):
    if not isinstance(shape,tuple): shape = (shape,)
    X = torch.randn(shape)
    X /= torch.norm(X,dim=-1,keepdim=True)
    return X

# Produces array of 3D points on the unit sphere.
def rand_points(shape):
    if not isinstance(shape,tuple): shape = (shape,)
    return rand_arr(shape + (3,))

# Produces random unit quaternions.
def rand_quats(shape,use_torch=False):
    if not isinstance(shape,tuple): shape = (shape,)
    return rand_arr(shape+(4,))


def quat_dist(q1,q2=None):
    """
    Computes distance between two quats. If q1 and q2 are on the unit sphere,
    this will return the arc length along the sphere. For points within the
    sphere, it reduces to a function of MSE.
    """
    if q2 is None: mse = (q1[...,0]-1)**2 + (q1[...,1:]**2).sum(-1)
    else: mse = ((q1-q2)**2).sum(-1)
    corr = 1 - (1/2)*mse
    assert torch.max(abs(corr)) < 1.001, "Correlation score is outside " + \
            "of [-1,1] range. Check that all inputs are inside unit ball"
    corr_clamp = torch.clamp(corr,-1,1)
    return torch.arccos(corr)

def rot_dist(q1,q2=None):
    """ Get dist between two rotations, with q <-> -q symmetry """
    q1_w_neg = torch.stack((q1,-q1),dim=-2)
    if q2 is not None: q2 = q2[...,None,:]
    dists = quat_dist(q1_w_neg,q2)
    dist_min = dists.min(-1)[0]
    return dist_min


def fz_reduce(q,syms):
    shape = q.shape
    q = q.reshape((-1,4))
    q_w_syms = outer_prod(q,syms)
    dists = rot_dist(q_w_syms)
    inds = dists.min(-1)[1]
    q_fz = q_w_syms[torch.arange(len(q_w_syms)),inds]
    q_fz *= torch.sign(q_fz[...,:1])
    q_fz = q_fz.reshape(shape)
    return q_fz


def scalar_first2last(X):
    return torch.roll(X,-1,-1)

def scalar_last2first(X):
    return torch.roll(X,1,-1)

def conj(q):
    q_out = q.clone()
    q_out[...,1:] *= -1
    return q_out


def rotate(q,points,element_wise=False):
    points = torch.as_tensor(points)
    P = torch.zeros(points.shape[:-1] + (4,)).to(q.device)
    assert points.shape[-1] == 3, 'Last dimension must be of size 3'
    P[...,1:] = points
    if element_wise:
        X_int = hadamard_prod(q,P)
        X_out = hadamard_prod(X_int,conj(q))
    else:
        X_int = outer_prod(q,P)
        inds = (slice(None),)*(len(q.shape)-1) + \
                (None,)*(len(P.shape)) + (slice(None),)
        X_out = (vec2mat(X_int) * conj(q)[inds]).sum(-1)
    return X_out[...,1:]



# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

    np.random.seed(1)
    N = 700
    M = 1000
    K = 13

    q1 = rand_quats(M)
    q2 = rand_quats(N)
    q3 = rand_quats(M)
    p1 = rand_points(K)

    p2 = rotate(q2,rotate(q1,p1))
    p3 = rotate(outer_prod(q2,q1),p1)
    p4 = rotate(conj(q1[:,None]),rotate(q1,p1),element_wise=True)


    print('Composition of rotation error:')
    err = abs(p2-p3).sum()/len(p2.reshape(-1))
    print('\t',err,'\n')

    print('Rotate then apply inverse rotation error:')
    err = abs(p4-p1).sum()/len(p1.reshape(-1))
    print('\t',err,'\n')



