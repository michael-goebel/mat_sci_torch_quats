import torch
from math import pi
import numpy as np


# There are many possible matrix representations for quats.
# This one was selected so that the column X[...,0] is the
# quat in vecotr form.
q1 = np.diag([1,1,1,1])
qj = np.roll(np.diag([-1,1,1,-1]),-2,axis=1)
qk = np.diag([-1,-1,1,1])[:,::-1]
qi = np.matmul(qj,qk)

Q_arr = np.array([q1,qi,qj,qk])


Q_arr_flat_np = Q_arr.reshape((4,16))
Q_arr_flat_torch = torch.as_tensor(Q_arr_flat_np).float()


def norm(X):
    if is_np(X): return np.linalg.norm(X,axis=-1,keepdims=True)
    else: return torch.norm(X,dim=-1,keepdim=True)

def arccos(X):
    return np.arccos(X) if is_np(X) else torch.arccos(X)

def moveaxis(X,src,dst):
    return np.moveaxis(X,src,dst) if is_np(X) else torch.movedim(X,src,dst)

def array_copy(X):
    return np.copy(X) if is_np(X) else X.clone()

def matmul(X1,X2):
    if is_np(X1) and is_np(X2):
        return np.matmul(X1,X2)
    elif not is_np(X1) and not is_np(X2):
        return torch.matmul(X1,X2)
    else:
        str_types = ['numpy' if is_np(X) else 'torch' for X in [X1,X2]]
        raise Exception(f'X1 is {str_types[0]} and X2 is {str_types[1]}')


def is_np(X): return isinstance(X,np.ndarray)


def broadcastable(s1,s2):
    if len(s1) != len(s2):
        return False
    else:
        return all((i==j) or (i==1) or (j==1) for i,j in zip(s1,s2))


def vec2mat(X):
    assert X.shape[-1] == 4, 'Last dimension must be of size 4'
    new_shape = X.shape[:-1] + (4,4)
    if is_np(X): return np.matmul(X,Q_arr_flat_np).reshape(new_shape)
    else: return torch.matmul(X,Q_arr_flat_torch).reshape(new_shape)


def hadamard_prod(q1,q2):
    assert broadcastable(q1.shape,q2.shape), f'Inputs of shapes {q1.shape}, {q2.shape} could not be broadcast together'

    X1 = vec2mat(q1.X)
    X_out = (X1 * q2.X[...,None,:]).sum(-1)
    return Quat(X_out)



# Performs outer product on ndarrays of quats
# Ex if X1.shape = (s1,s2,4) and X2.shape = (s3,s4,s5,4),
# output will be of size (s1,s2,s3,s4,s5,4)
def outer_prod(q1,q2):
    X1 = vec2mat(q1.X)
    X2 = moveaxis(q2.X,-1,0)
    X1_flat = X1.reshape((-1,4))
    X2_flat = X2.reshape((4,-1))
    X_out = matmul(X1_flat,X2_flat)
    X_out = X_out.reshape(q1.X.shape + q2.X.shape[:-1])
    X_out = moveaxis(X_out,len(q1.X.shape)-1,-1)

    return Quat(X_out)


def rand_arr(shape,use_torch):
    if not isinstance(shape,tuple): shape = (shape,)
    if use_torch: X = torch.randn(shape)
    else: X = np.random.standard_normal(shape)
    X /= norm(X)
    return X

def rand_points(shape,use_torch=False):
    if not isinstance(shape,tuple): shape = (shape,)
    return rand_arr(shape + (3,),use_torch)

def rand_quats(shape,use_torch=False):
    if not isinstance(shape,tuple): shape = (shape,)
    return Quat(rand_arr(shape+(4,),use_torch))


# get distance between two sets of quats in radians
def quat_dist(q1,q2=None):
    if q2 is None: corr = q1.X[...,0]
    else: corr = (q1.X*q2.X).sum(-1)
    return arccos(corr)

# get distance between two sets rotations, accounting
# for q <-> -q equivalence
def rot_dist(q1,q2=None):
    dq = quat_dist(q1,q2)
    return pi - abs(2*dq - pi)    

def rot_dist_w_syms(q1,q2,syms):
    q1_w_syms = q1.outer_prod(syms)
    if q2 is not None: q2 = q2[...,None]
    dists = rot_dist(q1_w_syms,q2)
    if is_np(dists): dist_min = dists.min(-1)
    else: dist_min = dists.min(-1)[0]
    return dist_min

class Quat:
    def __init__(self,X,use_torch=False):
        if not use_torch and not isinstance(X,torch.Tensor):
            self.X = np.asarray(X)
        else: self.X = torch.as_tensor(X)
        assert self.X.shape[-1] == 4, 'Last dimension must be of size 4'
        self.shape = self.X.shape[:-1]

    def __add__(self,q2):
        return Quat(self.X+q2.X)

    def __sub__(self,q2):
        return Quat(self.X-q2.X)

    def __mul__(self,q2):
        return hadamard_prod(self,q2)

    def outer_prod(self,q2):
        return outer_prod(self,q2)

    def __str__(self):
        return str(self.X)

    def __getitem__(self,index):
        if isinstance(index,tuple): index = index + (slice(None),)
        else: index = (index,slice(None))
        return Quat(self.X[index])

    def to_numpy(self): return Quat(self.X.numpy())

    def to_torch(self): return Quat(torch.as_tensor(self.X).float())

    def conjugate(self):
        X_out = array_copy(self.X)
        X_out[...,1:] *= -1
        return Quat(X_out)


    def reshape(self,axes):
        if isinstance(axes,tuple): return Quat(self.X.reshape(axes + (4,)))
        else: return Quat(self.X.reshape((axes,4)))

    def transpose(self,axes):
        assert min(axes) >= 0
        return Quat(self.X.transpose(axes+(-1,)))


    def rotate(self,points,element_wise=False):
        if is_np(self.X):
            points = np.asarray(points)
            P = np.zeros(points.shape[:-1] + (4,))
        else:
            points = torch.as_tensor(points)
            P = torch.zeros(points.shape[:-1] + (4,))
        assert points.shape[-1] == 3, 'Last dimension must be of size 3'

        P[...,1:] = points
        qp = Quat(P)

        if element_wise:
            X_out = (self * qp * self.conjugate()).X
        else:
            X_int = self.outer_prod(qp)
            inds = (slice(None),)*(len(self.X.shape)-1) + \
                    (None,)*(len(qp.X.shape)) + (slice(None),)
            X_out = (vec2mat(X_int.X) * self.conjugate().X[inds]).sum(-1)
        return X_out[...,1:]



if __name__ == '__main__':

    np.random.seed(1)

    N = 7
    M = 11
    K = 13

    for i in range(2):

        use_torch = bool(i)
        if i == 0:
            print('using numpy (default float64)')
        else: 
            print('using torch (default float32)')

        q1 = rand_quats(M,use_torch)
        q2 = rand_quats(N,use_torch)
        p1 = rand_points(K,use_torch)

        p2 = q2.rotate(q1.rotate(p1))
        p3 = q2.outer_prod(q1).rotate(p1)
        p4 = q1.conjugate()[:,None].rotate(q1.rotate(p1),element_wise=True)

        print('Composition of rotation error:')
        err = abs(p2-p3).sum()/np.prod(p2.shape)
        print('\t',err,'\n')

        print('Rotate then apply inverse rotation error:')
        err = abs(p4-p1).sum()/np.prod(p4.shape)
        print('\t',err,'\n')

