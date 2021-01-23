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
#Q_arr_flat = Q_arr.reshape((4,16))


def use_numpy(): set_arr_type('n')
def use_torch(): set_arr_type('t')

def set_arr_type(t):
    n = t = 'n'
    global asarray, matmul, moveaxis, randn, norm, arccos, \
            array_copy, zeros, Q_arr_flat
    asarray = np.asarray if n else torch.as_tensor
    matmul = np.matmul if n else torch.matmul
    moveaxis = np.moveaxis if n else torch.movedim
    randn = np.random.standard_normal if n else torch.randn
    norm = lambda x: np.linalg.norm(x,axis=-1,keepdims=True) if n else \
            torch.norm(x,dim=-1,keepdims=True)
    arccos = np.arccos if n else torch.arccos
    array_copy = np.copy if n else torch.clone
    zeros = np.zeros if n else torch.zeros
    Q_arr_flat = asarray(Q_arr.reshape((4,16)))
    


def broadcastable(s1,s2):
    if len(s1) != len(s2):
        return False
    else:
        return all((i==j) or (i==1) or (j==1) for i,j in zip(s1,s2))


def vec2mat(X):
    assert X.shape[-1] == 4, 'Last dimension must be of size 4'
    new_shape = X.shape[:-1] + (4,4)
    return matmul(X,Q_arr_flat).reshape(new_shape)


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



def quat_rand(shape):
    if not isinstance(shape,tuple): shape = (shape,)
    X = randn(shape + (4,))
    X /= norm(X)
    return Quat(X)


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


class Quat:
    def __init__(self,X):
        self.X = asarray(X)
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
        return Quat(self.X[index])

    def conjugate(self):
        #X_out = self.X.clone()
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
        #points = np.asarray(points)
        points = asarray(points)
        assert points.shape[-1] == 3, 'Last dimension must be of size 3'

        P = zeros(points.shape[:-1] + (4,))
        P[...,1:] = points
        qp = Quat(P)

        if element_wise:
            X_out = (self * qp * self.conjugate()).X

        else:
            X_int = self.outer_prod(qp)
            inds = (slice(None),)*(len(self.X.shape)-1) + (None,)*(len(qp.X.shape)) + (slice(None),)
            X_out = (vec2mat(X_int.X) * self.conjugate().X[inds]).sum(-1)

        return X_out[...,1:]


# default to using numpy
use_numpy()

if __name__ == '__main__':

    np.random.seed(1)

    N = 7
    M = 11
    K = 13

    for i in range(2):

        if i == 0:
            print('using numpy')
            use_numpy()
        else: 
            print('using torch')
            use_torch()


        q1 = quat_rand(M)
        q2 = quat_rand(N)

        p1 = randn((K,3))

        p2 = q2.rotate(q1.rotate(p1))
        p3 = q2.outer_prod(q1).rotate(p1)
        p4 = q1.conjugate()[:,None].rotate(q1.rotate(p1),element_wise=True)

        print('Composition of rotation error:')
        err = norm(p2-p3).sum()


        print('\t',err,'\n')

        print('Rotate then apply inverse rotation error:')
        err = norm(p4-p1).sum()

        print('\t',err,'\n')


