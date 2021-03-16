import torch
from quats import rand_quats, outer_prod

# Distance functions: get distance between q1 and q2. If q2 is unspecified,
# then q2 is assumed to be the identity quat <1,0,0,0>.
# Note that quat_dist(q1*q2.conjugate()) = quat_dist(q1,q2), but the second
# formula will be more efficient.

# Get distance between two sets of quats in radians

def quat_dist(q1,q2=None):
    if q2 is None: mse = (q1[...,0]-1)**2 + (q1[...,1:]**2).sum(-1)
    else: mse = ((q1-q2)**2).sum(-1)
    corr = 1 - (1/2)*mse
    assert torch.max(abs(corr)) < 1.001, "Correlation score is outside " + \
            "of [-1,1] range. Check that all inputs are inside unit ball"
    corr_clamp = torch.clamp(corr,-1,1)
    return torch.arccos(corr)

# Get distance between two sets rotations, accounting for q <-> -q symmetry

def rot_dist(q1,q2=None):
    q1_w_neg = torch.stack((q1,-q1),dim=-2)
    if q2 is not None: q2 = q2[...,None,:]
    dists = quat_dist(q1_w_neg,q2)
    dist_min = dists.min(-1)[0]
    return dist_min

def l1(q1,q2):
    return torch.mean(abs(q1-q2),dim=-1)

def l2(q1,q2):
    return torch.sqrt(torch.mean((q1-q2)**2,dim=-1))


class Loss:
    def __init__(self,dist_func,syms=None):
        self.dist_func = dist_func
        self.syms = syms
    def __call__(self,q1,q2):
        if self.syms is not None:
            q1_w_syms = outer_prod(q1,self.syms)
            if q2 is not None: q2 = q2[...,None,:]
            dists = self.dist_func(q1,q2)
            dist_min = dists.min(-1)[0]
        else:
            return self.dist_func(q1,q2)
    
    def __str__(self):
        return f'Dist -> dist_func: {self.dist_func}, ' + \
               f'syms: {self.syms is not None}'


def tanh_act(q):
    return q*tanhc(torch.norm(q,dim=-1,keepdim=True))
    
def safe_divide_act(q,eps=10**-5):
    return q/(eps+torch.norm(q,dim=-1,keepdim=True))


class ActAndLoss:
    def __init__(self,act,loss):
        self.act = act
        self.loss = loss
    def __call__(self,X,labels):
        X_act = X if self.act is None else self.act(X)
        return self.loss(X_act,labels)
    def __str__(self):
        return f'Act and Loss: ({self.act},{self.loss})'



# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

    from symmetries import hcp_syms


    torch.manual_seed(1)
    M = 10

    q1 = rand_quats(M)
    q2 = rand_quats(M)

    acts_and_losses = list()
    
    for act in [None,tanh_act,safe_divide_act]:
        for syms in [None,hcp_syms]:
            for dist in [l1,l2,rot_dist]:
                acts_and_losses.append(ActAndLoss(act,Loss(dist,syms)))
    

    for i in acts_and_losses:
        print(i)
        print(i(q1,q2))


