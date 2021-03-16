import torch
from quats import rand_quats, outer_prod


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

def l1(q1,q2):
    """ Basic L1 loss """
    return torch.mean(abs(q1-q2),dim=-1)

def l2(q1,q2):
    """ Basic L1 loss """
    return torch.sqrt(torch.mean((q1-q2)**2,dim=-1))


class Loss:
    """ Wrapper for loss. Inclues option for symmetry as well """
    def __init__(self,dist_func,syms=None):
        self.dist_func = dist_func
        self.syms = syms
    def __call__(self,q1,q2):
        if self.syms is not None:
            q1_w_syms = outer_prod(q1,self.syms)
            if q2 is not None: q2 = q2[...,None,:]
            dists = self.dist_func(q1,q2)
            dist_min = dists.min(-1)[0]
            return dist_min
        else:
            return self.dist_func(q1,q2)
    
    def __str__(self):
        return f'Dist -> dist_func: {self.dist_func}, ' + \
               f'syms: {self.syms is not None}'


def tanhc(x):
    """
    Computes tanh(x)/x. For x close to 0, the function is defined, but not
    numerically stable. For values less than eps, a taylor series is used.
    """
    eps = 0.05
    mask = (torch.abs(x) < eps).float()
    # clip x values, to plug into tanh(x)/x
    x_clip = torch.clamp(abs(x),min=eps)
    # taylor series evaluation
    output_ts = 1 - (x**2)/3 + 2*(x**4)/15 - 17*(x**6)/315
    # regular function evaluation for tanh(x)/x
    output_ht = torch.tanh(x_clip)/x_clip
    # use taylor series if x is close to 0, otherwise, use tanh(x)/x
    output = mask*output_ts + (1-mask)*output_ht
    return output


def tanh_act(q):
    """ Scale a vector q such that ||q|| = tanh(||q||) """
    return q*tanhc(torch.norm(q,dim=-1,keepdim=True))
    
def safe_divide_act(q,eps=10**-5):
    """ Scale a vector such that ||q|| ~= 1 """
    return q/(eps+torch.norm(q,dim=-1,keepdim=True))


class ActAndLoss:
    """ Wraps together activation and loss """
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

    q1.requires_grad = True

    acts_and_losses = list()
    
    for act in [None,tanh_act,safe_divide_act]:
        for syms in [None,hcp_syms]:
            for dist in [l1,l2,rot_dist]:
                acts_and_losses.append(ActAndLoss(act,Loss(dist,syms)))
    

    for i in acts_and_losses:
        print(i)
        d = i(q1,q2)
        L = d.sum()



