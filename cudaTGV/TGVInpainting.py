import math
from torch import nn
from torch.autograd import Function
import torch

#import torch.optim as opt
#import torch.nn.functional as F
import numpy as np

# see setup.py : CUDAExtension('tgvInpaint_cuda_ext', [ ..
import tgvInpaint_cuda_ext

#torch.manual_seed(42)
rng = np.random.RandomState(12345)

# note that 1st channel of diffusion tensor is assumed to have values betewen 0 and 1.
# The 2nd channel of the diffusion tensor can have arbitrarily large values (>=0 of course).
class TGVInpaintingFunction(Function):

    # cannot pass integers along -- so embedd into variable, set gradient there to 0 or None ..
    @staticmethod
    def forward(ctx, dx, dy, dc, db, di, its, pid):

        outputs = tgvInpaint_cuda_ext.forward(dx.contiguous(), dy.contiguous(), dc.contiguous(),
                                              db.contiguous(), di.contiguous(), its, pid)
        sol = outputs[0]

        # remember for backward pass
        variables = [dx, dy, dc, db, torch.from_numpy(np.array([its, pid]))]
        ctx.save_for_backward(*variables)

        return sol

    @staticmethod
    def backward(ctx, grad_sol):
        dx, dy, dc, db, ii = ctx.saved_variables

        outputs = tgvInpaint_cuda_ext.backward(grad_sol.contiguous(), dx.contiguous(), dy.contiguous(),
                                               dc.contiguous(), db.contiguous(), ii.data.cpu().numpy()[0], 
                                               ii.data.cpu().numpy()[1])
        d_oGx, d_oGy, d_oGc, d_oGb, d_oGi = outputs

        return d_oGx, d_oGy, d_oGc, d_oGb, d_oGi, None, None


class TGVInpaint(nn.Module):
    # pid is a unique number between 0 and 9. Otherwise define in cuda header must be adjusted.
    def __init__(self, device, its, pid=3):
        super(TGVInpaint, self).__init__()

        self.its = its
        self.id = pid
        self.device = device

    # assumes t_dx, t_dy have 2 channels, see below
    def forward(self, t_dx, t_dy, t_dc, t_db, t_di):

        Yx, Xx, Lx = t_dx.shape # here we assume no batches! 
        assert Lx==2, "tensors should have 2 channels\n"
        Yi, Xi, Li = t_di.shape # here we assume no batches! 
        assert Li==3, "input should have 3! channels (primary and two auxilliary variables)\n"

        t_dx_in = t_dx.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_dy_in = t_dy.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_dc_in = t_dc.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_db_in = t_db.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_di_in = t_di.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width

        # output has 3 channel, (primary and two auxilliary variables)
        sol = TGVInpaintingFunction.apply(t_dx_in, t_dy_in, t_dc_in, t_db_in, t_di_in, self.its, self.id)
        return sol.squeeze(0).permute(1, 2, 0).contiguous()

    def init(self):
        pass
