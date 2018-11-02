import math
from torch import nn
from torch.autograd import Function
import torch

#import torch.optim as opt
#import torch.nn.functional as F
import numpy as np

# see setup.py : CUDAExtension('QuadFit_cuda_ext', [ ..
import QuadFit_cuda_ext

#torch.manual_seed(42)
rng = np.random.RandomState(12345)


class QuadFittingFromFlowFunction(Function):

    @staticmethod
    def forward(ctx, feat0, feat1, flow):

        outputs = QuadFit_cuda_ext.forward(feat0.contiguous(), feat1.contiguous(), flow.contiguous())
        sol = outputs[0]

        # remember for backward pass
        variables = [feat0, feat1, flow]
        ctx.save_for_backward(*variables)

        return sol

    @staticmethod
    def backward(ctx, grad_sol):
        feat0, feat1, flow = ctx.saved_variables

        outputs = QuadFit_cuda_ext.backward(grad_sol.contiguous(), feat0.contiguous(), 
                                            feat1.contiguous(), flow.contiguous())
        d_oGfeat0, d_oGfeat1 = outputs

        return d_oGfeat0, d_oGfeat1


class QuadFittingFromFlow(nn.Module):
    # pid is a unique number between 0 and 9. Otherwise define in cuda header must be adjusted.
    def __init__(self, device):
        super(QuadFittingFromFlow, self).__init__()
        self.device = device

    # assumes t_dx, t_dy have 2 channels, see below
    def forward(self, t_feat0, t_feat1, t_flow):

        Yx, Xx, Lx = t_feat0.shape # here we assume no batches! 
        Yi, Xi, Li = t_flow.shape # here we assume no batches! 
        assert Li==3, "flow should have 2! channels displacement in X and Y direction\n"

        t_feat0_in = t_feat0.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_feat1_in = t_feat1.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_flow_in  = t_flow.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width

        # output has 3 channel, (primary and two auxilliary variables)
        sol = QuadFittingFromFlowFunction.apply(t_feat0_in, t_feat1_in, t_flow_in)
        return sol.squeeze(0).permute(1, 2, 0).contiguous()

    def init(self):
        pass
