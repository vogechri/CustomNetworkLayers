import math
from torch import nn
from torch.autograd import Function
import torch

import numpy as np

# see setup.py : CUDAExtension('tvInpaint_cuda_ext', [ ..
import tvInpaint_cuda_ext

class TVInpaintingFunction(Function):

    @staticmethod
    def forward(ctx, dx, dy, dc, db, di, its, pid):
        outputs = tvInpaint_cuda_ext.forward(dx.contiguous(), dy.contiguous(), dc.contiguous(),
                                             db.contiguous(), di.contiguous(), its, pid)
        sol = outputs[0]
        variables = [dx, dy, dc, db, torch.from_numpy(np.array([its, pid]))]
        ctx.save_for_backward(*variables)

        return sol

    @staticmethod
    def backward(ctx, grad_sol):
        dx, dy, dc, db, ii = ctx.saved_variables
        outputs = tvInpaint_cuda_ext.backward(grad_sol.contiguous(), dx.contiguous(), dy.contiguous(),
                                              dc.contiguous(), db.contiguous(), ii.data.cpu().numpy()[0], ii.data.cpu().numpy()[1])
        d_oGx, d_oGy, d_oGc, d_oGb, d_oGi = outputs
        return d_oGx, d_oGy, d_oGc, d_oGb, d_oGi, None, None


class TVInpaint(nn.Module):
    def __init__(self, device, its, pid=6):
        super(TVInpaint, self).__init__()

        self.its = its
        self.id = pid
        self.device = device

    def forward(self, t_dx, t_dy, t_dc, t_db, t_di):

        #Y, X, L = t_db.shape # here we assume no batches. 

        t_dx_in = t_dx.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_dy_in = t_dy.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_dc_in = t_dc.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_db_in = t_db.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width
        t_di_in = t_di.permute(2, 0, 1).unsqueeze(0) # order is 1,Channel,Height,Width

        sol = TVInpaintingFunction.apply(t_dx_in, t_dy_in, t_dc_in, t_db_in, t_di_in, self.its, self.id)
        return sol.squeeze(0).permute(1, 2, 0).contiguous()

    def init(self):
        pass
