import math
from torch import nn
from torch.autograd import Function
import torch

#import torch.optim as opt
#import torch.nn.functional as F
import numpy as np

import median_cuda_ext # the .cpp -> the thing from the setup py

#torch.manual_seed(42)
#rng = np.random.RandomState(12345)

class MedianFunction(Function):

    # cannot pass integers along -- lol .. so embedd into variable, set gradient there to 0 or None .. -> remains fix
    @staticmethod
    def forward( ctx, in_vol ):
        #outputs = skn_cuda.skn_forward(unaries, pairwise, its, pid) # compare PYBIND11_MODULE in skn_cuda.cpp
        #self.its=its # its a static methiod .. LOL
        #self.pid=pid
        #print("Forward:")
        #variables = [unaries, pairwise]
        #ctx.save_for_backward(*variables)

        # all 3 work now -> need python binding > 4.8
        #skn_cuda_ext.forward( unaries, pairwise, its, pid ) # compare PYBIND11_MODULE in skn_cuda.cpp        
        #skn_cuda_ext.dummy( ) # compare PYBIND11_MODULE in skn_cuda.cpp
        #skn_cuda_ext.dummy_only( ) # compare PYBIND11_MODULE in skn_cuda.cpp
        #return unaries; # works .. -> no call here .. 

        # this call seg faults immediately
        outputs = median_cuda_ext.forward( in_vol.contiguous() ) # compare PYBIND11_MODULE in skn_cuda.cpp

        #outputs = skn_lse_edge_cuda_ext.forward(unaries.contiguous(), pairwise.contiguous(), unaries.contiguous, its, pid, eps) # compare PYBIND11_MODULE in skn_cuda.cpp

        #return outputs[0];
        #outputs = skn_cuda.forward( unaries, pairwise ) # compare PYBIND11_MODULE in skn_cuda.cpp        
        sol = outputs[0]
        #variables = outputs[1:] + [weights]
        #variables = [unaries, pairwise, its, pid]
        variables = [in_vol]
        ctx.save_for_backward(*variables)

        return sol

    @staticmethod
    def backward(ctx, grad_sol):
        in_vol = ctx.saved_variables[0]
        #print("Backward:")
        #print(ii.data[0])
        #print(ii.data.cpu().numpy()[1])
        #return grad_sol, ctx.saved_variables[1], None, None;
        #outputs = skn_cuda_ext.backward( grad_sol.contiguous(), *ctx.saved_variables ) 
        outputs = median_cuda_ext.backward( grad_sol.contiguous(), in_vol.contiguous() )
        outGradVol = outputs[0]
        return outGradVol


class Median(nn.Module):
    def __init__( self, device ):
        super(Median, self).__init__()
        self.device = device
        self.reset_parameters()
  
    def reset_parameters(self):
        print("Not clear -- Likely nothing to do left..")

    def forward(self, t_vol):

        Y, X, L = t_vol.shape
        #print( "Median stuff " )  
        #print( t_vol.shape )

        sol   = MedianFunction.apply( t_vol.unsqueeze(0) )
        #sol   = MedianFunction.apply( t_vol.permute(2,0,1).unsqueeze(0) )

        #return sol.squeeze(0)[:,:] #.contiguous()
        return sol.squeeze(0).permute(1,2,0)[:,:,0] #.contiguous()

        #print( "Out stuff " )  
        #print( sol.shape )


    def init(self):
        pass
