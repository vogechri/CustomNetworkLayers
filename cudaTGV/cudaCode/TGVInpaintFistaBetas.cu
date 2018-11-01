#include <cuda_runtime.h>
#include "FistaTGVSystemH_id_betas.h"
#include "TGVInpaintFistaHelp.cu"

// produce png images from the gradients
//#define _pngWriting_
#ifdef _pngWriting_
#define plotClipValue 1000.25f
#include <pngwriter.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <vector>
#include <limits>

#include <cuda.h>

// Well .. I want - do I - beta per direction not per x/y, .. input?
// Model is fx,fy with fx and fy treated same.
// |W nabla f - (w1,w2)| + beta | nabla w1 | + beta |nabla w2| implemented as
// |W nabla f - (w1,w2)| + | beta  nabla w1 | + | beta  nabla w2| now implemented as
// |W nabla f - (w1,w2)| + | beta1  nabla w1 | + | beta2  nabla w2| implemented as
//
// could also learn different 1 for fx and fy .. stop at some point

#include <iu/iucore.h>
#include <iostream>     // cout
#include <math.h>       // acos
#include <float.h>      // DBL_MAX

//#include <thrust/functional.h>
//#include <thrust/transform.h>

//#define _DEBUG_FW_DIR
//#define _DEBUG_BW_DIR
//#define _DEBUG_FW_ENCODE
#define _NAN_Input_Check_
#define _NAN_Output_Check_
/////////////////////////////////////////
// total max .. 32 x 32 or 32 x 16 work same..
#define COMMON_BLOCK_SIZE_2D_X 32
#define COMMON_BLOCK_SIZE_2D_Y 16
/////////////////////////////////////////

///////////////////////////////////
// quadratic data term:
//#define __quadratic__
//
// huber data term:
#define __Huber1__
//
// both off: L1 data term
///////////////////////////////////
//
// Huber deltas for TGV are 0.2 for primary and auxilliary regularizers, can be set differently, eg. for Y a little smaller eg. 0.05.
#define __HDelta2__    0.2f
#define __HDeltaY1__   0.2f
#define __HDeltaY2__   0.2f
/*
// more aggressive .. consequences ? difference to baseline above .. LOWER LEARNING RATE NEEDED .. 
#define __HDelta2__    0.1f//0.2
#define __HDeltaY1__   0.05f
#define __HDeltaY2__   0.05f
*/
///////////////////////////////////

// use self computation of L -- rather off .. ? Haeh ? maybe less aggressive .. ?
#define beta 4.0f

// use Huber regularization if on; just L2 if off. Better with
#define __HuberRegularizer__
////////////////////////////////////

// clip local gradients before adding to buffer to avoid overflows.
// (Double buffers make this more or less obsolete). Keep for safety.
#define clipLocalGrad 2000.0f
//////////////////////////////////////////////
// scale down local (in cuda code) gradients for numerical stability.
// Probably useless for L1 and Huber but not for quadratic data term. Here the local gradients can become quite large.
//
// default for quadratic, since we will add a lot of b, that is normally < 100 :
//#define localGCMult 0.01f
//
// scale down confidence gradients
#define localGCMult 0.1f
// scale down edge/diffusion tensor gradients
#define localGXMult 0.1f
// scale down gradients for rhs (u_hat). These are always small anyway.
#define localGBMult 1.0f
// scale down all returned gradients. Probably useless.
#define globalGMult 0.1f
//////////////////////////////////////////////

// write out debug information about gradient sizes every __debug_info_period__ learning steps.
#define __debug_info_period__ 75

///////////////////////////////////////////////////////////////////////////
// The Lipshitz constant depends on the learned values for TGV. Here we compute it.
__global__ void FistaInpaintB2_findLipshitz( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t betaX, cudaTextureObject_t betaY, iu::TensorGpu_32f::TensorKernelData storage )
{
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  if ( (w >= storage.W || h >= storage.H) )
    return;

  const float wx_0 = (w > 0 )          ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// +0.5 is the exact location w,h, w-0.5 is pixel w-1.
  const float wx_1 = (w < storage.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )          ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < storage.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

  const float betaTX_1y  = (h < storage.H - 1) ? tex2D<float>(betaX, w + 0.5f, h + 0.5f) : 0;
  const float betaTX_1x  = (w < storage.W - 1) ? tex2D<float>(betaX, w + 0.5f, h + 0.5f) : 0;
  const float betaTX_0x = (w > 0 )          ? tex2D<float>(betaX, w + 0.5f, h - 0.5f) : 0;
  const float betaTX_0y = (h > 0 )          ? tex2D<float>(betaX, w - 0.5f, h + 0.5f) : 0;

  const float betaTY_1y = (h < storage.H - 1) ? tex2D<float>(betaY, w + 0.5f, h + 0.5f) : 0;
  const float betaTY_1x = (w < storage.W - 1) ? tex2D<float>(betaY, w + 0.5f, h + 0.5f) : 0;
  const float betaTY_0x = (w > 0) ? tex2D<float>(betaY, w + 0.5f, h - 0.5f) : 0;
  const float betaTY_0y = (h > 0) ? tex2D<float>(betaY, w - 0.5f, h + 0.5f) : 0;


  float weightx  = 0.f;
  float weighty1 = 0.f;
  float weighty2 = 0.f;

  if ( !(w >= storage.W || h >= storage.H) ) // well need to load also beyond borders the block could be
  {
    if ( h > 0 )
    {
      weightx   += 3 * wy_0;
      weighty1  += 2 * betaTX_0y;
      weighty2  += 2 * betaTY_0y;
    }
    if ( w > 0 )
    {
      weightx   += 3 * wx_0;
      weighty1  += 2 * betaTX_0x;
      weighty2  += 2 * betaTY_0x;
    }
    weightx   += 3 * (wy_1 + wx_1);
    weighty1  += 2 * betaTX_1x + 2 * betaTX_1y + 3 * wx_1;
    weighty2  += 2 * betaTY_1x + 2 * betaTY_1y + 3 * wy_1;
  }
  storage(0, 0, h, w) = max(weightx, max( weighty1, weighty2 ) );
}


__global__ void TGVB2_init_dc_db_dx05_dw( cudaTextureObject_t cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv_05,
    iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb,
    iu::TensorGpu_32f::TensorKernelData d_gxk05, iu::TensorGpu_32f::TensorKernelData d_gxk1,
    short it , short channels = 1, float t = 1)
{
  const short n = 0;
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  if ( (w >= bv.W || h >= bv.H) )
    return;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  for ( short ch = 0; ch < channels; ch++ )
  {
    float df_dxk = d_gxk1(n, ch, h, w);
#ifdef __Huber1__
    const float diff = bv(n, ch, h, w) - xv_05(3 * it, ch, h, w);
    const float sign = (diff < 0) ? -1. : 1.;
    if ( 1. + conf >= sign * diff )
    {
      // quadratic case EXACTLY:
      // x_loc = (conf * b + x05) / (1.f+conf);
      d_gxk05(n, ch, h, w) = df_dxk / (1.f + conf);
      d_gc(n, 0, h, w)    += df_dxk * ( bv(n, ch, h, w) / (1.f + conf)  - ( conf * bv(n, ch, h, w) + xv_05(3 * it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
      d_gb(n, ch, h, w)   += df_dxk * conf / (1.f + conf) * localGBMult;
    }
    else // c>diff (x,b) -> x+-c is valid
    {
      d_gxk05(n, ch, h, w) = df_dxk;
      d_gc(n, 0, h, w)    += sign * df_dxk * localGCMult;
      d_gb(n, ch, h, w)    = 0;
    }
#else
#ifdef __quadratic__
    // x = (conf.*b+x05)./(1+conf); derivative is
    // x_loc = (conf * b + x05) / (1.f+conf);

    // df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
    // df/db    = df/dx * conf/(1+conf)
    // df/dx05  = df/dx *  1/(1+conf)
    d_gxk05(n, ch, h, w) = df_dxk / (1.f + conf);

    d_gc(n, 0, h, w)    += df_dxk * ( bv(n, ch, h, w) / (1.f + conf)  - ( conf * bv(n, ch, h, w) + xv_05(3 * it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
    d_gb(n, ch, h, w)   += df_dxk * conf / (1.f + conf) * localGBMult;
#else
    const float diff = bv(n, ch, h, w) - xv_05(3 * it, ch, h, w);
    const float sign = (diff < 0) ? -1. : 1.;

    if ( conf >= sign * diff )
    {
      d_gb(n, ch, h, w)    = df_dxk * localGBMult;
      d_gxk05(n, ch, h, w) = 0;
    }
    else // c>diff (x,b) -> x+-c is valid
    {
      d_gxk05(n, ch, h, w) = df_dxk;
      d_gc(n, 0, h, w)    += sign * df_dxk * localGCMult;
      d_gb(n, ch, h, w)    = 0;
    }
#endif
#endif
    d_gxk05(n + 1, ch, h, w) = d_gxk1(n + 1, ch, h, w);
    d_gxk05(n + 2, ch, h, w) = d_gxk1(n + 2, ch, h, w);
  }
};

// that is the main backward part, solved by ping pong on the variables whose neighbors (pixelwise) 
// need to be kept at the current iteration (or last -- but consistent!)
// i nned to know the case we were in, in the forward pass to do the correct backward update ..
// d_gxk_m1 is written to, not read from, d_gxk is read from but swapped, last iteration was d_gxk_m1 ..
__global__ void TGVB2_simple_nCalls_2D_tex_bw_yk( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf, 
    cudaTextureObject_t betaX, cudaTextureObject_t betaY, iu::TensorGpu_32f::TensorKernelData bv, 
    iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is !! k-0.5 !! at it, at it-1 the one before that
    iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb, iu::TensorGpu_32f::TensorKernelData d_gwx, 
    iu::TensorGpu_32f::TensorKernelData d_gwy, iu::TensorGpu_32f::TensorKernelData d_gxk05_in, 
    iu::TensorGpu_32f::TensorKernelData d_gxk05_out, iu::TensorGpu_32f::TensorKernelData d_gxk, 
    iu::TensorGpu_32f::TensorKernelData d_gxk_m1, iu::TensorGpu_32f::TensorKernelData d_y, // new now yk known from here on
    short it, short channels = 1, float t = 1, float nits = 1., float oneL = 0.08332 )
{
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.

  __shared__ float y_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float y1_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! x-direction
  __shared__ float y2_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! y-direction

// each block holds ~ 48KB (my card at least) ..
  __shared__ float dx05_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float dy105_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float dy205_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!

  for ( short ch = 0; ch < channels; ch++ ) //channels is small so no need for template and unroll here ..
  {
    // step 0: compute and keep weights! from yk given as input !! -> need shared mem once, store 6 weights, continue as before..
    short local_unrolled_x,  local_unrolled_y, global_unrolled_x, global_unrolled_y;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // onto 2 tiling
    local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    if (ch > 0) __syncthreads(); // since y has to be loaded completely
    y_prev[local_unrolled_y][local_unrolled_x]     = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_y(0, ch, global_unrolled_y, global_unrolled_x) : 0;
    y1_prev[local_unrolled_y][local_unrolled_x]    = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_y(1, ch, global_unrolled_y, global_unrolled_x) : 0;
    y2_prev[local_unrolled_y][local_unrolled_x]    = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_y(2, ch, global_unrolled_y, global_unrolled_x) : 0;

    dx05_prev[local_unrolled_y][local_unrolled_x]  = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ?  d_gxk05_in(0, ch, global_unrolled_y, global_unrolled_x) : 0;
    dy105_prev[local_unrolled_y][local_unrolled_x] = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in( 1, ch, global_unrolled_y, global_unrolled_x) : 0;
    dy205_prev[local_unrolled_y][local_unrolled_x] = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in( 2, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // load 2
    if ( local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
    {
      y_prev[local_unrolled_y][local_unrolled_x]     = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_y(0, ch, global_unrolled_y, global_unrolled_x) : 0;
      y1_prev[local_unrolled_y][local_unrolled_x]    = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_y(1, ch, global_unrolled_y, global_unrolled_x) : 0;
      y2_prev[local_unrolled_y][local_unrolled_x]    = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_y(2, ch, global_unrolled_y, global_unrolled_x) : 0;
      dx05_prev[local_unrolled_y][local_unrolled_x]  = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(0, ch, global_unrolled_y, global_unrolled_x) : 0;
      dy105_prev[local_unrolled_y][local_unrolled_x] = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(1, ch, global_unrolled_y, global_unrolled_x) : 0;
      dy205_prev[local_unrolled_y][local_unrolled_x] = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(2, ch, global_unrolled_y, global_unrolled_x) : 0;

    }
    __syncthreads(); // since y has to be loaded completely
////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( !(w >= bv.W || h >= bv.H) ) // well need to load also beyond borders the block could be
    {
      /////////////////////////////// steps 5:
      //     y = y + (1/L)*laplace_W*y;
      const float df_dy    = dx05_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy_t = df_dy;//temporary it is only ..
      // cases needed .. to know the weights and hence the cases I need to
      const float df_dy1   = dy105_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy1_t = df_dy1;// 1 + zij dW/dy - W_t  .. scaled! by 1/L !! so extra !
      const float df_dy2   = dy205_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy2_t = df_dy2;// 1 + zij dW/dy - W_t  .. scaled! by 1/L !! so extra !
      /////////////////////////
#ifdef __HuberRegularizer__
      if ( h > 0 ) // df/dy = df/dx*dx/dy - df/dx_{i,j}-e_2 * dx_{i,j}-e_2/dy = df/dx * dy/dx - df/dx_{i,j}-e_2 * dy/dx_{i,j}-e_2 [symmetry!]
      {
        const float betaTX    = tex2D<float>(betaX, w + 0.5f, h - 0.5f);// must be !=0
        const float betaTY    = tex2D<float>(betaY, w + 0.5f, h - 0.5f);// must be !=0

        const float wx_2  = (h > 0 && w < bv.W - 1) ? tex2D<float>(dx,    w + 0.5f, h - 0.5f) : 0;
        {
          const float zij_1 = (w < bv.W - 1) ? ( (y_prev[threadIdx.y  ][threadIdx.x + 2] - y_prev[threadIdx.y][threadIdx.x + 1] ) - y1_prev[threadIdx.y  ][threadIdx.x + 1] ) : 0;
          const float zij_2 =                ( (y_prev[threadIdx.y + 1][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 1] ) - y2_prev[threadIdx.y  ][threadIdx.x + 1] ); // -> sqrt(1/L)

          const float hub_weight_x = rsqrtf( (zij_2 * zij_2 * wy_0 + zij_1 * zij_1 * wx_2 ) ); // sqrt(1/L)
          const float W_TIJ  = min(1.f , hub_weight_x * __HDelta2__); // -> sqrt(L) or 1
          df_dy_t  -= oneL * W_TIJ * ( (df_dy  - dx05_prev[threadIdx.y][threadIdx.x + 1]) - dy205_prev[threadIdx.y][threadIdx.x + 1] ) * wy_0; // -> sqrt(L) * 1/L = sqrt(1/L) .. or 1/L

          if ( W_TIJ < 1.0 )//triangle i+1,j ; i,, ; i,j+1: W_Tij-e2 * zij_2 * wys_0
          {
            const float dW_TIJ =  W_TIJ  * ( hub_weight_x * hub_weight_x ); //delta/2 * A^{-3/2} // note that 1/2 cancels later anyway .. // -> sqrt(L)^3
            const float a2 = zij_2 * wy_0; // -> 1/sqrt(L) * 1/sqrt(L) = // -> 1/L
            const float a1 = zij_1 * wx_2;//can be 0 if oob.   // -> 1/L

            df_dy_t += oneL * (dW_TIJ * a2) * ( a1 * (dx05_prev[threadIdx.y  ][threadIdx.x + 2] - dx05_prev[threadIdx.y][threadIdx.x + 1] - dy105_prev[threadIdx.y][threadIdx.x + 1]) +
                                                a2 * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y][threadIdx.x + 1] - dy205_prev[threadIdx.y][threadIdx.x + 1]) );
          }
        }
        { //triangle i+1,j ; i,, ; i,j+1: W_Yij-e2 * (yij-yij-e2)
          // deriv wrt yij and yij-e2 and yij - e2+e1
          const float uij_1 = (w < bv.W - 1) ? (y1_prev[threadIdx.y  ][threadIdx.x + 2] - y1_prev[threadIdx.y][threadIdx.x + 1] ) : 0; // * wxs_2;
          const float uij_2 =                (y1_prev[threadIdx.y + 1][threadIdx.x + 1] - y1_prev[threadIdx.y][threadIdx.x + 1] ); // * wys_0;
          const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX ); // rsqrt (0) = inf .. so prevented below ..

          const float W_TIJ  = min(1.f , hub_weight_y * __HDeltaY1__);
          df_dy1_t -= betaTX * oneL * W_TIJ * (df_dy1 - dy105_prev[threadIdx.y][threadIdx.x + 1]);

          if ( W_TIJ < 1.0 )
          {
            const float dW_TIJ =  W_TIJ * ( hub_weight_y * hub_weight_y ); //hub_weight_y>delta > 0 ! so ok ..
            // betaT >0 always, uil1 can be ==0 so ok like this here ..
            df_dy1_t += betaTX * oneL * (dW_TIJ * betaTX * uij_2) * ( uij_1 * (dy105_prev[threadIdx.y  ][threadIdx.x + 2] - dy105_prev[threadIdx.y][threadIdx.x + 1] ) +
                        uij_2 * (dy105_prev[threadIdx.y + 1][threadIdx.x + 1] - dy105_prev[threadIdx.y][threadIdx.x + 1] ) );
          }
        }
        { //triangle i+1,j ; i,j ; i,j+1: W_Yij-e2 * (yij-yij-e2) -- 1:1 copy but with y2 ..
          const float uij_1 = (w < bv.W - 1) ? (y2_prev[threadIdx.y  ][threadIdx.x + 2] - y2_prev[threadIdx.y][threadIdx.x + 1] ) : 0; // * wxs_2;
          const float uij_2 =                (y2_prev[threadIdx.y + 1][threadIdx.x + 1] - y2_prev[threadIdx.y][threadIdx.x + 1] ); // * wys_0;
          const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );

          const float W_TIJ  = min(1.f , hub_weight_y * __HDeltaY2__);
          df_dy2_t -= betaTY * oneL * W_TIJ * ( df_dy2 - dy205_prev[threadIdx.y][threadIdx.x + 1] ); //

          if ( W_TIJ < 1.0 )
          {
            const float dW_TIJ =  W_TIJ  / ( hub_weight_y * hub_weight_y );
            df_dy2_t += betaTY * oneL * (dW_TIJ * betaTY * uij_2) * ( uij_1 * ( dy205_prev[threadIdx.y  ][threadIdx.x + 2] - dy205_prev[threadIdx.y][threadIdx.x + 1]) +
                        uij_2 * ( dy205_prev[threadIdx.y + 1][threadIdx.x + 1] - dy205_prev[threadIdx.y][threadIdx.x + 1]) );
          }
        }
      }//h>0

      if ( w > 0 ) // df/dy = df/dx*dx/dy - df/dx_{i,j}-e_2 * dx_{i,j}-e_2/dy = df/dx * dy/dx - df/dx_{i,j}-e_2 * dy/dx_{i,j}-e_2 [symmetry!]
      {
        const float betaTX = tex2D<float>(betaX, w - 0.5f, h + 0.5f);
        const float betaTY = tex2D<float>(betaY, w - 0.5f, h + 0.5f);
        const float wy_2  = (w > 0 && h < bv.H - 1) ? tex2D<float>(dy,    w - 0.5f, h + 0.5f) : 0; // 1/sqrt(L)
        {
          const float zij_1 =                ( y_prev[threadIdx.y + 1][threadIdx.x + 1] - y_prev[threadIdx.y + 1][threadIdx.x] - y1_prev[threadIdx.y + 1][threadIdx.x  ] );
          const float zij_2 = (h < bv.H - 1) ? ( y_prev[threadIdx.y + 2][threadIdx.x  ] - y_prev[threadIdx.y + 1][threadIdx.x] - y2_prev[threadIdx.y + 1][threadIdx.x  ] ) : 0;
          const float hub_weight_x = rsqrtf( (zij_2 * zij_2 * wy_2 + zij_1 * zij_1 * wx_0 ) );

          const float W_TIJ  = min(1.f , hub_weight_x * __HDelta2__);
          df_dy_t  -= oneL * W_TIJ * ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x]) - dy105_prev[threadIdx.y + 1][threadIdx.x] ) * wx_0;

          if ( W_TIJ < 1.0 )
          {
            const float dW_TIJ =  W_TIJ  * ( hub_weight_x * hub_weight_x ); //delta/2 * A^{-3/2} // note that 1/2 cancels later anyway ..
            const float a1 = zij_1 * wx_0;
            const float a2 = zij_2 * wy_2;///can be 0 so below is not needed

            df_dy_t += oneL * (dW_TIJ * a1) * ( a1 * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 1][threadIdx.x] - dy105_prev[threadIdx.y + 1][threadIdx.x]) +
                                                a2 * (dx05_prev[threadIdx.y + 2][threadIdx.x  ] - dx05_prev[threadIdx.y + 1][threadIdx.x] - dy205_prev[threadIdx.y + 1][threadIdx.x]) );
          }
        }
        {
          const float uij_1 =                (y1_prev[threadIdx.y + 1][threadIdx.x + 1] - y1_prev[threadIdx.y + 1][threadIdx.x  ]); // wxs_0;
          const float uij_2 = (h < bv.H - 1) ? (y1_prev[threadIdx.y + 2][threadIdx.x  ] - y1_prev[threadIdx.y + 1][threadIdx.x  ] ) : 0; // * wys_2;
          const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX );

          const float W_TIJ  = min(1.f , hub_weight_y * __HDeltaY1__);
          df_dy1_t -= betaTX * oneL * W_TIJ *   (df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x]); //

          if ( W_TIJ < 1.0 )
          {
            const float dW_TIJ =  W_TIJ * ( hub_weight_y * hub_weight_y );
            df_dy1_t += betaTX * oneL * (dW_TIJ * betaTX * uij_1) * ( uij_2 * ( dy105_prev[threadIdx.y + 2][threadIdx.x  ] - dy105_prev[threadIdx.y + 1][threadIdx.x] ) +
                        uij_1 * ( dy105_prev[threadIdx.y + 1][threadIdx.x + 1] - dy105_prev[threadIdx.y + 1][threadIdx.x] ) );
          }
        }
        {
          const float uij_1 =                (y2_prev[threadIdx.y + 1][threadIdx.x + 1] - y2_prev[threadIdx.y + 1][threadIdx.x  ] );
          const float uij_2 = (h < bv.H - 1) ? (y2_prev[threadIdx.y + 2][threadIdx.x  ] - y2_prev[threadIdx.y + 1][threadIdx.x  ] ) : 0;
          const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );

          const float W_TIJ  = min(1.f , hub_weight_y * __HDeltaY2__);
          df_dy2_t -= betaTY * oneL * W_TIJ *   (df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x]);
          if ( W_TIJ < 1.0 )
          {
            const float dW_TIJ =  W_TIJ * ( hub_weight_y * hub_weight_y );

            df_dy2_t += betaTY * oneL * (dW_TIJ * betaTY * uij_1) * ( uij_2 * ( dy205_prev[threadIdx.y + 2][threadIdx.x  ] - dy205_prev[threadIdx.y + 1][threadIdx.x] ) +
                        uij_1 * ( dy205_prev[threadIdx.y + 1][threadIdx.x + 1] - dy205_prev[threadIdx.y + 1][threadIdx.x] ) );
          }
        }
      }// w>0
      {
        const float betaTX = tex2D<float>(betaX, w + 0.5f, h + 0.5f);// +0.5 is the exact location w,h, w-0.5 is pixel w-1.
        const float betaTY = tex2D<float>(betaY, w + 0.5f, h + 0.5f);
        { // xij
          const float aij_1 = (w < bv.W - 1) ? ( (y_prev[threadIdx.y + 1][threadIdx.x + 2] - y_prev[threadIdx.y + 1][threadIdx.x + 1] ) - y1_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0;
          const float aij_2 = (h < bv.H - 1) ? ( (y_prev[threadIdx.y + 2][threadIdx.x + 1] - y_prev[threadIdx.y + 1][threadIdx.x + 1] ) - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0;

          const float hub_weight_x = rsqrtf( (aij_2 * aij_2 * wy_1 + aij_1 * aij_1 * wx_1 ) ) ;
          const float W_TIJ  = min(1.f , hub_weight_x * __HDelta2__);

          const float dyij_1 = df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] + df_dy1;
          const float dyij_2 = df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1] + df_dy2;

          // SYMMETRY HERE ! same in df_dy1 and df_dy as well as in df_dy2
          df_dy_t  -= oneL * W_TIJ * ( dyij_2 * wy_1 + dyij_1 * wx_1 ); //A1: like A2
          df_dy1_t -= oneL * W_TIJ * dyij_1 * wx_1; //B2: like B1
          df_dy2_t -= oneL * W_TIJ * dyij_2 * wy_1; //A2: like A1

          // this is only tv part missing is part on yij,k += zij,k WT_ij or it is eactly present with df+dy1 ..
          // indeed wij,1 occurs only  xij -= wij,1 aij,1 * WTij and yij -= aij,1 wij,1 * WTij
          // and xij+e1 += wij,1 aij,1 * WTij [note the sign!] .. all 3 share the derivative wrt wij,1: (aij_1 * W_TIJ)

          d_gwx(0, 0, h,   w  ) = d_gwx(0, 0, h,   w  ) + oneL * localGXMult * dyij_1 * (aij_1 * W_TIJ);
          d_gwy(0, 0, h,   w  ) = d_gwy(0, 0, h,   w  ) + oneL * localGXMult * dyij_2 * (aij_2 * W_TIJ);

          if ( W_TIJ < 1.0 )
          {
            const float dW_TIJ =  W_TIJ * ( hub_weight_x * hub_weight_x ); //delta/2 * A^{-3/2} // note that 1/2 cancels later anyway ..
            const float a1 = aij_1 * wx_1;
            const float a2 = aij_2 * wy_1;//zij_2 * wys_1;
            ///////////////////////////////////
            const float deriv_Tij  = ( dyij_1 * a1 + dyij_2 * a2 ) * oneL;
            df_dy_t  +=  deriv_Tij * ((a1 + a2) * dW_TIJ) ;
            df_dy2_t +=  deriv_Tij * (a2 * dW_TIJ) ;
            df_dy1_t +=  deriv_Tij * (a1 * dW_TIJ) ;
            ///////////////////////////////////
            d_gwy(0, 0, h,   w  ) = d_gwy(0, 0, h,   w  ) - localGXMult * (aij_2 * dW_TIJ * aij_2) * 0.5 * deriv_Tij;
            d_gwx(0, 0, h,   w  ) = d_gwx(0, 0, h,   w  ) - localGXMult * (aij_1 * dW_TIJ * aij_1) * 0.5 * deriv_Tij;
          }
        }
        // first y1
        {
          const float uij_1 = (w < bv.W - 1) ? (y1_prev[threadIdx.y + 1][threadIdx.x + 2] - y1_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0; // wxs_0;
          const float uij_2 = (h < bv.H - 1) ? (y1_prev[threadIdx.y + 2][threadIdx.x + 1] - y1_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0; // * wys_2;
          const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX );
          const float W_YIJ  = min(1.f , hub_weight_y * __HDeltaY1__);

          if ( w < bv.W - 1 )
            df_dy1_t -= betaTX * oneL * W_YIJ * ( df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x + 2] ); // in x and y dir laplace: ok
          if ( h < bv.H - 1 )
            df_dy1_t -= betaTX * oneL * W_YIJ * ( df_dy1 - dy105_prev[threadIdx.y + 2][threadIdx.x + 1] ); // in x and y dir laplace: ok

          d_gwx(0, 1, h, w ) = d_gwx(0, 1, h, w ) + oneL * localGXMult * ( ( df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 +
                               ( df_dy1 - dy105_prev[threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 ) * W_YIJ;

          if ( W_YIJ < 1.0 )
          {
            const float dW_YIJ =  W_YIJ * ( hub_weight_y * hub_weight_y );
            const float deriv_Yij = ( ( df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 * betaTX +
                                      ( df_dy1 - dy105_prev[threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 * betaTX ) * oneL;

            df_dy1_t += betaTX * dW_YIJ * (uij_1 + uij_2) * deriv_Yij;
            d_gwx(0, 1, h, w  ) = d_gwx(0, 1, h, w  ) - localGXMult * (uij_2 * dW_YIJ * uij_2 + uij_1 * dW_YIJ * uij_1) * 0.5 * deriv_Yij;
          }
        }
        // now y2
        {
          const float uij_1 = (w < bv.W - 1) ? (y2_prev[threadIdx.y + 1][threadIdx.x + 2] - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0; // * wys_0;
          const float uij_2 = (h < bv.H - 1) ? (y2_prev[threadIdx.y + 2][threadIdx.x + 1] - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0; // * wxs_2;
          const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );

          const float W_YIJ  = min(1.f , hub_weight_y * __HDeltaY2__);

          if ( w < bv.W - 1 )
            df_dy2_t -= betaTY * oneL * W_YIJ * ( df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x + 2] ); // in x and y dir laplace
          if ( h < bv.H - 1 )
            df_dy2_t -= betaTY * oneL * W_YIJ * ( df_dy2 - dy205_prev[threadIdx.y + 2][threadIdx.x + 1] ); // in x and y dir laplace

          d_gwy(0, 1, h, w ) = d_gwy(0, 1, h, w ) + oneL * localGXMult * ( ( df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 +
                               ( df_dy2 - dy205_prev[threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 ) * W_YIJ;
          if ( W_YIJ < 1.0 )
          {
            const float dW_YIJ =  W_YIJ  * ( hub_weight_y * hub_weight_y );
            const float deriv_Yij = ( ( df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 * betaTY +
                                      ( df_dy2 - dy205_prev[threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 * betaTY ) * oneL;

            df_dy2_t += betaTY * dW_YIJ * (uij_1 + uij_2) * deriv_Yij;
            d_gwy(0, 1, h, w  ) = d_gwy(0, 1, h, w  ) - localGXMult * (uij_2 * dW_YIJ * uij_2 + uij_1 * dW_YIJ * uij_1) * 0.5 * deriv_Yij;
          }
        }
      }
/////////////////////////
#else // no huber on it -> a lot simpler 
      {
        const float wxs_0 = (w > 0 )     ? tex2D<float>(dx_sq, w - 0.5f, h + 0.5f) : 0;// *L
        const float wxs_1 = (w < bv.W - 1) ? tex2D<float>(dx_sq, w + 0.5f, h + 0.5f) : 0;
        const float wys_0 = (h > 0 )     ? tex2D<float>(dy_sq, w + 0.5f, h - 0.5f) : 0;
        const float wys_1 = (h < bv.H - 1) ? tex2D<float>(dy_sq, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.
        if ( h > 0 ) // df/dy = df/dx*dx/dy - df/dx_{i,j}-e_2 * dx_{i,j}-e_2/dy = df/dx * dy/dx - df/dx_{i,j}-e_2 * dy/dx_{i,j}-e_2 [symmetry!]
        {
          df_dy_t  -= oneL *  ( (df_dy  - dx05_prev[threadIdx.y][threadIdx.x + 1]) - dy205_prev[threadIdx.y][threadIdx.x + 1] ) * wys_0 * wys_0;
          df_dy1_t -= oneL *   (df_dy1 - dy105_prev[threadIdx.y][threadIdx.x + 1]) * beta;
          df_dy2_t -= oneL *   (df_dy2 - dy205_prev[threadIdx.y][threadIdx.x + 1]) * beta;
        }
        if ( h < bv.H - 1 )
        {
          df_dy_t  -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) + df_dy2 ) * wys_1 * wys_1; //L1
          df_dy1_t -= oneL * (                                                                   (df_dy1 - dy105_prev[threadIdx.y + 2][threadIdx.x + 1]) * beta );
          df_dy2_t -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wys_1 * wys_1 + (df_dy2 - dy205_prev[threadIdx.y + 2][threadIdx.x + 1]) * beta );
          df_dy2_t -= oneL * df_dy2 * wys_1 * wys_1;
        }
        if ( w > 0 )
        {
          df_dy_t  -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x]) - dy105_prev[threadIdx.y + 1][threadIdx.x] ) * wxs_0 * wxs_0;
          df_dy1_t -= oneL *   (df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x]) * beta;
          df_dy2_t -= oneL *   (df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x]) * beta;
        }
        if ( w < bv.W - 1 )
        {
          df_dy_t  -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] ) + df_dy1 ) * wxs_1 * wxs_1; //L2
          df_dy1_t -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] ) * wxs_1 * wxs_1 + (df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x + 2]) * beta );
          df_dy2_t -= oneL * (                                                                  + (df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x + 2]) * beta );
          df_dy1_t -= oneL * df_dy1 * wxs_1 * wxs_1; //L5
        }

        if (w < bv.W - 1 )
          d_gwx(0, 0, h,   w  ) = d_gwx(0, 0, h,   w  ) + oneL * localGXMult * (
                                    (y_prev[threadIdx.y + 1][threadIdx.x + 2] - y_prev[threadIdx.y + 1][threadIdx.x + 1] - y1_prev[threadIdx.y + 1][threadIdx.x + 1] ) * ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2])
                                    + (y_prev[threadIdx.y + 1][threadIdx.x + 2] - y_prev[threadIdx.y + 1][threadIdx.x + 1] - y1_prev[threadIdx.y + 1][threadIdx.x + 1] ) *   df_dy1
                                  );

        if (h < bv.H - 1)
          d_gwy(0, 0, h,   w  ) = d_gwy(0, 0, h,   w  ) + oneL * localGXMult * (
                                    (y_prev[threadIdx.y + 2][threadIdx.x + 1] - y_prev[threadIdx.y + 1][threadIdx.x + 1] - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) * ( df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1])
                                    + (y_prev[threadIdx.y + 2][threadIdx.x + 1] - y_prev[threadIdx.y + 1][threadIdx.x + 1] - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) *   df_dy2
                                  );
      }
#endif //end of huber or l2 regularization

// had scaled above by #define s_dx 0.1
//df_dy_t  /= s_dx;
//df_dy1_t /= s_dx;
//df_dy2_t /= s_dx;

      d_gxk_m1(1, ch, h, w) = -t * df_dy1_t;
      df_dy1_t = ( d_gxk(1, ch, h, w) + (1. + t) * df_dy1_t );
#ifdef clipLocalGrad
      df_dy1_t = max( -clipLocalGrad, min (df_dy1_t, clipLocalGrad) );
#endif
// overelaxed version .. ( d_gxk(n,ch,h,w) + (1. + t) * df_dy_t )
      d_gxk05_out(1, ch, h, w)  = df_dy1_t;
      d_gxk(1, ch, h, w)        = df_dy1_t;

      d_gxk_m1(2, ch, h, w) = -t * df_dy2_t;
      df_dy2_t = ( d_gxk(2, ch, h, w) + (1. + t) * df_dy2_t );
#ifdef clipLocalGrad
      df_dy2_t = max( -clipLocalGrad, min (df_dy2_t, clipLocalGrad) );
#endif
// overelaxed version .. ( d_gxk(n,ch,h,w) + (1. + t) * df_dy_t )
      d_gxk05_out(2, ch, h, w)  = df_dy2_t;
      d_gxk(2, ch, h, w)        = df_dy2_t;

////////////////////////////////////////////////////////////////////////
/////////////////////////////// steps 6 and 7 ///////////////////
      float df_dxk = ( d_gxk(0, ch, h, w) + (1. + t) * df_dy_t );
#ifdef clipLocalGrad
      df_dxk = max( -clipLocalGrad, min (df_dxk, clipLocalGrad) );
#endif
      d_gxk(0, ch, h, w)    = df_dxk; // TODO: WRITTEN JUST FOR DEBUGGING -> if runnig can just go ..
      d_gxk_m1(0, ch, h, w) = -t * df_dy_t;
////////////////////////////////////////////////////////////////////////
// steps 2,3,4
      const float bb   = bv(0, ch, h, w);

#ifdef __Huber1__

      float diff = bb - xv_m05(3 * it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      if ( 1. + conf >= sign * diff )
      {
        d_gc(0, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(3 * it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
        d_gxk05_out(0, ch, h, w)  = df_dxk / (1.f + conf);
        d_gb(0, ch, h, w)        += df_dxk * conf / (1.f + conf) * localGBMult;
      }
      else // l1 case
      {
        if ( conf >= sign * diff )
        {
          d_gxk05_out(0, ch, h, w) = 0;
          d_gb(0, ch, h, w)       += df_dxk * localGBMult;
        }
        else
        {
          d_gxk05_out(0, ch, h, w) = df_dxk;
          d_gc(0, 0, h, w)       += sign * df_dxk * localGCMult;
        }
      }

#else // not huber case any more /////////////////////////////////

#ifdef __quadratic__
// x = (conf.*b+x05)./(1+conf); derivative is
// df/db    = df/dx * conf/(1+conf)
// df/dx05  = df/dx *  1/(1+conf)
// df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
      d_gc(0, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(3 * it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
      d_gxk05_out(0, ch, h, w)  = df_dxk / (1.f + conf);
      d_gb(0, ch, h, w)        += df_dxk * conf / (1.f + conf) * localGBMult;
#else

      float diff = bb - xv_m05(3 * it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      if ( conf >= sign * diff )
      {
        d_gxk05_out(0, ch, h, w) = 0;
        d_gb(0, ch, h, w)       += df_dxk * localGBMult;
      }
      else
      {
        d_gxk05_out(0, ch, h, w) = df_dxk;
        d_gc(0, 0, h, w)       += sign * df_dxk * localGCMult;
      }
#endif // quadratic

#endif // huber 
////////////////////////////////
    }
  }
};


template<int channels>
__global__ void TGVB2_simple_nCalls_2D_tex_bw_yk_ch( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf, 
    cudaTextureObject_t betaX, cudaTextureObject_t betaY, iu::TensorGpu_32f::TensorKernelData bv, 
    iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is !! k-0.5 !! at it, at it-1 the one before that
    iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb, iu::TensorGpu_32f::TensorKernelData d_gwx, 
    iu::TensorGpu_32f::TensorKernelData d_gwy, iu::TensorGpu_32f::TensorKernelData d_gxk05_in, 
    iu::TensorGpu_32f::TensorKernelData d_gxk05_out, iu::TensorGpu_32f::TensorKernelData d_gxk, 
    iu::TensorGpu_32f::TensorKernelData d_gxk_m1, iu::TensorGpu_32f::TensorKernelData d_y, // new now yk known from here on
    short it, float t = 1, float nits = 1., float oneL = 0.08332 )
{
  short w  = blockIdx.x * blockDim.x + threadIdx.x;
  short h  = blockIdx.y * blockDim.y + threadIdx.y;
  short ch = blockIdx.z * blockDim.z + threadIdx.z;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

  __shared__ float y_prev[ channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float y1_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! x-direction
  __shared__ float y2_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! y-direction

  // each block holds ~ 48KB (my card at least) ..
  __shared__ float dx05_prev [channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float dy105_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float dy205_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!

  // step 0: compute and keep weights! from yk given as input !! -> need shared mem once, store 6 weights, continue as before..
  //const float yk   = d_y(n, ch, h, w);
  short local_unrolled_x,  local_unrolled_y, global_unrolled_x, global_unrolled_y;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // onto 2 tiling .. alright ..
  local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
  local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
  global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
  global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1; //0.01

  //if (ch>0) __syncthreads(); // since y has to be loaded completely
  if (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W)
  {
    y_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]    = d_y(0, ch, global_unrolled_y, global_unrolled_x);
    y1_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]    = d_y(1, ch, global_unrolled_y, global_unrolled_x);
    y2_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]    = d_y(2, ch, global_unrolled_y, global_unrolled_x);

    dx05_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_gxk05_in( 0, ch, global_unrolled_y, global_unrolled_x);
    dy105_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_gxk05_in( 1, ch, global_unrolled_y, global_unrolled_x);
    dy205_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_gxk05_in( 2, ch, global_unrolled_y, global_unrolled_x);
  }
  else
  {
    y_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]    = 0;
    y1_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]    = 0;
    y2_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]    = 0;

    dx05_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
    dy105_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
    dy205_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
  }

  local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
  local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
  global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
  global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

  // load 2
  if ( local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
  {
    if (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W)
    {
      y_prev[local_unrolled_y    + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_y(0, ch, global_unrolled_y, global_unrolled_x);
      y1_prev[local_unrolled_y   + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_y(1, ch, global_unrolled_y, global_unrolled_x);
      y2_prev[local_unrolled_y   + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_y(2, ch, global_unrolled_y, global_unrolled_x);
      dx05_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_gxk05_in(0, ch, global_unrolled_y, global_unrolled_x);
      dy105_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_gxk05_in(1, ch, global_unrolled_y, global_unrolled_x);
      dy205_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = d_gxk05_in(2, ch, global_unrolled_y, global_unrolled_x);
    }
    else
    {
      y_prev[local_unrolled_y    + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
      y1_prev[local_unrolled_y   + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
      y2_prev[local_unrolled_y   + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
      dx05_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
      dy105_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
      dy205_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x] = 0;
    }
  }
  __syncthreads(); // since y has to be loaded completely
////////////////////////////////////////////////////////////////////////////////////////////////////

  if ( !(w >= bv.W || h >= bv.H) ) // well need to load also beyond borders the block could be
  {
    /////////////////////////////// steps 5:
    //     y = y + (1/L)*laplace_W*y;
    const float df_dy    = dx05_prev[ ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1];
    float df_dy_t = df_dy;//temporary it is only ..
    // cases needed .. to know the weights and hence the cases I need to
    const float df_dy1   = dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1];
    float df_dy1_t = df_dy1;// 1 + zij dW/dy - W_t  .. scaled! by 1/L !! so extra !
    const float df_dy2   = dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1];
    float df_dy2_t = df_dy2;// 1 + zij dW/dy - W_t  .. scaled! by 1/L !! so extra !
    /////////////////////////
    if ( h > 0 ) // df/dy = df/dx*dx/dy - df/dx_{i,j}-e_2 * dx_{i,j}-e_2/dy = df/dx * dy/dx - df/dx_{i,j}-e_2 * dy/dx_{i,j}-e_2 [symmetry!]
    {
      const float betaTX = tex2D<float>(betaX, w + 0.5f, h - 0.5f);// must be !=0 .. so :
      const float betaTY = tex2D<float>(betaY, w + 0.5f, h - 0.5f);// must be !=0 .. so :

      const float wx_2  = (h > 0 && w < bv.W - 1) ? tex2D<float>(dx,    w + 0.5f, h - 0.5f) : 0;
      {
        const float zij_1 = (w < bv.W - 1) ? ( (y_prev[threadIdx.y  + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] - y_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)  ][threadIdx.x + 1] ) : 0;
        const float zij_2 =                ( (y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] - y_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) - y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)  ][threadIdx.x + 1] ); // -> sqrt(1/L)

        const float hub_weight_x = rsqrtf( (zij_2 * zij_2 * wy_0 + zij_1 * zij_1 * wx_2 ) ); // sqrt(1/L)
        const float W_TIJ  = min(1.f , hub_weight_x * __HDelta2__); // -> sqrt(L) or 1
        df_dy_t  -= oneL * W_TIJ * ( (df_dy  - dx05_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1]) - dy205_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) * wy_0; // -> sqrt(L) * 1/L = sqrt(1/L) .. or 1/L

        if ( W_TIJ < 1.0 )//triangle i+1,j ; i,, ; i,j+1: W_Tij-e2 * zij_2 * wys_0
        {
          const float dW_TIJ =  W_TIJ  * ( hub_weight_x * hub_weight_x ); //delta/2 * A^{-3/2} // note that 1/2 cancels later anyway .. // -> sqrt(L)^3
          const float a2 = zij_2 * wy_0; // -> 1/sqrt(L) * 1/sqrt(L) = // -> 1/L
          const float a1 = zij_1 * wx_2;//can be 0 if oob.   // -> 1/L

          df_dy_t += oneL * (dW_TIJ * a2) * ( a1 * (dx05_prev[threadIdx.y  + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] - dx05_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] - dy105_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1]) +
                                              a2 * (dx05_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] - dx05_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] - dy205_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1]) );
        }
      }
      { //triangle i+1,j ; i,, ; i,j+1: W_Yij-e2 * (yij-yij-e2)
        // deriv wrt yij and yij-e2 and yij - e2+e1
        const float uij_1 = (w < bv.W - 1) ? (y1_prev[threadIdx.y  + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) : 0; // * wxs_2;
        const float uij_2 =                (y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ); // * wys_0;
        const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX ); // rsqrt (0) = inf .. so prevented below .. although stupid ..

        // linear case -- no huber .. shit is not on y but on other stuff !
        const float W_TIJ  = min(1.f , hub_weight_y * __HDeltaY1__);

        df_dy1_t -= betaTX * oneL * W_TIJ * (df_dy1 - dy105_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1]); // .. hmm beta .. not really ideal is it ? can still to global val ..

        if ( W_TIJ < 1.0 )
        {
          const float dW_TIJ =  W_TIJ * ( hub_weight_y * hub_weight_y ); //hub_weight_y>delta > 0 ! so ok ..
          // betaT >0 always, uil1 can be ==0 so ok like this here ..
          df_dy1_t += betaTX * oneL * (dW_TIJ * betaTX * uij_2) * ( uij_1 * (dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y  ][threadIdx.x + 2] - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y][threadIdx.x + 1] ) +
                      uij_2 * (dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y][threadIdx.x + 1] ) );
        }
      }
      { //triangle i+1,j ; i,j ; i,j+1: W_Yij-e2 * (yij-yij-e2) -- 1:1 copy but with y2 ..
        const float uij_1 = (w < bv.W - 1) ? (y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y  ][threadIdx.x + 2] - y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y][threadIdx.x + 1] ) : 0; // * wxs_2;
        const float uij_2 =                (y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y][threadIdx.x + 1] ); // * wys_0;
        const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );

        const float W_TIJ  = min(1.f , hub_weight_y * __HDeltaY2__);
        df_dy2_t -= betaTY * oneL * W_TIJ * ( df_dy2 - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y][threadIdx.x + 1] ); //

        if ( W_TIJ < 1.0 )
        {
          const float dW_TIJ =  W_TIJ  / ( hub_weight_y * hub_weight_y );
          df_dy2_t += betaTY * oneL * (dW_TIJ * betaTY * uij_2) * ( uij_1 * ( dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y  ][threadIdx.x + 2] - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y][threadIdx.x + 1]) +
                      uij_2 * ( dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y][threadIdx.x + 1]) );
        }
      }
    }//h>0

    if ( w > 0 ) // df/dy = df/dx*dx/dy - df/dx_{i,j}-e_2 * dx_{i,j}-e_2/dy = df/dx * dy/dx - df/dx_{i,j}-e_2 * dy/dx_{i,j}-e_2 [symmetry!]
    {
      const float betaTX = tex2D<float>(betaX, w - 0.5f, h + 0.5f);
      const float betaTY = tex2D<float>(betaY, w - 0.5f, h + 0.5f);
      //const float wys_2 = (w > 0 && h < bv.H-1) ? tex2D<float>(dy_sq, w - 0.5f, h + 0.5f) : 0;
      const float wy_2  = (w > 0 && h < bv.H - 1) ? tex2D<float>(dy,    w - 0.5f, h + 0.5f) : 0; // 1/sqrt(L)
      {
        const float zij_1 =                ( y_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - y_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] - y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x  ] );
        const float zij_2 = (h < bv.H - 1) ? ( y_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x  ] - y_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] - y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x  ] ) : 0;

        const float hub_weight_x = rsqrtf( (zij_2 * zij_2 * wy_2 + zij_1 * zij_1 * wx_0 ) );
        const float W_TIJ  = min(1.f , hub_weight_x * __HDelta2__);
        df_dy_t  -= oneL * W_TIJ * ( (df_dy  - dx05_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x]) - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] ) * wx_0;

        if ( W_TIJ < 1.0 )
        {
          const float dW_TIJ =  W_TIJ  * ( hub_weight_x * hub_weight_x ); //delta/2 * A^{-3/2} // note that 1/2 cancels later anyway ..
          const float a1 = zij_1 * wx_0;
          const float a2 = zij_2 * wy_2;///can be 0 so below is not needed

          df_dy_t += oneL * (dW_TIJ * a1) * ( a1 * (dx05_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x]) +
                                              a2 * (dx05_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x  ] - dx05_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x]) );
        }
      }
      {
        const float uij_1 =                (y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x  ]); // wxs_0;
        const float uij_2 = (h < bv.H - 1) ? (y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x  ] - y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x  ] ) : 0; // * wys_2;
        const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX );
        // linear case -- no huber .. shit is not on y but on other stuff !
        const float W_TIJ  = min(1.f , hub_weight_y * __HDeltaY1__);
        df_dy1_t -= betaTX * oneL * W_TIJ *   (df_dy1 - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x]); //

        if ( W_TIJ < 1.0 )
        {
          const float dW_TIJ =  W_TIJ * ( hub_weight_y * hub_weight_y );
          df_dy1_t += betaTX * oneL * (dW_TIJ * betaTX * uij_1) * ( uij_2 * ( dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x  ] - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] ) +
                      uij_1 * ( dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] ) );
        }
      }
      {
        const float uij_1 =                (y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x  ] ); // * wys_0;
        const float uij_2 = (h < bv.H - 1) ? (y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x  ] - y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x  ] ) : 0; // * wxs_2;
        const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );

        // linear case -- no huber .. shit is not on y but on other stuff !
        const float W_TIJ  = min(1.f , hub_weight_y * __HDeltaY2__);
        df_dy2_t -= betaTY * oneL * W_TIJ *   (df_dy2 - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x]); //
        if ( W_TIJ < 1.0 )
        {
          const float dW_TIJ =  W_TIJ * ( hub_weight_y * hub_weight_y );

          df_dy2_t += betaTY * oneL * (dW_TIJ * betaTY * uij_1) * ( uij_2 * ( dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x  ] - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] ) +
                      uij_1 * ( dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x] ) );
        }
      }
    }// w>0
    {
      const float betaTX = tex2D<float>(betaX, w + 0.5f, h + 0.5f);
      const float betaTY = tex2D<float>(betaY, w + 0.5f, h + 0.5f);
      { // xij
        const float aij_1 = (w < bv.W - 1) ? ( (y_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] - y_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] ) - y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] ) : 0;
        const float aij_2 = (h < bv.H - 1) ? ( (y_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] - y_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] ) - y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] ) : 0;

        const float hub_weight_x = rsqrtf( (aij_2 * aij_2 * wy_1 + aij_1 * aij_1 * wx_1 ) ) ; // varaible as it is sqrt expensive
        const float W_TIJ  = min(1.f , hub_weight_x * __HDelta2__);

        const float dyij_1 = df_dy - dx05_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] + df_dy1;
        const float dyij_2 = df_dy - dx05_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] + df_dy2;

        // SYMMETRY HERE ! same in df_dy1 and df_dy as well as in df_dy2
        df_dy_t  -= oneL * W_TIJ * ( dyij_2 * wy_1 + dyij_1 * wx_1 ); //A1: like A2
        df_dy1_t -= oneL * W_TIJ * dyij_1 * wx_1; //B2: like B1
        df_dy2_t -= oneL * W_TIJ * dyij_2 * wy_1; //A2: like A1

        // this is only tv part missing is part on yij,k += zij,k WT_ij or it is eactly present with df+dy1 ..
        // indeed wij,1 occurs only  xij -= wij,1 aij,1 * WTij and yij -= aij,1 wij,1 * WTij
        // and xij+e1 += wij,1 aij,1 * WTij [note the sign!] .. all 3 share the derivative wrt wij,1: (aij_1 * W_TIJ)

        float dgwx = oneL * localGXMult * dyij_1 * (aij_1 * W_TIJ);
        float dgwy = oneL * localGXMult * dyij_2 * (aij_2 * W_TIJ);

        if ( W_TIJ < 1.0 )
        {
          const float dW_TIJ =  W_TIJ * ( hub_weight_x * hub_weight_x ); //delta/2 * A^{-3/2} // note that 1/2 cancels later anyway ..
          const float a1 = aij_1 * wx_1;
          const float a2 = aij_2 * wy_1;//zij_2 * wys_1;
          ///////////////////////////////////
          const float  deriv_Tij  = ( dyij_1 * a1 + dyij_2 * a2 ) * oneL;
          df_dy_t  +=  deriv_Tij * ((a1 + a2) * dW_TIJ) ;
          df_dy2_t +=  deriv_Tij * (a2 * dW_TIJ) ;
          df_dy1_t +=  deriv_Tij * (a1 * dW_TIJ) ;
          ///////////////////////////////////
          dgwx -= localGXMult * (aij_1 * dW_TIJ * aij_1) * 0.5 * deriv_Tij;
          dgwy -= localGXMult * (aij_2 * dW_TIJ * aij_2) * 0.5 * deriv_Tij;
        }
        atomicAdd( &(d_gwx(0, 0, h, w )), dgwx );
        atomicAdd( &(d_gwy(0, 0, h, w )), dgwy );
      }
      // first y1
      float dgwx1(0), dgwx2(0);
      {
        const float uij_1 = (w < bv.W - 1) ? (y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] - y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] ) : 0; // wxs_0;
        const float uij_2 = (h < bv.H - 1) ? (y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] - y1_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] ) : 0; // * wys_2;
        const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX );

        // linear case -- no huber .. shit is not on y but on other stuff !
        const float W_YIJ  = min(1.f , hub_weight_y * __HDeltaY1__);

        if ( w < bv.W - 1 )
          df_dy1_t -= betaTX * oneL * W_YIJ * ( df_dy1 - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] ); // in x and y dir laplace: ok
        if ( h < bv.H - 1 )
          df_dy1_t -= betaTX * oneL * W_YIJ * ( df_dy1 - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] ); // in x and y dir laplace: ok

        dgwx1 = oneL * localGXMult * ( ( df_dy1 - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 +
                                       ( df_dy1 - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 ) * W_YIJ;
        if ( W_YIJ < 1.0 )
        {
          const float dW_YIJ =  W_YIJ * ( hub_weight_y * hub_weight_y );
          const float deriv_Yij = ( ( df_dy1 - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 * betaTX +
                                    ( df_dy1 - dy105_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 * betaTX ) * oneL;

          df_dy1_t += betaTX * dW_YIJ * (uij_1 + uij_2) * deriv_Yij;
          dgwx1 -= localGXMult * (uij_2 * dW_YIJ * uij_2 + uij_1 * dW_YIJ * uij_1) * 0.5 * deriv_Yij;
        }
      }
      // now y2
      {
        const float uij_1 = (w < bv.W - 1) ? (y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] - y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] ) : 0; // * wys_0;
        const float uij_2 = (h < bv.H - 1) ? (y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] - y2_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 1] ) : 0; // * wxs_2;
        const float hub_weight_y = rsqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );

        const float W_YIJ  = min(1.f , hub_weight_y * __HDeltaY2__);

        if ( w < bv.W - 1 )
          df_dy2_t -= betaTY * oneL * W_YIJ * ( df_dy2 - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] ); // in x and y dir laplace: ok
        if ( h < bv.H - 1 )
          df_dy2_t -= betaTY * oneL * W_YIJ * ( df_dy2 - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] ); // in x and y dir laplace: ok

        dgwx2 += oneL * localGXMult * ( ( df_dy2 - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 +
                                        ( df_dy2 - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 ) * W_YIJ;
        if ( W_YIJ < 1.0 )
        {
          const float dW_YIJ =  W_YIJ  * ( hub_weight_y * hub_weight_y );
          const float deriv_Yij = ( ( df_dy2 - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 * betaTY +
                                    ( df_dy2 - dy205_prev[ch * (COMMON_BLOCK_SIZE_2D_Y + 2) + threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 * betaTY ) * oneL;

          df_dy2_t += betaTY * dW_YIJ * (uij_1 + uij_2) * deriv_Yij;
          dgwx2 -= localGXMult * (uij_2 * dW_YIJ * uij_2 + uij_1 * dW_YIJ * uij_1) * 0.5 * deriv_Yij;
        }
      }
      atomicAdd( &(d_gwx(0, 1, h, w )), dgwx1 );
      atomicAdd( &(d_gwy(0, 1, h, w )), dgwx2 );
    }
/////////////////////////


// had scaled above by #define s_dx 0.1
//df_dy_t  /= s_dx;
//df_dy1_t /= s_dx;
//df_dy2_t /= s_dx;

    d_gxk_m1(1, ch, h, w) = -t * df_dy1_t;
    df_dy1_t = ( d_gxk(1, ch, h, w) + (1. + t) * df_dy1_t );
#ifdef clipLocalGrad
    df_dy1_t = max( -clipLocalGrad, min (df_dy1_t, clipLocalGrad) );
#endif
// overelaxed version .. ( d_gxk(n,ch,h,w) + (1. + t) * df_dy_t )
    d_gxk05_out(1, ch, h, w)  = df_dy1_t;
    d_gxk(1, ch, h, w)        = df_dy1_t;

    d_gxk_m1(2, ch, h, w) = -t * df_dy2_t;
    df_dy2_t = ( d_gxk(2, ch, h, w) + (1. + t) * df_dy2_t );
#ifdef clipLocalGrad
    df_dy2_t = max( -clipLocalGrad, min (df_dy2_t, clipLocalGrad) );
#endif
// overelaxed version .. ( d_gxk(n,ch,h,w) + (1. + t) * df_dy_t )
    d_gxk05_out(2, ch, h, w)  = df_dy2_t;
    d_gxk(2, ch, h, w)        = df_dy2_t;

////////////////////////////////////////////////////////////////////////
/////////////////////////////// steps 6 and 7 ///////////////////
    float df_dxk = ( d_gxk(0, ch, h, w) + (1. + t) * df_dy_t );
#ifdef clipLocalGrad
    df_dxk = max( -clipLocalGrad, min (df_dxk, clipLocalGrad) );
#endif
    d_gxk(0, ch, h, w)    = df_dxk; // TODO: WRITTEN JUST FOR DEBUGGING -> if runnig can just go ..
    d_gxk_m1(0, ch, h, w) = -t * df_dy_t;
////////////////////////////////////////////////////////////////////////
// steps 2,3,4
    const float bb   = bv(0, ch, h, w);

#ifdef __Huber1__

    float diff = bb - xv_m05(3 * it, ch, h, w);
    float sign = (diff < 0) ? -1. : 1.;
    if ( 1. + conf >= sign * diff )
    {
      atomicAdd( &(d_gc(0, 0, h, w)), df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(3 * it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult );
      d_gxk05_out(0, ch, h, w)  = df_dxk / (1.f + conf);
      d_gb(0, ch, h, w)        += df_dxk * conf / (1.f + conf) * localGBMult;
    }
    else // l1 case
    {
      if ( conf >= sign * diff )
      {
        d_gxk05_out(0, ch, h, w) = 0;
        d_gb(0, ch, h, w)       += df_dxk * localGBMult;
      }
      else
      {
        d_gxk05_out(0, ch, h, w) = df_dxk;
        atomicAdd( &(d_gc(0, 0, h, w)), sign * df_dxk * localGCMult );
      }
    }

#else // not huber case any more /////////////////////////////////

#ifdef __quadratic__
// x = (conf.*b+x05)./(1+conf); derivative is
// df/db    = df/dx * conf/(1+conf)
// df/dx05  = df/dx *  1/(1+conf)
// df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
    atomicAdd( &(d_gc(0, 0, h, w)), df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(3 * it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult );
    d_gxk05_out(0, ch, h, w)  = df_dxk / (1.f + conf);
    d_gb(0, ch, h, w)        += df_dxk * conf / (1.f + conf) * localGBMult;
#else

    float diff = bb - xv_m05(3 * it, ch, h, w);
    float sign = (diff < 0) ? -1. : 1.;
    if ( conf >= sign * diff )
    {
      d_gxk05_out(0, ch, h, w) = 0;
      d_gb(0, ch, h, w)       += df_dxk * localGBMult;
    }
    else
    {
      d_gxk05_out(0, ch, h, w) = df_dxk;
      atomicAdd( &(d_gc(0, 0, h, w)), sign * df_dxk * localGCMult );
    }
#endif // quadratic

#endif // huber 
////////////////////////////////
  }
};


////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
/// this here can doe huber, extending FistaInpaint_end_2D_tex_bw by this functionality. to that end we have more input .. !
__global__ void TGVB2_end_2D_tex_bw_yk( 
   cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf, cudaTextureObject_t betaX, 
   cudaTextureObject_t betaY,  iu::TensorGpu_32f::TensorKernelData bv, 
   iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is !! k-0.5 !! at it, at it-1 the one before that
   iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb,
   iu::TensorGpu_32f::TensorKernelData d_gwx, iu::TensorGpu_32f::TensorKernelData d_gwy, iu::TensorGpu_32f::TensorKernelData yv_0,
   iu::TensorGpu_32f::TensorKernelData d_gxk05_in, iu::TensorGpu_32f::TensorKernelData d_gxk05_out,
   iu::TensorGpu_32f::TensorKernelData d_gxk, iu::TensorGpu_32f::TensorKernelData d_gxk_m1,
   short it, short channels = 1, float t = 1, float nits = 1., float oneL = 0.08332 )
{
  const short n = 0;
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.

  __shared__ float y_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float y1_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! x-direction
  __shared__ float y2_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! y-direction

  __shared__ float dx05_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float dy105_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float dy205_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!


  for ( short ch = 0; ch < channels; ch++ )
  {
    const float yk   = yv_0(n, ch, h, w); //const float yk   = d_y(n, ch, h, w);
    short local_unrolled_x, local_unrolled_y, global_unrolled_x, global_unrolled_y;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    // onto 2 tiling .. alright ..
    local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    if (ch > 0) __syncthreads(); // since y has to be loaded completely
    y_prev [local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv_0(n, ch, global_unrolled_y, global_unrolled_x) : 0;
    y1_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv_0(n + 1, ch, global_unrolled_y, global_unrolled_x) : 0;
    y2_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv_0(n + 2, ch, global_unrolled_y, global_unrolled_x) : 0;
    dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;
    dy105_prev[local_unrolled_y][local_unrolled_x]  = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n + 1, ch, global_unrolled_y, global_unrolled_x) : 0;
    dy205_prev[local_unrolled_y][local_unrolled_x]  = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n + 2, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // load 2
    if ( local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
    {
      y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv_0(n, ch, global_unrolled_y, global_unrolled_x) : 0;
      y1_prev[local_unrolled_y][local_unrolled_x]  = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv_0(n + 1, ch, global_unrolled_y, global_unrolled_x) : 0;
      y2_prev[local_unrolled_y][local_unrolled_x]  = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv_0(n + 2, ch, global_unrolled_y, global_unrolled_x) : 0;
      dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;
      dy105_prev[local_unrolled_y][local_unrolled_x]  = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n + 1, ch, global_unrolled_y, global_unrolled_x) : 0;
      dy205_prev[local_unrolled_y][local_unrolled_x]  = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n + 2, ch, global_unrolled_y, global_unrolled_x) : 0;
    }
    __syncthreads(); // since y has to be loaded completely
////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( !(w >= bv.W || h >= bv.H) ) // well need to load also beyond borders the block could be
    {
/////////////////////////////// steps 5:      y = y + (1/L)*laplace_W*y;
      const float df_dy    = dx05_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy_t = df_dy;//temporary it is only ..
      const float df_dy1   = dy105_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy1_t = df_dy1;// 1 + zij dW/dy - W_t  .. scaled! by 1/L !! so extra !
      const float df_dy2   = dy205_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy2_t = df_dy2;// 1 + zij dW/dy - W_t  .. scaled! by 1/L !! so extra !

#ifdef __HuberRegularizer__
      if ( h > 0 ) // df/dy = df/dx*dx/dy - df/dx_{i,j}-e_2 * dx_{i,j}-e_2/dy = df/dx * dy/dx - df/dx_{i,j}-e_2 * dy/dx_{i,j}-e_2 [symmetry!]
      {
        const float betaTX = tex2D<float>(betaX, w + 0.5f, h - 0.5f);// +0.5 is the exact location w,h, w-0.5 is pixel w-1.
        const float betaTY = tex2D<float>(betaY, w + 0.5f, h - 0.5f);
        //const float wxs_2 = (h > 0 && w < bv.W-1) ? tex2D<float>(dx_sq, w + 0.5f, h - 0.5f) : 0;// *L
        const float wx_2  = (h > 0 && w < bv.W - 1) ? tex2D<float>(dx,    w + 0.5f, h - 0.5f) : 0; // *L
        // needed later on ..
        {
          const float zij_1 = (w < bv.W - 1) ? ( (y_prev[threadIdx.y][threadIdx.x + 2] - y_prev[threadIdx.y][threadIdx.x + 1] ) - y1_prev[threadIdx.y  ][threadIdx.x + 1] ) : 0;
          const float zij_2 =                ( (yk                                 - y_prev[threadIdx.y][threadIdx.x + 1] ) - y2_prev[threadIdx.y  ][threadIdx.x + 1] );
          const float hub_weight_x = sqrtf( (zij_2 * zij_2 * wy_0 + zij_1 * zij_1 * wx_2 ) );

          const float W_TIJ  = 1. / max(1.f , hub_weight_x / __HDelta2__);
          df_dy_t  -= oneL * W_TIJ * ( (df_dy  - dx05_prev[threadIdx.y][threadIdx.x + 1]) - dy205_prev[threadIdx.y][threadIdx.x + 1] ) * wy_0;
          if ( hub_weight_x > __HDelta2__ )
          {
            const float dW_TIJ =  W_TIJ  / ( hub_weight_x * hub_weight_x ); //delta/2 * A^{-3/2} // note that 1/2 cancels later anyway ..
            const float a1 = zij_1 * wx_2;
            const float a2 = zij_2 * wy_0;

            df_dy_t += oneL * (dW_TIJ * a2) * ( a1 * (dx05_prev[threadIdx.y  ][threadIdx.x + 2] - dx05_prev[threadIdx.y][threadIdx.x + 1] - dy105_prev[threadIdx.y][threadIdx.x + 1]) +
                                                a2 * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y][threadIdx.x + 1] - dy205_prev[threadIdx.y][threadIdx.x + 1]) );
          }
        }
        {
          const float uij_1 = (w < bv.W - 1) ? (y1_prev[threadIdx.y  ][threadIdx.x + 2] - y1_prev[threadIdx.y][threadIdx.x + 1] ) : 0; // * wxs_2;
          const float uij_2 =                (y1_prev[threadIdx.y + 1][threadIdx.x + 1] - y1_prev[threadIdx.y][threadIdx.x + 1] ); // * wys_0;
          const float hub_weight_y = sqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX );

          // linear case -- no huber .. shit is not on y but on other stuff !
          const float W_TIJ  = 1. / max(1.f , hub_weight_y / __HDeltaY1__);
          df_dy1_t -= betaTX * oneL * W_TIJ * (df_dy1 - dy105_prev[threadIdx.y][threadIdx.x + 1]);
          if ( hub_weight_y > __HDeltaY1__ )
          {
            const float dW_TIJ =  W_TIJ  / ( hub_weight_y * hub_weight_y );
            df_dy1_t += betaTX * oneL * (dW_TIJ * betaTX * uij_2) * ( uij_1 * (dy105_prev[threadIdx.y  ][threadIdx.x + 2] - dy105_prev[threadIdx.y][threadIdx.x + 1] ) +
                        uij_2 * (dy105_prev[threadIdx.y + 1][threadIdx.x + 1] - dy105_prev[threadIdx.y][threadIdx.x + 1] ) );
          }
        }
        {
          const float uij_1 =  (w < bv.W - 1) ? (y2_prev[threadIdx.y  ][threadIdx.x + 2] - y2_prev[threadIdx.y][threadIdx.x + 1] ) : 0; // * wxs_2;
          const float uij_2 =                 (y2_prev[threadIdx.y + 1][threadIdx.x + 1] - y2_prev[threadIdx.y][threadIdx.x + 1] ); // * wys_0;
          const float hub_weight_y = sqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );

          // linear case -- no huber .. shit is not on y but on other stuff !
          const float W_TIJ  = 1. / max(1.f , hub_weight_y / __HDeltaY2__);
          df_dy2_t -= betaTY * oneL * W_TIJ *   (df_dy2 - dy205_prev[threadIdx.y][threadIdx.x + 1]);

          if ( hub_weight_y > __HDeltaY2__ )
          {
            const float dW_TIJ =  W_TIJ  / ( hub_weight_y * hub_weight_y );
            df_dy2_t += betaTY * oneL * (dW_TIJ * betaTY * uij_2) * ( uij_1 * (dy205_prev[threadIdx.y  ][threadIdx.x + 2] - dy205_prev[threadIdx.y][threadIdx.x + 1] ) +
                        uij_2 * (dy205_prev[threadIdx.y + 1][threadIdx.x + 1] - dy205_prev[threadIdx.y][threadIdx.x + 1] ) );
          }
        }
      }//h>0

      if ( w > 0 ) // df/dy = df/dx*dx/dy - df/dx_{i,j}-e_2 * dx_{i,j}-e_2/dy = df/dx * dy/dx - df/dx_{i,j}-e_2 * dy/dx_{i,j}-e_2 [symmetry!]
      {
        const float betaTX = tex2D<float>(betaX, w - 0.5f, h + 0.5f);// +0.5 is the exact location w,h, w-0.5 is pixel w-1.
        const float betaTY = tex2D<float>(betaY, w - 0.5f, h + 0.5f);
        //const float wys_2 = (w > 0 && h < bv.H-1) ? tex2D<float>(dy_sq, w - 0.5f, h + 0.5f) : 0;
        const float wy_2  = (w > 0 && h < bv.H - 1) ? tex2D<float>(dy,    w - 0.5f, h + 0.5f) : 0;
        {
          const float zij_1 =                ( yk                                 - y_prev[threadIdx.y + 1][threadIdx.x] - y1_prev[threadIdx.y + 1][threadIdx.x] );
          const float zij_2 = (h < bv.H - 1) ? ( y_prev[threadIdx.y + 2][threadIdx.x] - y_prev[threadIdx.y + 1][threadIdx.x] - y2_prev[threadIdx.y + 1][threadIdx.x] ) : 0;
          const float hub_weight_x = sqrtf( (zij_2 * zij_2 * wy_2 + zij_1 * zij_1 * wx_0 ) );

          const float W_TIJ  = 1. / max(1.f , hub_weight_x / __HDelta2__);
          df_dy_t  -= oneL * W_TIJ * ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x]) - dy105_prev[threadIdx.y + 1][threadIdx.x] ) * wx_0;
          if ( hub_weight_x > __HDelta2__ )
          {
            const float dW_TIJ =  W_TIJ  / ( hub_weight_x * hub_weight_x ); //delta/2 * A^{-3/2} // note that 1/2 cancels later anyway ..
            const float a1 = zij_1 * wx_0;
            const float a2 = zij_2 * wy_2;

            df_dy_t += oneL * (dW_TIJ * a1) * ( a1 * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 1][threadIdx.x] - dy105_prev[threadIdx.y + 1][threadIdx.x]) +
                                                a2 * (dx05_prev[threadIdx.y + 2][threadIdx.x  ] - dx05_prev[threadIdx.y + 1][threadIdx.x] - dy205_prev[threadIdx.y + 1][threadIdx.x]) );
          }
        }
        {
          const float uij_1 =                 (y1_prev[threadIdx.y + 1][threadIdx.x + 1] - y1_prev[threadIdx.y + 1][threadIdx.x  ] ); // wxs_0;
          const float uij_2 = (h < bv.H - 1) ?  (y1_prev[threadIdx.y + 2][threadIdx.x  ] - y1_prev[threadIdx.y + 1][threadIdx.x  ] ) : 0; // * wys_2;
          const float hub_weight_y = sqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX );

          const float W_TIJ  = 1. / max(1.f , hub_weight_y / __HDeltaY1__);
          df_dy1_t -= betaTX * oneL * W_TIJ * (df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x]); //
          if ( hub_weight_y > __HDeltaY1__ )
          {
            const float dW_TIJ =  W_TIJ  / ( hub_weight_y * hub_weight_y );
            df_dy1_t += betaTX * oneL * (dW_TIJ * betaTX * uij_1) * ( uij_2 * ( dy105_prev[threadIdx.y + 2][threadIdx.x  ] - dy105_prev[threadIdx.y + 1][threadIdx.x] ) +
                        uij_1 * ( dy105_prev[threadIdx.y + 1][threadIdx.x + 1] - dy105_prev[threadIdx.y + 1][threadIdx.x] ) );
          }
        }
        {
          const float uij_1 =                (y2_prev[threadIdx.y + 1][threadIdx.x + 1] - y2_prev[threadIdx.y + 1][threadIdx.x  ] );
          const float uij_2 = (h < bv.H - 1) ? (y2_prev[threadIdx.y + 2][threadIdx.x  ] - y2_prev[threadIdx.y + 1][threadIdx.x  ] ) : 0;
          const float hub_weight_y = sqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );

          const float W_TIJ  = 1. / max(1.f , hub_weight_y / __HDeltaY2__);
          df_dy2_t -= betaTY * oneL * W_TIJ *   (df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x]);
          if ( hub_weight_y > __HDeltaY2__ )
          {
            const float dW_TIJ =  W_TIJ  / ( hub_weight_y * hub_weight_y );
            df_dy2_t += betaTY * oneL * (dW_TIJ * betaTY * uij_1) * ( uij_2 * ( dy205_prev[threadIdx.y + 2][threadIdx.x  ] - dy205_prev[threadIdx.y + 1][threadIdx.x] ) +
                        uij_1 * ( dy205_prev[threadIdx.y + 1][threadIdx.x + 1] - dy205_prev[threadIdx.y + 1][threadIdx.x] ) );
          }
        }
      }
      // h< bv.H-1 h< bv.W-1
      // if covered by setting weights to 0. //if ( h < xv.H-1 && w < xv.W-1 )
      {
        const float betaTX = tex2D<float>(betaX, w + 0.5f, h + 0.5f);// +0.5 is the exact location w,h, w-0.5 is pixel w-1.
        const float betaTY = tex2D<float>(betaY, w + 0.5f, h + 0.5f);
        { // xij
          const float aij_1 = (w < bv.W - 1) ? ( (y_prev[threadIdx.y + 1][threadIdx.x + 2] - y_prev[threadIdx.y + 1][threadIdx.x + 1] ) - y1_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0;
          const float aij_2 = (h < bv.H - 1) ? ( (y_prev[threadIdx.y + 2][threadIdx.x + 1] - y_prev[threadIdx.y + 1][threadIdx.x + 1] ) - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0;

          const float hub_weight_x = sqrtf( (aij_2 * aij_2 * wy_1 + aij_1 * aij_1 * wx_1 ) ) ;
          const float W_TIJ  = 1. / max(1.f , hub_weight_x / __HDelta2__);

          df_dy_t  -= oneL *  ( (df_dy  - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) + dy205_prev[threadIdx.y + 1][threadIdx.x + 1] ) * wy_1 * W_TIJ;
          df_dy_t  -= oneL *  ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) + dy105_prev[threadIdx.y + 1][threadIdx.x + 1] ) * wx_1 * W_TIJ;

          df_dy1_t -= oneL * W_TIJ *  (  df_dy1 - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] + df_dy ) * wx_1; //
          df_dy2_t -= oneL * W_TIJ *  (  df_dy2 - dx05_prev[threadIdx.y + 2][threadIdx.x + 1] + df_dy ) * wy_1; // blue alternative

          d_gwx(n, 0, h,   w  ) = d_gwx(n, 0, h,   w  ) + oneL * localGXMult * ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] + df_dy1 ) * aij_1 * W_TIJ;
          d_gwy(n, 0, h,   w  ) = d_gwy(n, 0, h,   w  ) + oneL * localGXMult * ( df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1] + df_dy2 ) * aij_2 * W_TIJ;

          if ( hub_weight_x > __HDelta2__ )
          {
            const float dW_TIJ =  W_TIJ  / ( hub_weight_x * hub_weight_x );
            const float a2 = aij_2 * wy_1;
            const float a1 = aij_1 * wx_1;
            ///////////////////////////////////
            const float deriv_Tij = ( ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] + df_dy1 ) * a1 +
                                      ( df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1] + df_dy2 ) * a2 ) * oneL;
            df_dy_t  +=  deriv_Tij * ((a1 + a2) * dW_TIJ) ;
            df_dy2_t +=  deriv_Tij * (a2 * dW_TIJ) ;
            df_dy1_t +=  deriv_Tij * (a1 * dW_TIJ) ;
            ///////////////////////////////////
            d_gwy(0, 0, h,   w  ) = d_gwy(0, 0, h,   w  ) - localGXMult * (aij_2 * dW_TIJ * aij_2) * 0.5 * deriv_Tij;
            d_gwx(0, 0, h,   w  ) = d_gwx(0, 0, h,   w  ) - localGXMult * (aij_1 * dW_TIJ * aij_1) * 0.5 * deriv_Tij;
          }
        }
        { // now y1
          const float uij_1 = (w < bv.W - 1) ? (y1_prev[threadIdx.y + 1][threadIdx.x + 2] - y1_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0;
          const float uij_2 = (h < bv.H - 1) ? (y1_prev[threadIdx.y + 2][threadIdx.x + 1] - y1_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0;
          const float hub_weight_y = sqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTX );
          const float W_YIJ  = 1. / max(1.f , hub_weight_y / __HDeltaY1__);

          if ( h < bv.H - 1 )
            df_dy1_t -= betaTX * oneL * W_YIJ *  ( df_dy1 - dy105_prev[threadIdx.y + 2][threadIdx.x + 1] );
          if ( w < bv.W - 1 )
            df_dy1_t -= betaTX * oneL * W_YIJ *  ( df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x + 2] );

          d_gwx(0, 1, h, w ) = d_gwx(0, 1, h, w ) + oneL * localGXMult * ( ( df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 +
                               ( df_dy1 - dy105_prev[threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 ) * W_YIJ;
          if ( hub_weight_y > __HDeltaY1__ ) // open
          {
            const float dW_YIJ =  W_YIJ  / ( hub_weight_y * hub_weight_y );
            const float deriv_Yij = ( ( df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 * betaTX +
                                      ( df_dy1 - dy105_prev[threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 * betaTX ) * oneL;

            df_dy1_t += betaTX * dW_YIJ * (uij_1 + uij_2) * deriv_Yij;
            d_gwx(0, 1, h, w  ) = d_gwx(0, 1, h, w  ) - localGXMult * (uij_2 * dW_YIJ * uij_2 + uij_1 * dW_YIJ * uij_1) * 0.5 * deriv_Yij;
          }
        }
        // now y2
        {
          const float uij_1 = (w < bv.W - 1) ? (y2_prev[threadIdx.y + 1][threadIdx.x + 2] - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0; // * wys_0;
          const float uij_2 = (h < bv.H - 1) ? (y2_prev[threadIdx.y + 2][threadIdx.x + 1] - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) : 0; // * wxs_2;
          const float hub_weight_y = sqrtf( (uij_1 * uij_1 + uij_2 * uij_2 ) * betaTY );
          const float W_YIJ  = 1. / max(1.f , hub_weight_y / __HDeltaY2__);

          if ( w < bv.W - 1 )
            df_dy2_t -= betaTY * oneL * W_YIJ *  ( df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x + 2] );
          if ( h < bv.H - 1 )
            df_dy2_t -= betaTY * oneL * W_YIJ *  ( df_dy2 - dy205_prev[threadIdx.y + 2][threadIdx.x + 1] );

          d_gwy(0, 1, h, w ) = d_gwy(0, 1, h, w ) + oneL * localGXMult * ( ( df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 +
                               ( df_dy2 - dy205_prev[threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 ) * W_YIJ;
          if ( hub_weight_y > __HDeltaY2__ )
          {
            const float dW_YIJ =  W_YIJ  / ( hub_weight_y * hub_weight_y );
            const float deriv_Yij = ( ( df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x + 2] ) * uij_1 * betaTY +
                                      ( df_dy2 - dy205_prev[threadIdx.y + 2][threadIdx.x + 1] ) * uij_2 * betaTY ) * oneL;

            df_dy2_t += betaTY * dW_YIJ * (uij_1 + uij_2) * deriv_Yij;
            d_gwy(0, 1, h, w  ) = d_gwy(0, 1, h, w  ) - localGXMult * (uij_2 * dW_YIJ * uij_2 + uij_1 * dW_YIJ * uij_1) * 0.5 * deriv_Yij;
          }
        }
      }
/////////////////////////
#else // no huber on it -> a lot simpler 

      const float wxs_0 = (w > 0 )     ? tex2D<float>(dx_sq, w - 0.5f, h + 0.5f) : 0;// *L
      const float wxs_1 = (w < bv.W - 1) ? tex2D<float>(dx_sq, w + 0.5f, h + 0.5f) : 0;
      const float wys_0 = (h > 0 )     ? tex2D<float>(dy_sq, w + 0.5f, h - 0.5f) : 0;
      const float wys_1 = (h < bv.H - 1) ? tex2D<float>(dy_sq, w + 0.5f, h + 0.5f) : 0;

      if ( h > 0 ) // df/dy = df/dx*dx/dy - df/dx_{i,j}-e_2 * dx_{i,j}-e_2/dy = df/dx * dy/dx - df/dx_{i,j}-e_2 * dy/dx_{i,j}-e_2 [symmetry!]
      {
        df_dy_t  -= oneL *  ( (df_dy  - dx05_prev[threadIdx.y][threadIdx.x + 1]) - dy205_prev[threadIdx.y][threadIdx.x + 1] ) * wys_0 * wys_0;
        df_dy1_t -= oneL *   (df_dy1 - dy105_prev[threadIdx.y][threadIdx.x + 1]) * beta; //
        df_dy2_t -= oneL *   (df_dy2 - dy205_prev[threadIdx.y][threadIdx.x + 1]) * beta; //
      }
      if ( h < bv.H - 1 )
      {
        df_dy_t  -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) + df_dy2 ) * wys_1 * wys_1; //L1
        df_dy1_t -= oneL * (                                                                   (df_dy1 - dy105_prev[threadIdx.y + 2][threadIdx.x + 1]) * beta );
        df_dy2_t -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wys_1 * wys_1 + (df_dy2 - dy205_prev[threadIdx.y + 2][threadIdx.x + 1]) * beta );
        df_dy2_t -= oneL * df_dy2 * wys_1 * wys_1;
      }
      if ( w > 0 )
      {
        df_dy_t  -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x]) - dy105_prev[threadIdx.y + 1][threadIdx.x] ) * wxs_0 * wxs_0;
        df_dy1_t -= oneL *   (df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x]) * beta; //
        df_dy2_t -= oneL *   (df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x]) * beta; //
      }
      if ( w < bv.W - 1 )
      {
        df_dy_t  -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] ) + df_dy1 ) * wxs_1 * wxs_1; //L2
        df_dy1_t -= oneL * ( (df_dy  - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] ) * wxs_1 * wxs_1 + (df_dy1 - dy105_prev[threadIdx.y + 1][threadIdx.x + 2]) * beta );
        df_dy2_t -= oneL * (                                                                  + (df_dy2 - dy205_prev[threadIdx.y + 1][threadIdx.x + 2]) * beta );
        df_dy1_t -= oneL * df_dy1 * wxs_1 * wxs_1; //L5
      }

      if (w < bv.W - 1)
        d_gwx(n, 0, h,   w  ) = d_gwx(n, 0, h,   w  ) + oneL * localGXMult * (
                                  (y_prev[threadIdx.y + 1][threadIdx.x + 2] - yk ) * ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) // part1: '-' of it ?
                                  - y1_prev[threadIdx.y + 1][threadIdx.x + 1]       * ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) // part2
                                  + (y_prev[threadIdx.y + 1][threadIdx.x + 2] - yk ) * df_dy1 // part3
                                  - y1_prev[threadIdx.y + 1][threadIdx.x + 1]        * df_dy1
                                );

      if (h < bv.H - 1) // really important !
        d_gwy(n, 0, h,   w  ) = d_gwy(n, 0, h,   w  ) + oneL * localGXMult  * (
                                  (y_prev[threadIdx.y + 2][threadIdx.x + 1] - yk - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) * ( df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1])
                                  + (y_prev[threadIdx.y + 2][threadIdx.x + 1] - yk - y2_prev[threadIdx.y + 1][threadIdx.x + 1] ) *   df_dy2
                                );
#endif // case of Huber on regularizer .. or not .. 

      d_gxk_m1(n + 1, ch, h, w) = -t * df_dy1_t;
      df_dy1_t = ( d_gxk(n + 1, ch, h, w) + (1. + t) * df_dy1_t );
#ifdef clipLocalGrad
      df_dy1_t = max( -clipLocalGrad, min (df_dy1_t, clipLocalGrad) );
#endif
      d_gxk(n + 1, ch, h, w)    = df_dy1_t;

      d_gxk_m1(n + 2, ch, h, w) = -t * df_dy2_t;
      df_dy2_t = ( d_gxk(n + 2, ch, h, w) + (1. + t) * df_dy2_t );
#ifdef clipLocalGrad
      df_dy2_t = max( -clipLocalGrad, min (df_dy2_t, clipLocalGrad) );
#endif
      d_gxk(n + 2, ch, h, w)    = df_dy2_t;

// overelaxed version .. ( d_gxk(n,ch,h,w) + (1. + t) * df_dy_t )
      d_gxk05_out(n + 1, ch, h, w)  = df_dy1_t;
      d_gxk05_out(n + 2, ch, h, w)  = df_dy2_t;

////////////////////////////////////////////////////////////////////////
/////////////////////////////// steps 6 and 7 ///////////////////
      float df_dxk = d_gxk(n, ch, h, w) + (1. + t) * df_dy_t;
#ifdef clipLocalGrad
      df_dxk = max( -clipLocalGrad, min (df_dxk, clipLocalGrad) );
#endif
      d_gxk(n, ch, h, w)    = df_dxk;
      d_gxk_m1(n, ch, h, w) = -t * df_dy_t;
////////////////////////////////////////////////////////////////////////
// steps 2,3,4
      const float bb   = bv(n, ch, h, w);

#ifdef __Huber1__

      const float diff = bb - xv_m05(3 * it, ch, h, w);
      const float sign = (diff < 0) ? -1. : 1.;
      if ( 1. + conf >= sign * diff )
      {
// quadratic case EXACTLY:
// x = (conf.*b+x05)./(1+conf); derivative is
// df/dconf = df/dx *  b./(1+conf) - conf*b/*(1+conf*conf)
// df/db    = df/dx * conf/(1+conf)
// df/dy    = df/dx *  1/(1+conf)
// df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
        d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(3 * it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
        d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
        d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf)  * localGBMult;
      }
      else // L1 case exactly
      {
        if ( conf >= sign * diff )
        {
          d_gxk05_out(n, ch, h, w) = 0;
          d_gb(n, ch, h, w)       += df_dxk * localGBMult;
        }
        else
        {
          d_gxk05_out(n, ch, h, w) = df_dxk;
          d_gc(n, 0, h, w)       += sign * df_dxk * localGCMult;
        }
      }

#else // not huber:

#ifdef __quadratic__

      d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(3 * it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
      d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
      d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf)  * localGBMult;

#else

      const float diff = bb - xv_m05(3 * it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      if ( conf >= sign * diff )
      {
        d_gxk05_out(n, ch, h, w) = 0;
        d_gb(n, ch, h, w)       += df_dxk * localGBMult;
      }
      else
      {
        d_gxk05_out(n, ch, h, w) = df_dxk;
        d_gc(n, 0, h, w)       += sign * df_dxk * localGCMult;
      }

#endif // not quadratic -> L1

#endif // HUBER
////////////////////////////////
    }
  }
};
///////////////////////////////////////////
// compute yk from xk+0.5 etc ..

/// forward step, computing y^k+1 from x^k+0.5 -> data and fista overelax ..
__global__ void TGVB2_simple_get_yk(
  cudaTextureObject_t cf, iu::TensorGpu_32f::TensorKernelData bv, 
  iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is !! k-0.5 !! at it, at it-1 the one before that
  iu::TensorGpu_32f::TensorKernelData yv_out_k, short it, short channels = 1, float t = 1 )
{
  const short n = 0;

  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  for ( short ch = 0; ch < channels; ch++ )
  {
    if ( !(w >= bv.W || h >= bv.H) ) // well need to load also beyond borders the block could be
    {
// cases needed .. to know the weights and hence the cases I need to
////////////////////////////////////////////////////////////////////////
// steps 2,3,4
      const float bb   = bv(n, ch, h, w);

#ifdef __Huber1__

      float diff = bb - xv_m05(3 * it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      float yk(0);
      if ( 1. + conf >= sign * diff )
      {
        const float xk  = (conf * bb + xv_m05(3 * it,     ch, h, w)) / (1.f + conf);
        const float xkm = (conf * bb + xv_m05(3 * (it - 1), ch, h, w)) / (1.f + conf);
        yk  = xk + t * (xk - xkm);
      }
      else // l1 case
      {
        const float xk   = ( conf <= sign * diff   ) ? bb - diff + sign * conf : bb;
        diff = bb - xv_m05(3 * (it - 1), ch, h, w);  sign = (diff < 0) ? -1. : 1.;
        yk   = xk + t * (xk - (( conf <= sign * diff   ) ? bb - diff + sign * conf : bb) );
      }

#else // not huber case any more /////////////////////////////////

#ifdef __quadratic__
      const float xk  = (conf * bb + xv_m05(3 * it,     ch, h, w)) / (1.f + conf);
      const float xkm = (conf * bb + xv_m05(3 * (it - 1), ch, h, w)) / (1.f + conf);
      const float yk  = xk + t * (xk - xkm);
#else

      float diff = bb - xv_m05(3 * it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;

      const float xk   = ( conf <= sign * diff   ) ? bb - diff + sign * conf : bb;
      diff = bb - xv_m05(3 * (it - 1), ch, h, w);  sign = (diff < 0) ? -1. : 1.;
      const float yk   = xk + t * (xk - (( conf <= sign * diff   ) ? bb - diff + sign * conf : bb) );

#endif // quadratic
#endif // huber 

      yv_out_k(n, ch, h, w) = yk;
      yv_out_k(n + 1, ch, h, w) = xv_m05(3 * it + 1, ch, h, w);
      yv_out_k(n + 2, ch, h, w) = xv_m05(3 * it + 2, ch, h, w);

    }
  }
};

///////////////////////////////////////////////

// forward kernel 1: trying to adjust ..
__global__ void TGVB2_simple_nCalls_2D_tex_clean( 
    cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf, cudaTextureObject_t betaX, 
    cudaTextureObject_t betaY, // iu::TensorGpu_32f::TensorKernelData cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv, 
    iu::TensorGpu_32f::TensorKernelData yv, iu::TensorGpu_32f::TensorKernelData yv_out,
    short channels = 1, float t = 1, float oneL = 0.08332)
{
  short w  = blockIdx.x * blockDim.x + threadIdx.x;
  short h  = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.

// this one holds y of previous iteration of the current block of ids PLUS one more to the right (forward differences)
// huber needs to adjust the weights according to the |sqrt(w) nabla x|_2 norm ..

  __shared__ float  y_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float y1_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! x-direction
  __shared__ float y2_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! y-direction

  for (short ch = 0; ch < channels; ch++)
  {
    // loop here over channels ..
    // onto 2 tiling .. alright ..
    short local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    short local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    short global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    short global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // read local_unrolled_x + blockDim.x * blockIdx.x, local_unrolled_y + blockDim.y * blockIdx.y -(1,1)
    // write to local id: local_unrolled_y, local_unrolled_x
    // load if inside image
    if (ch > 0) __syncthreads(); // since y has to be loaded completely
    y_prev[local_unrolled_y][local_unrolled_x]    = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(0, ch, global_unrolled_y, global_unrolled_x) : 0;
    y1_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(1, ch, global_unrolled_y, global_unrolled_x) : 0;
    y2_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(2, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;
    // load 2nd part
    if (local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
    {
      y_prev[local_unrolled_y][local_unrolled_x]    = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(0, ch, global_unrolled_y, global_unrolled_x) : 0;
      y1_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(1, ch, global_unrolled_y, global_unrolled_x) : 0;
      y2_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(2, ch, global_unrolled_y, global_unrolled_x) : 0;
    }
    __syncthreads(); // since y has to be loaded completely

    if ( !(w >= bv.W || h >= bv.H) )
    {
      const float x_   = xv(0, ch, h, w);
      const float b    = bv(0, ch, h, w); // as in the right hand side ..
      float x_loc      = b;
      //////////////////////////// in the loop?
      //     y = y + (1/L)*laplace_W*y;
      float yc  = y_prev[threadIdx.y + 1][threadIdx.x + 1];
      float x05 = yc;
#ifdef __HuberRegularizer__
      {
        float y1_new = y1_prev[threadIdx.y + 1][threadIdx.x + 1]; // its a gradient update .. !
        float y2_new = y2_prev[threadIdx.y + 1][threadIdx.x + 1];
        const float y1_c = y1_new;
        const float y2_c = y2_new;
        if ( h > 0 ) // backward-diff
        {
          const float betaTX = tex2D<float>(betaX, w + 0.5f, h - 0.5f);
          const float betaTY = tex2D<float>(betaY, w + 0.5f, h - 0.5f);
          const float wx_2  = (h > 0 && w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h - 0.5f) : 0; // *L

          const float zij_1 = (w < bv.W - 1) ? ( (y_prev[threadIdx.y][threadIdx.x + 2] - y_prev[threadIdx.y][threadIdx.x + 1] ) - y1_prev[threadIdx.y  ][threadIdx.x + 1] ) : 0;
          const float zij_2 =                ( (yc                                 - y_prev[threadIdx.y][threadIdx.x + 1] ) - y2_prev[threadIdx.y  ][threadIdx.x + 1] );

          const float hub_weight_x = min(1.f, rsqrtf( (zij_2 * zij_2 * wy_0 + zij_1 * zij_1 * wx_2 ) ) * __HDelta2__ ); // min(1., delta * rsqrt( 0 )) = min (1., inf) =1. -> OK

          // flip these two and get different results ..
          const float uij1_2 =                ( y1_c                                  - y1_prev[threadIdx.y][threadIdx.x + 1] );
          const float uij1_1 = (w < bv.W - 1) ? ( y1_prev[threadIdx.y]  [threadIdx.x + 2] - y1_prev[threadIdx.y][threadIdx.x + 1] ) : 0;
          const float hub_weight_y1 = min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

          const float uij2_1 = (w < bv.W - 1) ? ( y2_prev[threadIdx.y]  [threadIdx.x + 2] - y2_prev[threadIdx.y][threadIdx.x + 1] ) : 0;
          const float uij2_2 =                ( y2_c                                  - y2_prev[threadIdx.y][threadIdx.x + 1] );
          const float hub_weight_y2 = min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

          x05    -=    wy_0  * oneL  * (zij_2 * hub_weight_x);
          y1_new -=   (betaTX * oneL) * (uij1_2 * hub_weight_y1);
          y2_new -=   (betaTY * oneL) * (uij2_2 * hub_weight_y2);
        }
        if ( w > 0 ) // backward, so (nabla x - y) * 1
        {
          const float betaTX = tex2D<float>(betaX, w - 0.5f, h + 0.5f);
          const float betaTY = tex2D<float>(betaY, w - 0.5f, h + 0.5f);
          const float wy_2  = (w > 0 && h < bv.H - 1) ? tex2D<float>(dy, w - 0.5f, h + 0.5f) : 0; // stored as wx_0

          const float zij_1 =                ( (yc                                 - y_prev[threadIdx.y + 1][threadIdx.x] ) - y1_prev[threadIdx.y + 1 ][threadIdx.x] );
          const float zij_2 = (h < bv.H - 1) ? ( (y_prev[threadIdx.y + 2][threadIdx.x] - y_prev[threadIdx.y + 1][threadIdx.x] ) - y2_prev[threadIdx.y + 1 ][threadIdx.x] ) : 0;

          const float hub_weight_x =   min(1.f, rsqrtf( (zij_2 * zij_2 * wy_2 + zij_1 * zij_1 * wx_0 ) ) * __HDelta2__ );

          const float uij1_1 =                ( (y1_c                                  - y1_prev[threadIdx.y + 1][threadIdx.x] ) );
          const float uij1_2 = (h < bv.H - 1) ? ( (y1_prev[threadIdx.y + 2][threadIdx.x]   - y1_prev[threadIdx.y + 1][threadIdx.x] ) ) : 0;
          const float hub_weight_y1 =   min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

          const float uij2_1 =                ( (y2_c                                  - y2_prev[threadIdx.y + 1][threadIdx.x] ) );
          const float uij2_2 = (h < bv.H - 1) ? ( (y2_prev[threadIdx.y + 2][threadIdx.x]   - y2_prev[threadIdx.y + 1][threadIdx.x] ) ) : 0;
          const float hub_weight_y2 =   min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

          x05    -=    wx_0  * oneL  * (zij_1 * hub_weight_x); // correct order nabla x - y
          y1_new -=   (betaTX * oneL) * (uij1_1 * hub_weight_y1);
          y2_new -=   (betaTY * oneL) * (uij2_1 * hub_weight_y2);
        }

        const float betaTX = tex2D<float>(betaX, w + 0.5f, h + 0.5f);
        const float betaTY = tex2D<float>(betaY, w + 0.5f, h + 0.5f);
        const float zij_1 = (w < bv.W - 1) ? ((yc  - y_prev[threadIdx.y + 1][threadIdx.x + 2]) + y1_c ) : 0; // flipped order nabla x - y -> -nabla x + y
        const float zij_2 = (h < bv.H - 1) ? ((yc  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) + y2_c ) : 0;

        const float hub_weight_x =    min(1.f, rsqrtf( (zij_2 * zij_2 * wy_1 + zij_1 * zij_1 * wx_1) ) * __HDelta2__ );

        const float uij1_1 = (w < bv.W - 1) ? ( (y1_c                                  - y1_prev[threadIdx.y + 1][threadIdx.x + 2] ) ) : 0;
        const float uij1_2 = (h < bv.H - 1) ? ( (y1_c                                  - y1_prev[threadIdx.y + 2][threadIdx.x + 1] ) ) : 0;
        const float hub_weight_y1 =    min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1) ) * __HDeltaY1__ );

        const float uij2_1 = (w < bv.W - 1) ? ( (y2_c                                  - y2_prev[threadIdx.y + 1][threadIdx.x + 2] ) ) : 0;
        const float uij2_2 = (h < bv.H - 1) ? ( (y2_c                                  - y2_prev[threadIdx.y + 2][threadIdx.x + 1] ) ) : 0;
        const float hub_weight_y2 =    min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1) ) * __HDeltaY2__ );

        x05    -=  wy_1  * oneL  * (zij_2 * hub_weight_x);
        y1_new -= (betaTX * oneL) * (uij1_2 * hub_weight_y1);
        y2_new -= (betaTY * oneL) * (uij2_2 * hub_weight_y2)  + wy_1 * oneL * (zij_2 * hub_weight_x); //seoncd change to W(Dx-y)

        x05    -=   wx_1  * oneL  * (zij_1 * hub_weight_x);
        y1_new -=  (betaTX * oneL) * (uij1_1 * hub_weight_y1)  + wx_1 * oneL * (zij_1 * hub_weight_x); // second change W(Dx-y) !
        y2_new -=  (betaTY * oneL) * (uij2_1 * hub_weight_y2);

        yv_out(1, ch, h, w) = y1_new + t * (y1_new - xv(1, ch, h, w));
        yv_out(2, ch, h, w) = y2_new + t * (y2_new - xv(2, ch, h, w));
        // x^k+1
        xv(1, ch, h, w) = y1_new;
        xv(2, ch, h, w) = y2_new;
      }
#else
      {
        const float wxs_0 = (w > 0 )     ? tex2D<float>(dx_sq, w - 0.5f, h + 0.5f) : 0;// *L
        const float wxs_1 = (w < bv.W - 1) ? tex2D<float>(dx_sq, w + 0.5f, h + 0.5f) : 0;
        const float wys_0 = (h > 0 )     ? tex2D<float>(dy_sq, w + 0.5f, h - 0.5f) : 0;
        const float wys_1 = (h < bv.H - 1) ? tex2D<float>(dy_sq, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.

        float y1_new = y1_prev[threadIdx.y + 1][threadIdx.x + 1]; //its a gradient update ..
        float y2_new = y2_prev[threadIdx.y + 1][threadIdx.x + 1];
        const float y1_c = y1_new;
        const float y2_c = y2_new;
        if ( h > 0 ) // backward-diff
        {
          x05    -=  (  (yc -  y_prev[threadIdx.y][threadIdx.x + 1]) - y2_prev[threadIdx.y  ][threadIdx.x + 1] ) * wys_0 * wys_0 * oneL;
          y1_new -=  beta * (y1_c - y1_prev[threadIdx.y][threadIdx.x + 1]) * oneL;
          y2_new -=  beta * (y2_c - y2_prev[threadIdx.y][threadIdx.x + 1]) * oneL;
        }
        if ( h < bv.H - 1 ) // forward
        {
          const float zij_2 = ((yc  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) + y2_c ) * wys_1 * oneL;
          x05    -=  zij_2 * wys_1;
          y1_new -= beta * ( (y1_c - y1_prev[threadIdx.y + 2][threadIdx.x + 1]) ) * oneL;
          y2_new -= beta * ( (y2_c - y2_prev[threadIdx.y + 2][threadIdx.x + 1]) ) * oneL + zij_2 * wys_1;
        }
        if ( w > 0 ) // backward, so (nabla x - y) * 1
        {
          x05    -=   (  (yc -  y_prev[threadIdx.y + 1][threadIdx.x]) - y1_prev[threadIdx.y + 1][threadIdx.x  ] ) * wxs_0 * wxs_0 * oneL; // correct order nabla x - y
          y1_new -=   beta * (y1_c - y1_prev[threadIdx.y + 1][threadIdx.x]) * oneL;
          y2_new -=   beta * (y2_c - y2_prev[threadIdx.y + 1][threadIdx.x]) * oneL;
        }
        if ( w < bv.W - 1 ) // forward, so (nabla x - y) * -1 = (nabla x - y) * 1
        {
          const float zij_1 = ((yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) + y1_c ) * wxs_1 * oneL; // flipped order nabla x - y -> -nabla x + y
          x05    -=   zij_1 * wxs_1;
          y1_new -=   beta * (y1_c - y1_prev[threadIdx.y + 1][threadIdx.x + 2]) * oneL + zij_1 * wxs_1;
          y2_new -=   beta * (y2_c - y2_prev[threadIdx.y + 1][threadIdx.x + 2]) * oneL;
        }

        yv_out(1, ch, h, w) = y1_new + t * (y1_new - xv(n + 1, ch, h, w));
        yv_out(2, ch, h, w) = y2_new + t * (y2_new - xv(n + 2, ch, h, w));
        // x^k+1
        xv(n + 1, ch, h, w) = y1_new;
        xv(n + 2, ch, h, w) = y2_new;
      }
#endif
////////////////////////////////////////////////
#ifdef __Huber1__

      x_loc = (conf * b + x05) / (1.f + conf);
      yc  = x05 - conf; // *L above ..
      x05 = x05 + conf;
      if (yc  > 1.f + b )  x_loc = yc;
      if (x05 < b - 1.f )  x_loc = x05;

#else // not huber cases   

// data term:
#ifdef __quadratic__
      // x = (conf.*b+x05)./(1+conf); derivative is
      // df/db    = df/dx * conf/(1+conf)
      // df/dx05  = df/dx *  1/(1+conf)
      // df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
      x_loc = (conf * b + x05) / (1.f + conf);
#else // L1 case  
      //     x1 = y-conf./L;
      //     x2 = y+conf./L;
      //     b1 = (x1>x3);
      //     b2 = (x2<x3);
      //     x_= x;
      //     x = x1 .* b1 + x2 .* b2 + x3 .* (1-b1-b2);
      yc  = x05 - conf; // *L above ..
      x05 = x05 + conf;
      if (yc > b)  x_loc = yc;
      if (x05 < b) x_loc = x05;
#endif
#endif // huber or not 
      //
      //     t_=t
      //     t=(1+sqrt(1+4*t_^2))/2;
      //     y = x + (t_-1)/t * (x-x_); % that is y over-relaxed
      yv_out(0, ch, h, w) = x_loc + t * (x_loc - x_); //
      xv    (0, ch, h, w) = x_loc;
    }
  }
}

// forward kernel 1:
template<int channels>
__global__ void TGVB2_simple_nCalls_2D_tex_clean_ch( 
    cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf, cudaTextureObject_t betaX, 
    cudaTextureObject_t betaY, iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv,
    iu::TensorGpu_32f::TensorKernelData yv, iu::TensorGpu_32f::TensorKernelData yv_out, float t = 1, float oneL = 0.08332)
{
  short w  = blockIdx.x * blockDim.x + threadIdx.x;
  short h  = blockIdx.y * blockDim.y + threadIdx.y;
  short ch = blockIdx.z * blockDim.z + threadIdx.z;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

// this one holds y of previous iteration of the current block of ids PLUS one more to the right (forward differences)
// huber needs to adjust the weights according to the |sqrt(w) nabla x|_2 norm ..

  __shared__ float  y_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float y1_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! x-direction
  __shared__ float y2_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! y-direction

// onto 2 tiling .. alright ..
  short local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
  short local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
  short global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
  short global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

// read local_unrolled_x + blockDim.x * blockIdx.x, local_unrolled_y + blockDim.y * blockIdx.y -(1,1)
// write to local id: local_unrolled_y, local_unrolled_x
// load if inside image
  y_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(0, ch, global_unrolled_y, global_unrolled_x) : 0;
  y1_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(1, ch, global_unrolled_y, global_unrolled_x) : 0;
  y2_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(2, ch, global_unrolled_y, global_unrolled_x) : 0;

  local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
  local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
  global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
  global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;
// load the rest
  if (local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
  {
    y_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]    = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(0, ch, global_unrolled_y, global_unrolled_x) : 0;
    y1_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(1, ch, global_unrolled_y, global_unrolled_x) : 0;
    y2_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv(2, ch, global_unrolled_y, global_unrolled_x) : 0;
  }
  __syncthreads(); // since y has to be loaded completely

  if ( !(w >= bv.W || h >= bv.H) ) // well need to load also beyond borders the block could be
  {
    const float x_   = xv(0, ch, h, w);
    const float b    = bv(0, ch, h, w); // as in the right hand side ..
    float x_loc      = b;
//     y = y + (1/L)*laplace_W*y;
    float yc  = y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1];
    float x05 = yc;
#ifdef __HuberRegularizer__
    {
      float y1_new = y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1]; // its a gradient update .. !
      float y2_new = y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1];
      const float y1_c = y1_new;
      const float y2_c = y2_new;
      if ( h > 0 ) // backward-diff
      {
        const float betaTX = tex2D<float>(betaX, w + 0.5f, h - 0.5f);
        const float betaTY = tex2D<float>(betaY, w + 0.5f, h - 0.5f);
        const float wx_2  = (h > 0 && w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h - 0.5f) : 0; // *L
//const float wxs_2 = (h > 0 && w < bv.W-1) ? tex2D<float>(dx_sq, w + 0.5f, h - 0.5f) : 0;// *L
        const float zij_1 = (w < bv.W - 1) ? ( (y_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] - y_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)  ][threadIdx.x + 1] ) : 0;
        const float zij_2 =                ( (yc                                                                  - y_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) - y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)  ][threadIdx.x + 1] );
        const float hub_weight_x = min(1.f, rsqrtf( (zij_2 * zij_2 * wy_0 + zij_1 * zij_1 * wx_2 ) ) * __HDelta2__ ); // min(1., delta * rsqrt( 0 )) = min (1., inf) =1. -> OK

// flip these two and get different results ..
        const float uij1_2 =                ( y1_c                                                                   - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] );
        const float uij1_1 = (w < bv.W - 1) ? ( y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)]  [threadIdx.x + 2] - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) : 0;
        const float hub_weight_y1 = min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

        const float uij2_1 = (w < bv.W - 1) ? ( y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)]  [threadIdx.x + 2] - y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) : 0;
        const float uij2_2 =                ( y2_c                                                                   - y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] );
        const float hub_weight_y2 = min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

        x05    -=    wy_0  * oneL  * (zij_2 * hub_weight_x);
        y1_new -=   (betaTX * oneL) * (uij1_2 * hub_weight_y1);
        y2_new -=   (betaTY * oneL) * (uij2_2 * hub_weight_y2);
      }
      if ( w > 0 ) // backward, so (nabla x - y) * 1
      {
        const float betaTX = tex2D<float>(betaX, w - 0.5f, h + 0.5f);// +0.5 is the exact location w,h, w-0.5 is pixel w-1.
        const float betaTY = tex2D<float>(betaY, w - 0.5f, h + 0.5f);
        const float wy_2  = (w > 0 && h < bv.H - 1) ? tex2D<float>(dy, w - 0.5f, h + 0.5f) : 0; // stored as wx_0
        const float zij_1 =                ( (yc                                                                  - y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) - y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2) ][threadIdx.x] );
        const float zij_2 = (h < bv.H - 1) ? ( (y_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] - y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) - y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2) ][threadIdx.x] ) : 0;

        const float hub_weight_x =   min(1.f, rsqrtf( (zij_2 * zij_2 * wy_2 + zij_1 * zij_1 * wx_0 ) ) * __HDelta2__ );

        const float uij1_1 =                ( (y1_c                                                                   - y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) );
        const float uij1_2 = (h < bv.H - 1) ? ( (y1_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x]   - y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) ) : 0;

        const float hub_weight_y1 =   min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

        const float uij2_1 =                ( (y2_c                                                                   - y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) );
        const float uij2_2 = (h < bv.H - 1) ? ( (y2_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x]   - y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) ) : 0;
        const float hub_weight_y2 =   min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

        x05    -=    wx_0  * oneL  * (zij_1 * hub_weight_x); // correct order nabla x - y
        y1_new -=   (betaTX * oneL) * (uij1_1 * hub_weight_y1);
        y2_new -=   (betaTY * oneL) * (uij2_1 * hub_weight_y2);
      }

      const float betaTX = tex2D<float>(betaX, w + 0.5f, h + 0.5f);
      const float betaTY = tex2D<float>(betaY, w + 0.5f, h + 0.5f);
      const float zij_1 = (w < bv.W - 1) ? ((yc  - y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2]) + y1_c ) : 0; // flipped order nabla x - y -> -nabla x + y
      const float zij_2 = (h < bv.H - 1) ? ((yc  - y_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1]) + y2_c ) : 0;
      const float hub_weight_x =    min(1.f, rsqrtf( (zij_2 * zij_2 * wy_1 + zij_1 * zij_1 * wx_1) ) * __HDelta2__ );

      const float uij1_1 = (w < bv.W - 1) ? ( (y1_c                                  - y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] ) ) : 0;
      const float uij1_2 = (h < bv.H - 1) ? ( (y1_c                                  - y1_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) ) : 0;

      const float hub_weight_y1 =    min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1) ) * __HDeltaY1__ );

      const float uij2_1 = (w < bv.W - 1) ? ( (y2_c                                  - y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] ) ) : 0;
      const float uij2_2 = (h < bv.H - 1) ? ( (y2_c                                  - y2_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) ) : 0;

      const float hub_weight_y2 =    min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1) ) * __HDeltaY2__ );

      x05    -=  wy_1   * oneL  * (zij_2 * hub_weight_x);
      y1_new -= (betaTX * oneL) * (uij1_2 * hub_weight_y1);
      y2_new -= (betaTY * oneL) * (uij2_2 * hub_weight_y2)  + wy_1 * oneL * (zij_2 * hub_weight_x); // second change to W(Dx-y)

      x05    -=   wx_1   * oneL  * (zij_1 * hub_weight_x);
      y1_new -=  (betaTX * oneL) * (uij1_1 * hub_weight_y1)  + wx_1 * oneL * (zij_1 * hub_weight_x); // second change W(Dx-y) !
      y2_new -=  (betaTY * oneL) * (uij2_1 * hub_weight_y2);

      yv_out(1, ch, h, w) = y1_new + t * (y1_new - xv(1, ch, h, w));
      yv_out(2, ch, h, w) = y2_new + t * (y2_new - xv(2, ch, h, w));
// x^k+1
      xv(1, ch, h, w) = y1_new;
      xv(2, ch, h, w) = y2_new;
    }
#else
    {
      const float wxs_0 = (w > 0 )     ? tex2D<float>(dx_sq, w - 0.5f, h + 0.5f) : 0;// *L
      const float wxs_1 = (w < bv.W - 1) ? tex2D<float>(dx_sq, w + 0.5f, h + 0.5f) : 0;
      const float wys_0 = (h > 0 )     ? tex2D<float>(dy_sq, w + 0.5f, h - 0.5f) : 0;
      const float wys_1 = (h < bv.H - 1) ? tex2D<float>(dy_sq, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.

      float y1_new = y1_prev[threadIdx.y + 1][threadIdx.x + 1]; //its a gradient update .. !
      float y2_new = y2_prev[threadIdx.y + 1][threadIdx.x + 1];
      const float y1_c = y1_new;
      const float y2_c = y2_new;
      if ( h > 0 ) // backward-diff
      {
        x05    -=  (  (yc -  y_prev[threadIdx.y][threadIdx.x + 1]) - y2_prev[threadIdx.y  ][threadIdx.x + 1] ) * wys_0 * wys_0 * oneL;
        y1_new -=  beta * (y1_c - y1_prev[threadIdx.y][threadIdx.x + 1]) * oneL;
        y2_new -=  beta * (y2_c - y2_prev[icthreadIdx.y][threadIdx.x + 1]) * oneL;
      }
      if ( h < bv.H - 1 ) // forward
      {
        const float zij_2 = ((yc  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) + y2_c ) * wys_1 * oneL;
        x05    -=  zij_2 * wys_1;
        y1_new -= beta * ( (y1_c - y1_prev[threadIdx.y + 2][threadIdx.x + 1]) ) * oneL;
        y2_new -= beta * ( (y2_c - y2_prev[threadIdx.y + 2][threadIdx.x + 1]) ) * oneL + zij_2 * wys_1;
      }
      if ( w > 0 ) // backward, so (nabla x - y) * 1
      {
        x05    -=   (  (yc -  y_prev[threadIdx.y + 1][threadIdx.x]) - y1_prev[threadIdx.y + 1][threadIdx.x  ] ) * wxs_0 * wxs_0 * oneL; // correct order nabla x - y
        y1_new -=   beta * (y1_c - y1_prev[threadIdx.y + 1][threadIdx.x]) * oneL;
        y2_new -=   beta * (y2_c - y2_prev[threadIdx.y + 1][threadIdx.x]) * oneL;
      }
      if ( w < bv.W - 1 ) // forward, so (nabla x - y) * -1 = (nabla x - y) * 1
      {
        const float zij_1 = ((yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) + y1_c ) * wxs_1 * oneL; // flipped order nabla x - y -> -nabla x + y
        x05    -=   zij_1 * wxs_1;
        y1_new -=   beta * (y1_c - y1_prev[threadIdx.y + 1][threadIdx.x + 2]) * oneL + zij_1 * wxs_1;
        y2_new -=   beta * (y2_c - y2_prev[threadIdx.y + 1][threadIdx.x + 2]) * oneL;
      }

      yv_out(1, ch, h, w) = y1_new + t * (y1_new - xv(n + 1, ch, h, w));
      yv_out(2, ch, h, w) = y2_new + t * (y2_new - xv(n + 2, ch, h, w));
// x^k+1
      xv(n + 1, ch, h, w) = y1_new;
      xv(n + 2, ch, h, w) = y2_new;
    }
#endif
////////////////////////////////////////////////
#ifdef __Huber1__

    x_loc = (conf * b + x05) / (1.f + conf);
    yc  = x05 - conf; // *L above ..
    x05 = x05 + conf;
    if (yc  > 1.f + b )  x_loc = yc;
    if (x05 < b - 1.f )  x_loc = x05;

#else // not huber cases   

#ifdef __quadratic__
// x = (conf.*b+x05)./(1+conf); derivative is
// df/db    = df/dx * conf/(1+conf)
// df/dx05  = df/dx *  1/(1+conf)
// df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
    x_loc = (conf * b + x05) / (1.f + conf);
#else // L1 case  
//     x1 = y-conf./L;
//     x2 = y+conf./L;
//     b1 = (x1>x3);
//     b2 = (x2<x3);
//     x_= x;
//     x = x1 .* b1 + x2 .* b2 + x3 .* (1-b1-b2);
    yc  = x05 - conf; // *L above ..
    x05 = x05 + conf;
    if (yc > b)  x_loc = yc;
    if (x05 < b) x_loc = x05;
#endif
#endif // huber or not 
//
//     t_=t
//     t=(1+sqrt(1+4*t_^2))/2;
//     y = x + (t_-1)/t * (x-x_); % that is y over-relaxed
    yv_out(0, ch, h, w) = x_loc + t * (x_loc - x_); //
    xv    (0, ch, h, w) = x_loc;
  }
}

//forward kernel 2
__global__ void TGVB2_simple_nCalls_2D_tex_clean_writexk05( 
    cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf, cudaTextureObject_t betaX, cudaTextureObject_t betaY, 
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv, iu::TensorGpu_32f::TensorKernelData yv, 
    iu::TensorGpu_32f::TensorKernelData yv_out, iu::TensorGpu_32f::TensorKernelData xv_k05_out,
    short it, short channels = 1, float t = 1, float oneL = 0.08332)
{
  const short n = 0;
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < xv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < xv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

  __shared__ float y_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float y1_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! x-direction
  __shared__ float y2_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! y-direction

  for (short ch = 0; ch < channels; ch++) // loop here over channels ..
  {
    // onto 2 tiling .. alright ..
    short local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    short local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    short global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    short global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // read local_unrolled_x + blockDim.x * blockIdx.x, local_unrolled_y + blockDim.y * blockIdx.y -(1,1)
    // write to local id: local_unrolled_y, local_unrolled_x
    // load if inside image ..
    if (ch > 0) __syncthreads(); // since y has to be loaded completely ..

    y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n, ch, global_unrolled_y, global_unrolled_x) : 0;
    y1_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n + 1, ch, global_unrolled_y, global_unrolled_x) : 0;
    y2_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n + 2, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;
    // load the rest
    if (local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
    {
      y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n, ch, global_unrolled_y, global_unrolled_x) : 0;
      y1_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n + 1, ch, global_unrolled_y, global_unrolled_x) : 0;
      y2_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n + 2, ch, global_unrolled_y, global_unrolled_x) : 0;
    }
    __syncthreads(); // since y has to be loaded completely

    if ( !(w >= xv.W || h >= xv.H) ) // well need to load also beyond borders the block could be
    {
      const float x_   = xv(n, ch, h, w);
      const float b    = bv(n, ch, h, w); // as in the right hand side ..
      float x_loc      = b;

      float yc = y_prev[threadIdx.y + 1][threadIdx.x + 1];
      float yt = yc;

#ifdef __HuberRegularizer__
      {
        float y1_new = y1_prev[threadIdx.y + 1][threadIdx.x + 1]; // its a gradient update .. !
        float y2_new = y2_prev[threadIdx.y + 1][threadIdx.x + 1];
        const float y1_c = y1_new;
        const float y2_c = y2_new;
        if ( h > 0 ) // backward-diff
        {
          const float betaTX = tex2D<float>(betaX, w + 0.5f, h - 0.5f);
          const float betaTY = tex2D<float>(betaY, w + 0.5f, h - 0.5f);
          const float wx_2 = (h > 0 && w < xv.W - 1) ? tex2D<float>(dx, w + 0.5f, h - 0.5f) : 0; // *L
          const float zij_1 = (w < xv.W - 1) ? ( (y_prev[threadIdx.y][threadIdx.x + 2] - y_prev[threadIdx.y][threadIdx.x + 1] ) - y1_prev[threadIdx.y  ][threadIdx.x + 1] ) : 0;
          const float zij_2 =                ( (yc                                 - y_prev[threadIdx.y][threadIdx.x + 1] ) - y2_prev[threadIdx.y  ][threadIdx.x + 1] );

          const float hub_weight_x =   min(1.f, rsqrtf( (zij_2 * zij_2 * wy_0 + zij_1 * zij_1 * wx_2 ) ) * __HDelta2__ );

          const float uij1_2 =                ( (y1_c                                  - y1_prev[threadIdx.y][threadIdx.x + 1] ) );
          const float uij1_1 = (w < xv.W - 1) ? ( (y1_prev[threadIdx.y]  [threadIdx.x + 2] - y1_prev[threadIdx.y][threadIdx.x + 1] ) ) : 0;
          const float hub_weight_y1 =   min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

          const float uij2_2 =                ( (y2_c                                  - y2_prev[threadIdx.y][threadIdx.x + 1] ) );
          const float uij2_1 = (w < xv.W - 1) ? ( (y2_prev[threadIdx.y]  [threadIdx.x + 2] - y2_prev[threadIdx.y][threadIdx.x + 1] ) ) : 0;
          const float hub_weight_y2 =   min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

          yt     -=   wy_0  * oneL * (zij_2 * hub_weight_x);
          y1_new -=   betaTX  * oneL * (uij1_2 * hub_weight_y1);
          y2_new -=   betaTY  * oneL * (uij2_2 * hub_weight_y2);
        }
        if ( w > 0 ) // backward, so (nabla x - y) * 1
        {
          const float betaTX = tex2D<float>(betaX, w - 0.5f, h + 0.5f);// +0.5 is the exact location w,h, w-0.5 is pixel w-1.
          const float betaTY = tex2D<float>(betaY, w - 0.5f, h + 0.5f);
          const float wy_2  = (w > 0 && h < xv.H - 1) ? tex2D<float>(dy, w - 0.5f, h + 0.5f) : 0;
          const float zij_1 =                ( (yc                                 - y_prev[threadIdx.y + 1][threadIdx.x] ) - y1_prev[threadIdx.y + 1 ][threadIdx.x] );
          const float zij_2 = (h < xv.H - 1) ? ( (y_prev[threadIdx.y + 2][threadIdx.x] - y_prev[threadIdx.y + 1][threadIdx.x] ) - y2_prev[threadIdx.y + 1 ][threadIdx.x] ) : 0;

          const float hub_weight_x =   min(1.f, rsqrtf( (zij_2 * zij_2 * wy_2 + zij_1 * zij_1 * wx_0 ) ) * __HDelta2__ );

          const float uij1_1 =                ( (y1_c                                  - y1_prev[threadIdx.y + 1][threadIdx.x] ) );
          const float uij1_2 = (h < xv.H - 1) ? ( (y1_prev[threadIdx.y + 2][threadIdx.x]   - y1_prev[threadIdx.y + 1][threadIdx.x] ) ) : 0;
          const float hub_weight_y1 =   min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

          const float uij2_1 =                ( (y2_c                                  - y2_prev[threadIdx.y + 1][threadIdx.x] ) );
          const float uij2_2 = (h < xv.H - 1) ? ( (y2_prev[threadIdx.y + 2][threadIdx.x]   - y2_prev[threadIdx.y + 1][threadIdx.x] ) ) : 0;
          const float hub_weight_y2 =   min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

          yt     -=   wx_0 * oneL *   (zij_1 * hub_weight_x); // correct order nabla x - y
          y1_new -=   betaTX * oneL * (uij1_1 * hub_weight_y1);
          y2_new -=   betaTY * oneL * (uij2_1 * hub_weight_y2);
        }

        const float betaTX = tex2D<float>(betaX, w + 0.5f, h + 0.5f);// +0.5 is the exact location w,h, w-0.5 is pixel w-1.
        const float betaTY = tex2D<float>(betaY, w + 0.5f, h + 0.5f);
        const float zij_1 = (w < xv.W - 1) ? ((yc  - y_prev[threadIdx.y + 1][threadIdx.x + 2]) + y1_c ) : 0; // flipped order nabla x - y -> -nabla x + y
        const float zij_2 = (h < xv.H - 1) ? ((yc  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) + y2_c ) : 0;

        const float hub_weight_x =   min(1.f, rsqrtf( (zij_2 * zij_2 * wy_1 + zij_1 * zij_1 * wx_1 ) ) * __HDelta2__ );

        const float uij1_1 = (w < xv.W - 1) ? ( (y1_c                                  - y1_prev[threadIdx.y + 1][threadIdx.x + 2] ) ) : 0;
        const float uij1_2 = (h < xv.H - 1) ? ( (y1_c                                  - y1_prev[threadIdx.y + 2][threadIdx.x + 1] ) ) : 0;
        const float hub_weight_y1 =   min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

        const float uij2_1 = (w < xv.W - 1) ? ( (y2_c                                  - y2_prev[threadIdx.y + 1][threadIdx.x + 2] ) ) : 0;
        const float uij2_2 = (h < xv.H - 1) ? ( (y2_c                                  - y2_prev[threadIdx.y + 2][threadIdx.x + 1] ) ) : 0;
        const float hub_weight_y2 =   min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );


        yt     -= wy_1 * oneL * ( zij_2 * hub_weight_x );
        y1_new -= betaTX * oneL * (uij1_2 * hub_weight_y1);
        y2_new -= betaTY * oneL * (uij2_2 * hub_weight_y2) + wy_1 * oneL * (zij_2 * hub_weight_x) ; //seoncd change to W(Dx-y)

        yt     -= wx_1 * oneL * ( zij_1 * hub_weight_x );
        y1_new -= betaTX * oneL * (uij1_1 * hub_weight_y1) + wx_1 * oneL * (zij_1 * hub_weight_x); // second change W(Dx-y) !
        y2_new -= betaTY * oneL * (uij2_1 * hub_weight_y2);

        yv_out(1, ch, h, w) = y1_new + t * (y1_new - xv(n + 1, ch, h, w));
        yv_out(2, ch, h, w) = y2_new + t * (y2_new - xv(n + 2, ch, h, w));
        // x^k+1
        xv(n + 1, ch, h, w) = y1_new;
        xv(n + 2, ch, h, w) = y2_new;

        xv_k05_out(3 * it + 1, ch, h, w) = y1_new;
        xv_k05_out(3 * it + 2, ch, h, w) = y2_new;
      }

#else
      const float wxs_0 = (w > 0 )     ? tex2D<float>(dx_sq, w - 0.5f, h + 0.5f) : 0;// *L
      const float wxs_1 = (w < xv.W - 1) ? tex2D<float>(dx_sq, w + 0.5f, h + 0.5f) : 0;
      const float wys_0 = (h > 0 )     ? tex2D<float>(dy_sq, w + 0.5f, h - 0.5f) : 0;
      const float wys_1 = (h < xv.H - 1) ? tex2D<float>(dy_sq, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.
      {
        float y1_new = y1_prev[threadIdx.y + 1][threadIdx.x + 1]; //its a gradient update .. !
        float y2_new = y2_prev[threadIdx.y + 1][threadIdx.x + 1];
        const float y1_c = y1_new;
        const float y2_c = y2_new;
        if ( h > 0 ) // backward-diff
        {
          yt     -=   ( (yc  -  y_prev[threadIdx.y][threadIdx.x + 1]) - y2_prev[threadIdx.y  ][threadIdx.x + 1] ) * wys_0 * wys_0 * oneL;
          y1_new -=   beta * (y1_c - y1_prev[threadIdx.y][threadIdx.x + 1]) * oneL;
          y2_new -=   beta * (y2_c - y2_prev[threadIdx.y][threadIdx.x + 1]) * oneL;
        }
        if ( h < xv.H - 1 ) // forward
        {
          const float zij_2 = ((yc  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) + y2_c ) * wys_1 * oneL;
          yt     -= zij_2 * wys_1;
          y1_new -= beta * ( (y1_c - y1_prev[threadIdx.y + 2][threadIdx.x + 1]) ) * oneL;
          y2_new -= beta * ( (y2_c - y2_prev[threadIdx.y + 2][threadIdx.x + 1]) ) * oneL + zij_2 * wys_1;
        }
        if ( w > 0 ) // backward, so (nabla x - y) * 1
        {
          yt     -=   (  (yc -  y_prev[threadIdx.y + 1][threadIdx.x]) - y1_prev[threadIdx.y + 1][threadIdx.x  ] ) * wxs_0 * wxs_0 * oneL; // correct order nabla x - y
          y1_new -=   beta * (y1_c - y1_prev[threadIdx.y + 1][threadIdx.x]) * oneL;
          y2_new -=   beta * (y2_c - y2_prev[threadIdx.y + 1][threadIdx.x]) * oneL;
        }
        if ( w < xv.W - 1 ) // forward, so (nabla x - y) * -1 = (nabla x - y) * 1
        {
          const float zij_1 = ((yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) + y1_c ) * wxs_1 * oneL; // flipped order nabla x - y -> -nabla x + y
          yt     -=   zij_1 * wxs_1;
          y1_new -=   beta * (y1_c - y1_prev[threadIdx.y + 1][threadIdx.x + 2]) * oneL + zij_1 * wxs_1;
          y2_new -=   beta * (y2_c - y2_prev[threadIdx.y + 1][threadIdx.x + 2]) * oneL; // - zij_1;
        }
        yv_out(1, ch, h, w) = y1_new + t * (y1_new - xv(n + 1, ch, h, w));
        yv_out(2, ch, h, w) = y2_new + t * (y2_new - xv(n + 2, ch, h, w));
        // x^k+1
        xv(n + 1, ch, h, w) = y1_new; // not needed actualy ..
        xv(n + 2, ch, h, w) = y2_new;
        xv_k05_out(3 * it + 1, ch, h, w) = y1_new;
        xv_k05_out(3 * it + 2, ch, h, w) = y2_new;
      }
#endif

      xv_k05_out(3 * it,  ch, h, w) = yt;

#ifdef __Huber1__

      x_loc = (conf * b + yt) / (1.f + conf);
      yc = yt - conf; // *L above ..
      yt = yt + conf;
      if (yc  > 1.f + b )  x_loc = yc;
      if (yt  < b - 1.f )  x_loc = yt;

#else // not huber cases   

#ifdef __quadratic__
      // x = (conf.*b+x05)./(1+conf); derivative is
      // df/db    = df/dx * conf/(1+conf)
      // df/dx05  = df/dx *  1/(1+conf)
      // df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
      x_loc = (conf * b + yt) / (1.f + conf);
#else
      //     x1 = y-conf./L;
      //     x2 = y+conf./L;
      //     b1 = (x1>x3);
      //     b2 = (x2<x3);
      //     x_= x;
      //     x = x1 .* b1 + x2 .* b2 + x3 .* (1-b1-b2);
      yc = yt - conf; // *L above ..
      yt = yt + conf;
      if (yc > b) x_loc = yc;
      if (yt < b) x_loc = yt;
#endif
#endif
      //
      //     t_=t
      //     t=(1+sqrt(1+4*t_^2))/2;
      //     y = x + (t_-1)/t * (x-x_); % that is y over-relaxed
      yv_out(0, ch, h, w) = x_loc + t * (x_loc - x_); //
      xv(n, ch, h, w) = x_loc;
    }
  }
}

template<int channels>
__global__ void TGVB2_simple_nCalls_2D_tex_clean_writexk05_ch( 
    cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf, 
    cudaTextureObject_t betaX, cudaTextureObject_t betaY,// iu::TensorGpu_32f::TensorKernelData cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv,
    iu::TensorGpu_32f::TensorKernelData yv, iu::TensorGpu_32f::TensorKernelData yv_out,
    iu::TensorGpu_32f::TensorKernelData xv_k05_out, short it, float t = 1, float oneL = 0.08332)
{
  short w  = blockIdx.x * blockDim.x + threadIdx.x;
  short h  = blockIdx.y * blockDim.y + threadIdx.y;
  short ch = blockIdx.z * blockDim.z + threadIdx.z;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < xv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < xv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.

  __shared__ float y_prev [channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
  __shared__ float y1_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! x-direction
  __shared__ float y2_prev[channels * (COMMON_BLOCK_SIZE_2D_Y + 2)][COMMON_BLOCK_SIZE_2D_X + 2]; // those are the auxilliary vectors! y-direction

// loop here over channels ..
// onto 2 tiling .. alright ..
  short local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
  short local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
  short global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
  short global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

// read local_unrolled_x + blockDim.x * blockIdx.x, local_unrolled_y + blockDim.y * blockIdx.y -(1,1)
// write to local id: local_unrolled_y, local_unrolled_x
// load if inside image
  y_prev[ local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(0, ch, global_unrolled_y, global_unrolled_x) : 0;
  y1_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(1, ch, global_unrolled_y, global_unrolled_x) : 0;
  y2_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(2, ch, global_unrolled_y, global_unrolled_x) : 0;

  local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
  local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
  global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
  global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

  if (local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
  {
    y_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(0, ch, global_unrolled_y, global_unrolled_x) : 0;
    y1_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(1, ch, global_unrolled_y, global_unrolled_x) : 0;
    y2_prev[local_unrolled_y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(2, ch, global_unrolled_y, global_unrolled_x) : 0;
  }
  __syncthreads(); // since y has to be loaded completely

  if ( !(w >= xv.W || h >= xv.H) ) // well need to load also beyond borders the block could be
  {
    const float x_   = xv(0, ch, h, w);
    const float b    = bv(0, ch, h, w); // as in the right hand side ..
    float x_loc      = b;

    float yc = y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1];
    float yt = yc;

#ifdef __HuberRegularizer__
    {
      float y1_new = y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1]; // its a gradient update .. !
      float y2_new = y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1];
      const float y1_c = y1_new;
      const float y2_c = y2_new;
      if ( h > 0 ) // backward-diff
      {
        const float betaTX = tex2D<float>(betaX, w + 0.5f, h - 0.5f);
        const float betaTY = tex2D<float>(betaY, w + 0.5f, h - 0.5f);
        const float wx_2 = (h > 0 && w < xv.W - 1) ? tex2D<float>(dx, w + 0.5f, h - 0.5f) : 0; // *L
        const float zij_1 = (w < xv.W - 1) ? ( (y_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] - y_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)  ][threadIdx.x + 1] ) : 0;
        const float zij_2 =                ( (yc                                                                  - y_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) - y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)  ][threadIdx.x + 1] );

        const float hub_weight_x =   min(1.f, rsqrtf( (zij_2 * zij_2 * wy_0 + zij_1 * zij_1 * wx_2 ) ) * __HDelta2__ );

        const float uij1_2 =                ( (y1_c                                                                   - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) );
        const float uij1_1 = (w < xv.W - 1) ? ( (y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)]  [threadIdx.x + 2] - y1_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) ) : 0;
        const float hub_weight_y1 =   min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

        const float uij2_2 =                ( (y2_c                                                                   - y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) );
        const float uij2_1 = (w < xv.W - 1) ? ( (y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)]  [threadIdx.x + 2] - y2_prev[threadIdx.y + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) ) : 0;
        const float hub_weight_y2 =   min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

        yt     -=   wy_0  * oneL * (zij_2 * hub_weight_x);
        y1_new -=   betaTX  * oneL * (uij1_2 * hub_weight_y1); //(y1_c - y1_prev[threadIdx.y][threadIdx.x+1]) * wys_0
        y2_new -=   betaTY  * oneL * (uij2_2 * hub_weight_y2); //(y2_c - y2_prev[threadIdx.y][threadIdx.x+1]) * wys_0
      }
      if ( w > 0 ) // backward, so (nabla x - y) * 1
      {
        const float betaTX = tex2D<float>(betaX, w - 0.5f, h + 0.5f);
        const float betaTY = tex2D<float>(betaY, w - 0.5f, h + 0.5f);
        const float wy_2  = (w > 0 && h < xv.H - 1) ? tex2D<float>(dy, w - 0.5f, h + 0.5f) : 0;
        const float zij_1 =                ( (yc                                                                  - y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) - y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2) ][threadIdx.x] );
        const float zij_2 = (h < xv.H - 1) ? ( (y_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] - y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) - y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2) ][threadIdx.x] ) : 0;

        const float hub_weight_x =   min(1.f, rsqrtf( (zij_2 * zij_2 * wy_2 + zij_1 * zij_1 * wx_0 ) ) * __HDelta2__ );

        const float uij1_1 =                ( (y1_c                                                                  - y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) );
        const float uij1_2 = (h < xv.H - 1) ? ( (y1_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x]  - y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) ) : 0;
        const float hub_weight_y1 =   min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

        const float uij2_1 =                ( (y2_c                                                                 - y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) );
        const float uij2_2 = (h < xv.H - 1) ? ( (y2_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] - y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x] ) ) : 0;
        const float hub_weight_y2 =   min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

        yt     -=   wx_0 * oneL *   (zij_1 * hub_weight_x); // correct order nabla x - y
        y1_new -=   betaTX * oneL * (uij1_1 * hub_weight_y1); //(y1_c - y1_prev[threadIdx.y+1][threadIdx.x]) * wxs_0
        y2_new -=   betaTY * oneL * (uij2_1 * hub_weight_y2); //(y2_c - y2_prev[threadIdx.y+1][threadIdx.x]) * wxs_0
      }

      const float betaTX = tex2D<float>(betaX, w + 0.5f, h + 0.5f);
      const float betaTY = tex2D<float>(betaY, w + 0.5f, h + 0.5f);
      const float zij_1 = (w < xv.W - 1) ? ((yc  - y_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2]) + y1_c ) : 0; // flipped order nabla x - y -> -nabla x + y
      const float zij_2 = (h < xv.H - 1) ? ((yc  - y_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1]) + y2_c ) : 0;

      const float hub_weight_x =   min(1.f, rsqrtf( (zij_2 * zij_2 * wy_1 + zij_1 * zij_1 * wx_1 ) ) * __HDelta2__ );

      const float uij1_1 = (w < xv.W - 1) ? ( (y1_c                                  - y1_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] ) ) : 0;
      const float uij1_2 = (h < xv.H - 1) ? ( (y1_c                                  - y1_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) ) : 0;
      const float hub_weight_y1 =   min(1.f, rsqrtf( betaTX * (uij1_2 * uij1_2 + uij1_1 * uij1_1 ) ) * __HDeltaY1__ );

      const float uij2_1 = (w < xv.W - 1) ? ( (y2_c                                  - y2_prev[threadIdx.y + 1 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 2] ) ) : 0;
      const float uij2_2 = (h < xv.H - 1) ? ( (y2_c                                  - y2_prev[threadIdx.y + 2 + ch * (COMMON_BLOCK_SIZE_2D_Y + 2)][threadIdx.x + 1] ) ) : 0;
      const float hub_weight_y2 =   min(1.f, rsqrtf( betaTY * (uij2_2 * uij2_2 + uij2_1 * uij2_1 ) ) * __HDeltaY2__ );

      yt     -= wy_1 * oneL * ( zij_2 * hub_weight_x );
      y1_new -= betaTX * oneL * (uij1_2 * hub_weight_y1); //(y1_c - y1_prev[threadIdx.y+2][threadIdx.x+1]) * wys_1
      y2_new -= betaTY * oneL * (uij2_2 * hub_weight_y2) + wy_1 * oneL * (zij_2 * hub_weight_x) ; //seoncd change to W(Dx-y)

      yt     -= wx_1 * oneL * ( zij_1 * hub_weight_x );
      y1_new -= betaTX * oneL * (uij1_1 * hub_weight_y1) + wx_1 * oneL * (zij_1 * hub_weight_x); // second change W(Dx-y) !
      y2_new -= betaTY * oneL * (uij2_1 * hub_weight_y2); //(y2_c - y2_prev[threadIdx.y+1][threadIdx.x+2]) * wxs_1

      yv_out(1, ch, h, w) = y1_new + t * (y1_new - xv(1, ch, h, w));
      yv_out(2, ch, h, w) = y2_new + t * (y2_new - xv(2, ch, h, w));
// x^k+1
      xv(1, ch, h, w) = y1_new;
      xv(2, ch, h, w) = y2_new;

      xv_k05_out(3 * it + 1, ch, h, w) = y1_new;
      xv_k05_out(3 * it + 2, ch, h, w) = y2_new;
    }

#else
    const float wxs_0 = (w > 0 )     ? tex2D<float>(dx_sq, w - 0.5f, h + 0.5f) : 0;// *L
    const float wxs_1 = (w < xv.W - 1) ? tex2D<float>(dx_sq, w + 0.5f, h + 0.5f) : 0;
    const float wys_0 = (h > 0 )     ? tex2D<float>(dy_sq, w + 0.5f, h - 0.5f) : 0;
    const float wys_1 = (h < xv.H - 1) ? tex2D<float>(dy_sq, w + 0.5f, h + 0.5f) : 0; // +0.5 is the exact location w,h, w-0.5 is pixel w-1.

    {
      float y1_new = y1_prev[threadIdx.y + 1][threadIdx.x + 1]; //its a gradient update
      float y2_new = y2_prev[threadIdx.y + 1][threadIdx.x + 1];
      const float y1_c = y1_new;
      const float y2_c = y2_new;
      if ( h > 0 ) // backward-diff
      {
        yt     -=   ( (yc  -  y_prev[threadIdx.y][threadIdx.x + 1]) - y2_prev[threadIdx.y  ][threadIdx.x + 1] ) * wys_0 * wys_0 * oneL;
        y1_new -=   beta * (y1_c - y1_prev[threadIdx.y][threadIdx.x + 1]) * oneL;
        y2_new -=   beta * (y2_c - y2_prev[threadIdx.y][threadIdx.x + 1]) * oneL;
      }
      if ( h < xv.H - 1 ) // forward
      {
        const float zij_2 = ((yc  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) + y2_c ) * wys_1 * oneL;
        yt     -= zij_2 * wys_1;
        y1_new -= beta * ( (y1_c - y1_prev[threadIdx.y + 2][threadIdx.x + 1]) ) * oneL;
        y2_new -= beta * ( (y2_c - y2_prev[threadIdx.y + 2][threadIdx.x + 1]) ) * oneL + zij_2 * wys_1;
      }
      if ( w > 0 ) // backward, so (nabla x - y) * 1
      {
        yt     -=   (  (yc -  y_prev[threadIdx.y + 1][threadIdx.x]) - y1_prev[threadIdx.y + 1][threadIdx.x  ] ) * wxs_0 * wxs_0 * oneL; // correct order nabla x - y
        y1_new -=   beta * (y1_c - y1_prev[threadIdx.y + 1][threadIdx.x]) * oneL;
        y2_new -=   beta * (y2_c - y2_prev[threadIdx.y + 1][threadIdx.x]) * oneL;
      }
      if ( w < xv.W - 1 ) // forward, so (nabla x - y) * -1 = (nabla x - y) * 1
      {
        const float zij_1 = ((yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) + y1_c ) * wxs_1 * oneL; // flipped order nabla x - y -> -nabla x + y
        yt     -=   zij_1 * wxs_1;
        y1_new -=   beta * (y1_c - y1_prev[threadIdx.y + 1][threadIdx.x + 2]) * oneL + zij_1 * wxs_1;
        y2_new -=   beta * (y2_c - y2_prev[threadIdx.y + 1][threadIdx.x + 2]) * oneL;
      }
// extrapolated ..
      yv_out(1, ch, h, w) = y1_new + t * (y1_new - xv(n + 1, ch, h, w));
      yv_out(2, ch, h, w) = y2_new + t * (y2_new - xv(n + 2, ch, h, w));
// x^k+1
      xv(1, ch, h, w) = y1_new; // not needed actualy ..
      xv(2, ch, h, w) = y2_new;
      xv_k05_out(3 * it + 1, ch, h, w) = y1_new;
      xv_k05_out(3 * it + 2, ch, h, w) = y2_new;
    }
#endif
    xv_k05_out(3 * it,  ch, h, w) = yt;

#ifdef __Huber1__

    x_loc = (conf * b + yt) / (1.f + conf);
    yc = yt - conf;
    yt = yt + conf;
    if (yc  > 1.f + b )  x_loc = yc;
    if (yt  < b - 1.f )  x_loc = yt;

#else // not huber cases   

#ifdef __quadratic__
// x = (conf.*b+x05)./(1+conf); derivative is
// df/db    = df/dx * conf/(1+conf)
// df/dx05  = df/dx *  1/(1+conf)
// df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
    x_loc = (conf * b + yt) / (1.f + conf);
#else
//     x1 = y-conf./L;
//     x2 = y+conf./L;
//     b1 = (x1>x3);
//     b2 = (x2<x3);
//     x_= x;
//     x = x1 .* b1 + x2 .* b2 + x3 .* (1-b1-b2);
    yc = yt - conf; // *L above ..
    yt = yt + conf;
    if (yc > b) x_loc = yc;
    if (yt < b) x_loc = yt;
#endif

#endif
//
//     t_=t
//     t=(1+sqrt(1+4*t_^2))/2;
//     y = x + (t_-1)/t * (x-x_); % that is y over-relaxed
    yv_out(0, ch, h, w) = x_loc + t * (x_loc - x_); //
    xv(0, ch, h, w)     = x_loc;
  }
}


// dx,dy: tensors/edge weights in x and y direction. d_c: confidence, d_b rhs, eg. u_hat;
// d_i: initial solution used at iteration 0. simplest is d_b or solution from lower resolution
// d_out: output. ic: channels(# of rhs/channels of b), iw: width ih: height of inputs
// its: iterations to run.
// id: we can run multiple inpainters, eg. for different resolutions. To identify the buffers, etc.
// we need an id, that is specified by the user. Its a number between 0 and 9 so far..
int FistaTGVHuberB2_id::forward( float *d_x, float *d_y, float *d_c, float *d_b, float *d_i, float *d_out,
                                 int ic, int ih, int iw, int its, int id )
{
  const int in = 1;     // d_in has 3 channels: primary + 2x auxxiliary variables .. i use 3*in
  const int c_xyc = 2;  // edge weights have 2 channel: primary & auxxiliary variables
  const int c_dims = 1;// dimension of confidence -- note this could also be 2 == different for flowx and flowy
  const int ingrad_c = ic;

  if (static_cast<size_t> (id) >= FistaTGVHuberB2_id::id.size())
  {
    std::cerr << "Cannot use an id that is negative or larger/equal than the maximal one:" << id << "/" << FistaTGVHuberB2_id::id.size() << "\n";
    return 1;
  }

  if (FistaTGVHuberB2_id::id[id].run == 0)
    std::cout << "FistaTGVHuberB2_id::forward \n Sizes: n:" << in << " c:" << ic << " h:" << ih << " w:" << iw << " cd:" << c_dims << " xyc:" << c_xyc << " its  " << its << "\n";

  iu::TensorGpu_32f d_db(  d_b,    in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW); // from here extract in loop rhs and copy to output !
  iu::TensorGpu_32f d_dout(d_out, 3 * in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_din( d_i,  3 * in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);

  iu::ImageGpu_32f_C1 tex_dx(d_x, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_dy(d_y, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_cf(d_c, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_betaX(d_x + iw * ih, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_betaY(d_y + iw * ih, iw, ih, sizeof(float) * iw, false);

  // those are larger ! separate LOGICALLY !!! in 1st dimension
  iu::TensorGpu_32f y0_temp( 3, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for y ping
  iu::TensorGpu_32f y1_temp( 3, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for y pong
  iu::TensorGpu_32f  x_temp( 3, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x

  dim3 dimBlock(COMMON_BLOCK_SIZE_2D_X, COMMON_BLOCK_SIZE_2D_Y);
  dim3 dimGrid( std::ceil( ( d_db.width()     ) / static_cast<float>(dimBlock.x)) ,
                std::ceil( ( d_db.height()    ) / static_cast<float>(dimBlock.y)), 2 );// special case .. hence y is 'channeled'
  dim3 dimGrid_xy( std::ceil( ( d_db.width()  ) / static_cast<float>(dimBlock.x)) ,
                   std::ceil( ( d_db.height() ) / static_cast<float>(dimBlock.y)) );

  // x_temp is later set to 0, dummy here to compute local Lipshitz that becomes th eglobal value
  FistaInpaintB2_findLipshitz <<< dimGrid_xy, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), x_temp );
  // reduction after copy ..
  float oneL = 1.;
  oneL =  float( (std::max( 12., 8.* beta ) + 0.000001) );
  {
    std::vector<float> tempv( ih * iw, 0);
    cudaMemcpy(tempv.data(), x_temp.data(), tempv.size()*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < iw * ih; i++) oneL = std::max(oneL, tempv[i]);
  }
  if (FistaTGVHuberB2_id::id[id].run % 50 == 0)
    std::cout << "Lipshitz found: " << oneL << "\n";
  oneL = 1. / (oneL + 0.01f);
  using namespace thrust::placeholders;

  double L   =  oneL; oneL = 1.0;
  thrust::transform(tex_dx.begin(), tex_dx.end(), tex_dx.begin(), L * _1);
  thrust::transform(tex_dy.begin(), tex_dy.end(), tex_dy.begin(), L * _1);
  thrust::transform(tex_betaX.begin(), tex_betaX.end(), tex_betaX.begin(), L * _1);
  thrust::transform(tex_betaY.begin(), tex_betaY.end(), tex_betaY.begin(), L * _1);

  thrust::fill( x_temp.begin(),  x_temp.end(), 0.0f); cudaMemcpy(   x_temp.data(), d_din.data(), 3 * in * ic * ih * iw * sizeof(float),  cudaMemcpyDeviceToDevice); // init value .. only for x ..
  thrust::fill(y0_temp.begin(), y0_temp.end(), 0.0f); cudaMemcpy(  y0_temp.data(), d_din.data(), 3 * in * ic * ih * iw * sizeof(float),  cudaMemcpyDeviceToDevice);
  thrust::fill(y1_temp.begin(), y1_temp.end(), 0.0f); cudaMemcpy(  y1_temp.data(), d_din.data(), 3 * in * ic * ih * iw * sizeof(float),  cudaMemcpyDeviceToDevice);

  FistaTGVHuberB2_id::id[id].iterations.resize( std::floor( sqrt( its ) ), 0 );
  FistaTGVHuberB2_id::id[id].maxInterIts = 0;
#ifdef _DEBUG_FW_ENCODE
  std::cout << FistaTGVHuberB2_id::id[id].iterations.size() << "intermediate storage points " << its << " " << sqrt( its ) <<  " "  << std::ceil( sqrt( its ) ) << "\n";
#endif
  for (int i = 0; i < FistaTGVHuberB2_id::id[id].iterations.size(); i++)
  {
    FistaTGVHuberB2_id::id[id].iterations[i] = std::floor(i * sqrt(its) * 0.5) * 2; // i miss the irregular ones ..
#ifdef _DEBUG_FW_ENCODE
    std::cout << i << "  Storing at " << FistaTGVHuberB2_id::id[id].iterations[i] << "\n";
#endif
    if (i > 0)
    {
#ifdef _DEBUG_FW_ENCODE
      std::cout << i << " Diff: " << FistaTGVHuberB2_id::id[id].iterations[i] - FistaTGVHuberB2_id::id[id].iterations[i - 1] << "\n";
#endif
      FistaTGVHuberB2_id::id[id].maxInterIts = std::max(FistaTGVHuberB2_id::id[id].maxInterIts, FistaTGVHuberB2_id::id[id].iterations[i] - FistaTGVHuberB2_id::id[id].iterations[i - 1] );
    }

    if (FistaTGVHuberB2_id::id[id].intermediate_y[ i ])
    {
      free( FistaTGVHuberB2_id::id[id].intermediate_y[ i ] );
      FistaTGVHuberB2_id::id[id].intermediate_y[ i ] = NULL;
    }
    FistaTGVHuberB2_id::id[id].intermediate_y[ i ] = (float*) calloc( in * 3 * ic * ih * iw, sizeof(float) );
    if (FistaTGVHuberB2_id::id[id].intermediate_x[ i ])
    {
      free( FistaTGVHuberB2_id::id[id].intermediate_x[ i ] );
      FistaTGVHuberB2_id::id[id].intermediate_x[ i ] = NULL;
    }
    FistaTGVHuberB2_id::id[id].intermediate_x[ i ] = (float*) calloc( in * 3 * ic * ih * iw, sizeof(float) );
  }
  FistaTGVHuberB2_id::id[id].maxInterIts = std::max(FistaTGVHuberB2_id::id[id].maxInterIts, its - FistaTGVHuberB2_id::id[id].iterations[FistaTGVHuberB2_id::id[id].iterations.size() - 1] );
#ifdef _DEBUG_FW_ENCODE
  std::cout << FistaTGVHuberB2_id::id[id].maxInterIts << " maximal its in between \n";
  std::cout << "End (not) Storing at " << (its / 2) * 2 << "\n";
  std::cout << FistaTGVHuberB2_id::id[id].iterations.size() << " Diff: " << (its / 2) * 2 - FistaTGVHuberB2_id::id[id].iterations[FistaTGVHuberB2_id::id[id].iterations.size() - 1] << "\n";
#endif

  FistaTGVHuberB2_id::id[id].stepSizes.resize( its, 0);
  double t(1), t_(1);
  for (int i = 0; i < FistaTGVHuberB2_id::id[id].stepSizes.size(); i++)
  {
    t_ = t;
    t  = (1. + sqrt(1. + 4.*t * t)) / 2.;
    FistaTGVHuberB2_id::id[id].stepSizes[i] = std::max( 0.0, (t_ - 1.) / t);
  }

  int start_it = 0;
  {
    float step(1);

    for ( int it = 0; it < its / 2; it++ )
    {
      // store the y vector
      if ( start_it < FistaTGVHuberB2_id::id[id].iterations.size() && FistaTGVHuberB2_id::id[id].iterations[start_it] == 2 * it && FistaTGVHuberB2_id::id[id].intermediate_y[ start_it ] != NULL )
      {
        cudaMemcpy( FistaTGVHuberB2_id::id[id].intermediate_y[ start_it ], y0_temp.data(), 3 * in * ic * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
        cudaMemcpy( FistaTGVHuberB2_id::id[id].intermediate_x[ start_it ],  x_temp.data(), 3 * in * ic * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );

#ifdef _DEBUG_FW_ENCODE
        std::cout << start_it << "copy Y vector " << 2 * it << "  " << FistaTGVHuberB2_id::id[id].intermediate_y[ start_it ][0] << "  " << FistaTGVHuberB2_id::id[id].intermediate_y[ start_it ][1] << "\n";
        std::cout << start_it << "copy x vector " << 2 * it << "  " << FistaTGVHuberB2_id::id[id].intermediate_x[ start_it ][0] << "  " << FistaTGVHuberB2_id::id[id].intermediate_x[ start_it ][1] << "\n";
#endif
        start_it++;
      }

      step = FistaTGVHuberB2_id::id[id].stepSizes[2 * it];
      if (ic == 2)
        TGVB2_simple_nCalls_2D_tex_clean_ch<2> <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_db, x_temp, y0_temp, y1_temp, step, oneL ); // samples
      else
        TGVB2_simple_nCalls_2D_tex_clean <<< dimGrid_xy, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_db, x_temp, y0_temp, y1_temp, ic, step, oneL ); // samples
      step = FistaTGVHuberB2_id::id[id].stepSizes[2 * it + 1];
      if (ic == 2)
        TGVB2_simple_nCalls_2D_tex_clean_ch<2> <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_db, x_temp, y1_temp, y0_temp, step, oneL ); // samples
      else
        TGVB2_simple_nCalls_2D_tex_clean <<< dimGrid_xy, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_db, x_temp, y1_temp, y0_temp, ic, step, oneL ); // samples
    }
    cudaMemcpy( d_dout.data(), x_temp.data(), 3 * in * ic * ih * iw * sizeof(float),  cudaMemcpyDeviceToDevice );
  }
  return 0;
}


int FistaTGVHuberB2_id::backward( float *d_x, float *d_y, float *d_c, float *d_b,
                                  float *d_inGrad, float *d_outGradX, float *d_outGradY, float *d_outGradC, float *d_outGradB, float *d_outGradI,
                                  int ic, int ih, int iw, int its, int id )
{
  if (static_cast<size_t> (id) >= FistaTGVHuberB2_id::id.size())
  {
    std::cerr << "Cannot use an id that is negative or larger/equal than the maximal one:" << id << "/" << FistaTGVHuberB2_id::id.size() << "\n";
    return 1;
  }
  const int in = 1;     // d_inGrad has 3 channels primary + 2x auxxiliary variables
  const int c_xyc = 2;  // edge weights have 2 channel: primary & auxxiliary variables
  const int c_dims = 1; // dimension of confidence channels .. here just 1.
  const int ingrad_c = ic;

  int lastIt = its;
  if (FistaTGVHuberB2_id::id[id].run == 0)
    std::cout << "FistaTGVHuberB2_id::backward \n Sizes: n:" << in << " c:" << ic << " h:" << ih << " w:" << iw << " cd:" << c_dims << " xyc:" << c_xyc << " its  " << its << "\n";

  float oneL = 1.;  // reduction after copy ..
  oneL =  float( (std::max( 12., 8.*beta ) + 0.000001) );

  iu::ImageGpu_32f_C1 tex_dx(d_x, iw, ih, sizeof(float) * iw, false); // bool ext_data_pointer=true
  iu::ImageGpu_32f_C1 tex_dy(d_y, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_cf(d_c, iw, ih, sizeof(float) * iw, false);

  iu::ImageGpu_32f_C1 tex_betaX(d_x + iw * ih, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_betaY(d_y + iw * ih, iw, ih, sizeof(float) * iw, false);

  using namespace thrust::placeholders;
  iu::TensorGpu_32f d_gi(d_outGradI, 3 * in, ingrad_c, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_db(d_b,         in,  ingrad_c, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gx(d_outGradX,  in,  c_xyc,    ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gy(d_outGradY,  in,  c_xyc,    ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gc(d_outGradC,  in,  c_dims,   ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gb(d_outGradB,  in,  ingrad_c, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gin(d_inGrad,  3 * in, ingrad_c, ih, iw, true, iu::TensorGpu_32f::NCHW);

  dim3 dimBlock(COMMON_BLOCK_SIZE_2D_X, COMMON_BLOCK_SIZE_2D_Y);
  dim3 dimGrid_xy( std::ceil( ( d_db.width()  ) / static_cast<float>(dimBlock.x)) ,
                   std::ceil( ( d_db.height() ) / static_cast<float>(dimBlock.y)) );
  dim3 dimGrid( std::ceil( ( d_db.width()     ) / static_cast<float>(dimBlock.x)) ,
                std::ceil( ( d_db.height()    ) / static_cast<float>(dimBlock.y)) );
  dim3 dimGrid_ch( std::ceil( ( d_db.width()  ) / static_cast<float>(dimBlock.x)) ,
                   std::ceil( ( d_db.height() ) / static_cast<float>(dimBlock.y)), 2 );// special case .. hence y is 'channeled'

  // temporary abuse of d_gy to get lipshitz constant.
  FistaInpaintB2_findLipshitz <<< dimGrid_xy, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_gy );
  {
    std::vector<float> tempv( ih * iw, 0);
    cudaMemcpy(tempv.data(), d_gy.data(), tempv.size()*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < iw * ih; i++) oneL = std::max(oneL, tempv[i]);
  }
  oneL = 1. / (oneL + 0.01f);
  double L   =  oneL; oneL = 1.0;
  thrust::transform(tex_dx.begin(), tex_dx.end(), tex_dx.begin(), L * _1);
  thrust::transform(tex_dy.begin(), tex_dy.end(), tex_dy.begin(), L * _1);
  thrust::transform(tex_betaX.begin(), tex_betaX.end(), tex_betaX.begin(), L * _1);
  thrust::transform(tex_betaY.begin(), tex_betaY.end(), tex_betaY.begin(), L * _1);

  thrust::fill(d_gi.begin(), d_gi.end(), 0.0f);
  thrust::fill(d_gx.begin(), d_gx.end(), 0.0f);
  thrust::fill(d_gy.begin(), d_gy.end(), 0.0f);
  thrust::fill(d_gc.begin(), d_gc.end(), 0.0f);
  thrust::fill(d_gb.begin(), d_gb.end(), 0.0f);

#ifdef _NAN_Input_Check_
  std::vector<float> tempv( 3 * ingrad_c * ih * iw, 0);
  cudaMemcpy( tempv.data(), d_gin.data(), tempv.size()*sizeof(float),  cudaMemcpyDeviceToHost);
  int is_nan = 0; int nnz = 0;
  double maxinG = 0;
  for (int i = 0; i < tempv.size(); i++)
  {
    if (!is_valid(tempv[i]) ) { is_nan++; tempv[i] = 0;}
    if (tempv[i] != 0 ) nnz++;

    maxinG = std::max( maxinG, (double)std::abs(tempv[i]) );
  }
  if ( is_nan != 0 || nnz == 0 )
  {
    std::cerr << "Input gradient to Fista is inf or nan " <<  is_nan  << " times or " << nnz  << " non-zeros: skip !\n";
    cudaMemcpy(d_gin.data(), tempv.data(), tempv.size()*sizeof(float),  cudaMemcpyHostToDevice);
    return 0;
  }

#ifdef _pngWriting_
  if (id == 2)
  {
    pngwriter png4(iw, ih, 0, "dIn.png");
    for (int i = 0; i < ih * iw; i++)
      png4.plot( i % iw, (ih - 1) - i / iw, std::max(0.f, std::min( -tempv[i], 1.f) ), std::max(0.f, std::min( tempv[i], 1.f) ), std::max(0.f, std::min( tempv[i + ih * iw], 1.f) ) );
    png4.close();
  }
#endif

#endif

  iu::TensorGpu_32f y0_temp( 3, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x
  iu::TensorGpu_32f y1_temp( 3, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x

  iu::TensorGpu_32f d_dout(         3, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // will hold x variable
  iu::TensorGpu_32f d_gxk05_0_temp( 3, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for dx/dx^k+0.5 .. need ping pong
  iu::TensorGpu_32f d_gxk05_1_temp( 3, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for dx/dx^k+0.5 .. need ping pong
  iu::TensorGpu_32f d_gxk1_temp(    3, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for dx/dx^k  ..
  iu::TensorGpu_32f d_gxk0_temp(    3, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for dx/dx^k-1  .. yes i need both of them ..

  thrust::fill(d_gxk1_temp.begin(), d_gxk1_temp.end(), 0.0f);
  thrust::fill(d_gxk0_temp.begin(), d_gxk0_temp.end(), 0.0f);

  cudaMemcpy( d_gxk0_temp.data(), d_gin.data(), 3 * in * ingrad_c * ih * iw * sizeof(float), cudaMemcpyDeviceToDevice );

  thrust::fill(d_gxk05_0_temp.begin(), d_gxk05_0_temp.end(), 0.0f);
  thrust::fill(d_gxk05_1_temp.begin(), d_gxk05_1_temp.end(), 0.0f);

  iu::TensorGpu_32f d_yk_temp( 3, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for yk neede to get weights for huber on regularizer
  thrust::fill(d_yk_temp.begin(), d_yk_temp.end(), 0.0f);

  // backwards works like this :
  // take last y, for how many iterations: its-FistaTGVHuberB2_id::id[id].iterations[]
  // generate memory for x storage: how many: lastIt-nextLastIt
  iu::TensorGpu_32f d_xk05( 3 * FistaTGVHuberB2_id::id[id].maxInterIts, ingrad_c, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x+0.5

  std::vector<double> gc_storage( c_dims * ih * iw, 0 );
  std::vector<double> gx_storage( c_xyc  * ih * iw, 0 );
  std::vector<double> gy_storage( c_xyc  * ih * iw, 0 );// changed
  std::vector<double> gb_storage( ingrad_c * ih * iw, 0 );

  float nIts(1.); double scaleFactor(1.);
  for ( int outer = FistaTGVHuberB2_id::id[id].iterations.size() - 1; outer >= 0; outer--)
  {
    int nextLastIt = FistaTGVHuberB2_id::id[id].iterations[ outer ];
    // init y/x to kickstart process:
    cudaMemcpy( y0_temp.data(), FistaTGVHuberB2_id::id[id].intermediate_y[ outer ], 3 * in * ingrad_c * ih * iw * sizeof(float), cudaMemcpyHostToDevice ); // write on other 1st ..
    cudaMemcpy( y1_temp.data(), FistaTGVHuberB2_id::id[id].intermediate_y[ outer ], 3 * in * ingrad_c * ih * iw * sizeof(float), cudaMemcpyHostToDevice ); // write on other 1st ..
    cudaMemcpy(  d_dout.data(), FistaTGVHuberB2_id::id[id].intermediate_x[ outer ], 3 * in * ingrad_c * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

#ifdef _DEBUG_FW_DIR
    std::cout << " outer " << outer << " " << nextLastIt << "\n";

    // loop over y's; 1st create all x in between
    std::cout << outer << " Y vector " << nextLastIt << "  " << FistaTGVHuberB2_id::id[id].intermediate_y[ outer ][0] << "  " << FistaTGVHuberB2_id::id[id].intermediate_y[ outer ][1] << "\n";
    std::cout << outer << " x vector " << nextLastIt << "  " << FistaTGVHuberB2_id::id[id].intermediate_x[ outer ][0] << "  " << FistaTGVHuberB2_id::id[id].intermediate_x[ outer ][1] << "\n";

    std::cout << " Loaded x-vec is :\n";
    std::vector<float> dgb1(10, 0);
    cudaMemcpy( dgb1.data(), d_dout.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
    std::cout << " " << dgb1[0] << " " << dgb1[1] << " " << dgb1[2] << " " << dgb1[3] << "\n";
    std::cout << " Loaded y-vec is :\n";
    cudaMemcpy( dgb1.data(), y0_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
    std::cout << " " << dgb1[0] << " " << dgb1[1] << " " << dgb1[2] << " " << dgb1[3] << "\n";
    cudaMemcpy( dgb1.data(), y1_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
    std::cout << " " << dgb1[0] << " " << dgb1[1] << " " << dgb1[2] << " " << dgb1[3] << "\n";

    std::cout << "Iterations to do are : " << (lastIt - nextLastIt) << "\n";
#endif

    // build d_xk05 .. for the smallish subpart of iterations ..
    for ( int it = 0; it < (lastIt - nextLastIt) / 2; it++ )
    {
      float step = FistaTGVHuberB2_id::id[id].stepSizes[ nextLastIt + 2 * it    ];
#ifdef _DEBUG_FW_DIR
      std::cout << "---------------- inner " << 2 * it << "-> " << nextLastIt + 2 * it << "\n";
      std::cout << "parameters are : " << 2 * it << " " << samples << " " << " " << ingrad_c << " " <<  step << "\n";
#endif
      // attention, my first thing is ONLY FROM Y, generate whole process. what do i need to kickstart ?
      // well only from y .. not possible. i get xk+0.5, x+k+1 but next y .. not possible
      // i need x AND y, xk and yk
      if (ic == 2)
        TGVB2_simple_nCalls_2D_tex_clean_writexk05_ch<2> <<< dimGrid_ch, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_db, d_dout, y0_temp, y1_temp, d_xk05, 2 * it, step, oneL ); //
      else
        TGVB2_simple_nCalls_2D_tex_clean_writexk05 <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_db, d_dout, y0_temp, y1_temp, d_xk05, 2 * it, ic, step, oneL ); //samples,

#ifdef _DEBUG_FW_DIR
      std::vector<float> dgb(10, 0);
      cudaMemcpy( dgb.data(), d_dout.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << "d_out: " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";

      cudaMemcpy( dgb.data(), d_xk05.data( (2 * it)*ingrad_c * iw * ih ), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << "d_xk05: " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";

      std::cout << " Loaded y-vec 1 is again :\n";
      cudaMemcpy( dgb1.data(), y1_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << "y1: " << dgb1[0] << " " << dgb1[1] << " " << dgb1[2] << " " << dgb1[3] << "\n";
#endif

      step = FistaTGVHuberB2_id::id[id].stepSizes[ nextLastIt + 2 * it + 1 ];

      if (ingrad_c == 2)
        TGVB2_simple_nCalls_2D_tex_clean_writexk05_ch<2> <<< dimGrid_ch, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_db, d_dout, y1_temp, y0_temp, d_xk05, 2 * it + 1, step, oneL );
      else
        TGVB2_simple_nCalls_2D_tex_clean_writexk05 <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(), d_db, d_dout, y1_temp, y0_temp, d_xk05, 2 * it + 1, ingrad_c, step, oneL ); //samples,

#ifdef _DEBUG_FW_DIR
      std::cout << "---------------- inner " << 2 * it + 1 << "-> " << nextLastIt + 2 * it + 1 << "\n";
      cudaMemcpy( dgb.data(), d_dout.data( ), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << "d_out " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";

      cudaMemcpy( dgb.data(), d_xk05.data( (2 * it + 1)*ingrad_c * iw * ih ), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << "d_xk05: " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
#endif
    }
#ifdef _DEBUG_FW_DIR
    std::vector<float> dgb(10, 0);
    // check if last output == last .. etc . debuggin help
    std::cout << "x05 build: " << (lastIt - nextLastIt - 1) << "\n";
    cudaMemcpy( dgb.data(), d_xk05.data( ingrad_c * iw * ih * (lastIt - nextLastIt - 1) ), 10 * sizeof(float), cudaMemcpyDeviceToHost );
    std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
    cudaMemcpy( dgb.data(), d_xk05.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
    std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
    if ( outer <  FistaTGVHuberB2_id::id[id].iterations.size() - 1 )
    {
      std::cout << "DOUT must be equal to checkmarks stored, ie : " << "\n";
      cudaMemcpy( dgb.data(), d_dout.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
      std::cout << " must be equal to : " << "\n";
      std::cout << outer << " Y vector " << nextLastIt << "  " << FistaTGVHuberB2_id::id[id].intermediate_x[ outer + 1 ][0] << "  " << FistaTGVHuberB2_id::id[id].intermediate_x[ outer + 1 ][1] << "\n";
      std::cout << "DOUT must be equal to checkmarks stored, ie : " << "\n";
      cudaMemcpy( dgb.data(), y0_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
      std::cout << outer << " x vector " << nextLastIt << "  " << FistaTGVHuberB2_id::id[id].intermediate_y[ outer + 1 ][0] << "  " << FistaTGVHuberB2_id::id[id].intermediate_y[ outer + 1 ][1] << "\n";
    }
#endif
    //////////////////////////////////////////////
    // now parse again, backward this time, aggregating gradients.
    // trick was to kick-start ?
    // need to ping pong with df/dx^(k+0.5) -> xtra kernel to init this one ..
    //
    // now can do special kernel to fill-in 1st part of df/dx05 df/dc df/db ....  df/dw?
    // short xk_05_id = (lastIt-nextLastIt-1);
    // now backwards kernel
    //
    // here the d_gxk05_0_temp, d_gxk05_1_temp, d_gxk0_temp, d_gxk1_temp grow large
    // and are differences of large numbers often .. -> dramatic loss of precision..
    for ( int it = (lastIt - nextLastIt) / 2 - 1; it >= 0; it-- )
    {
      float step = FistaTGVHuberB2_id::id[id].stepSizes[ nextLastIt + 2 * it + 1 ];
      nIts += 1.;
      // attention, my first thing is ONLY FROM Y, generate whoel process. what do i need to kickstart ?
      // well only from y .. not possible. i get xk+0.5, x+k+1 but next y .. not possible
      // Hence, I need x AND y, xk and yk
      // df/dy intern. extern: df/dxk, df/dxk-1, d_gxk0_temp, d_gxk1_temp
      //d_xk05 load: it is in cudaMemcpy( d_xk05.data(ic*iw*ih * (2*it+1) ), x_temp.data(), ic*ih*iw*sizeof(float), cudaMemcpyDeviceToDevice ); channel/sample it!
      if ( outer == FistaTGVHuberB2_id::id[id].iterations.size() - 1  && it == ((lastIt - nextLastIt) / 2 - 1) )
        TGVB2_init_dc_db_dx05_dw <<< dimGrid, dimBlock>>>( tex_cf.getTexture(),
            d_db, d_xk05, d_gc, d_gb, d_gxk05_1_temp, d_gin, 2 * it + 1, ingrad_c, step );
      else
      {
        TGVB2_simple_get_yk <<< dimGrid, dimBlock>>>( tex_cf.getTexture(), d_db, d_xk05, d_yk_temp, 2 * it + 1, ingrad_c, step );

        if ( ingrad_c == 2 )
          TGVB2_simple_nCalls_2D_tex_bw_yk_ch<2> <<< dimGrid_ch, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(),
              d_db, d_xk05, d_gc, d_gb, d_gx, d_gy, d_gxk05_0_temp, d_gxk05_1_temp, d_gxk0_temp, d_gxk1_temp, d_yk_temp, 2 * it + 1, step, nIts, oneL );
        else
          TGVB2_simple_nCalls_2D_tex_bw_yk <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(),
              d_db, d_xk05, d_gc, d_gb, d_gx, d_gy, d_gxk05_0_temp, d_gxk05_1_temp, d_gxk0_temp, d_gxk1_temp, d_yk_temp, 2 * it + 1, ingrad_c, step, nIts, oneL );
      }
#ifdef _DEBUG_BW_DIR
      std::cout << "DB Iteartion " << 2 * it + 1 << "\n";
      cudaMemcpy( dgb.data(), d_gb.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
      std::cout << "DC " << "\n";
      cudaMemcpy( dgb.data(), d_gc.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";

      std::cout << "d_gxk0_temp " << "\n";
      cudaMemcpy( dgb.data(), d_gxk0_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
      std::cout << "d_gxk1_temp " << "\n";
      cudaMemcpy( dgb.data(), d_gxk1_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";

      std::cout << "d_gxk05_0_temp " << "\n";
      cudaMemcpy( dgb.data(), d_gxk05_0_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
      std::cout << "d_gxk05_1_temp " << "\n";
      cudaMemcpy( dgb.data(), d_gxk05_1_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
#endif

      nIts += 1.;
      step = FistaTGVHuberB2_id::id[id].stepSizes[ nextLastIt + 2 * it    ];
      if (it > 0)
      {
        TGVB2_simple_get_yk <<< dimGrid, dimBlock>>>( tex_cf.getTexture(), d_db, d_xk05, d_yk_temp, 2 * it, ingrad_c, step );

        if ( ingrad_c == 2 )
          TGVB2_simple_nCalls_2D_tex_bw_yk_ch<2> <<< dimGrid_ch, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(),
              d_db, d_xk05, d_gc, d_gb, d_gx, d_gy, d_gxk05_1_temp, d_gxk05_0_temp, d_gxk1_temp, d_gxk0_temp, d_yk_temp, 2 * it, step, nIts, oneL );
        else
          TGVB2_simple_nCalls_2D_tex_bw_yk <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(),
              d_db, d_xk05, d_gc, d_gb, d_gx, d_gy, d_gxk05_1_temp, d_gxk05_0_temp, d_gxk1_temp, d_gxk0_temp, d_yk_temp, 2 * it, ingrad_c, step, nIts, oneL );
      }
      else
      {
        cudaMemcpy( y0_temp.data(), FistaTGVHuberB2_id::id[id].intermediate_y[ outer ], 3 * in * ingrad_c * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

        TGVB2_end_2D_tex_bw_yk <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), tex_betaX.getTexture(), tex_betaY.getTexture(),
            d_db, d_xk05, d_gc, d_gb, d_gx, d_gy, y0_temp, d_gxk05_1_temp, d_gxk05_0_temp, d_gxk1_temp, d_gxk0_temp, 2 * it, ingrad_c, step, nIts, oneL );
      }

#ifdef _DEBUG_BW_DIR
      std::cout << "DB Iteartion " << 2 * it << "\n";
      cudaMemcpy( dgb.data(), d_gb.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
      std::cout << "DC " << "\n";
      cudaMemcpy( dgb.data(), d_gc.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";

      std::cout << "d_gxk0_temp " << "\n";
      cudaMemcpy( dgb.data(), d_gxk0_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
      std::cout << "d_gxk1_temp " << "\n";
      cudaMemcpy( dgb.data(), d_gxk1_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";

      std::cout << "d_gxk05_0_temp " << "\n";
      cudaMemcpy( dgb.data(), d_gxk05_0_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
      std::cout << "d_gxk05_1_temp " << "\n";
      cudaMemcpy( dgb.data(), d_gxk05_1_temp.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost );
      std::cout << " " << dgb[0] << " " << dgb[1] << " " << dgb[2] << " " << dgb[3] << "\n";
#endif
    }
    std::vector<float> vf( 3 * ingrad_c * ih * iw, 0 );
    double maxG = 0;
    double localScaleFactor = 1.;
    cudaMemcpy( vf.data(), d_gxk05_0_temp.data(), 3 * ingrad_c * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
    for (int i = 0; i < vf.size(); i++)
      maxG = std::max( maxG, (double)std::abs(vf[i]) );

    copy_and_set_to_0( gc_storage, gx_storage, gy_storage, gb_storage, d_gb, d_gc, d_gx, d_gy, c_dims, ingrad_c, ih, iw, 1., 2, 2 );
    /////////////////////////////////////////
    lastIt    = nextLastIt;// 'one down
  }

#ifdef _pngWriting_
  if (id == 0)
  {
    /*
      pngwriter png(iw,ih,0,"test.png");
      for(int i =0;i< ih* iw;i++)
        png.plot( i%iw, (ih-1) - i/iw, std::min( gx_storage[i]/1000., 1.), std::min( gy_storage[i]/1000., 1.), std::min( gc_storage[i]/1000., 1.) );
      png.close();
    */
    const double scalePNG = 1. / plotClipValue; //globalGMult
    pngwriter png(iw, ih, 0, "dX.png");
    for (int i = 0; i < ih * iw; i++)
      png.plot( i % iw, (ih - 1) - i / iw, std::max(0., std::min( -10.*gx_storage[i] * scalePNG, 1.) ), std::max(0., std::min( 10.*gx_storage[i] * scalePNG, 1.) ), 0. );
    png.close();
    pngwriter png1(iw, ih, 0, "dY.png");
    for (int i = 0; i < ih * iw; i++)
      png1.plot( i % iw, (ih - 1) - i / iw, std::max(0., std::min( -10.*gy_storage[i] * scalePNG, 1.) ), std::max(0., std::min( 10.*gy_storage[i] * scalePNG, 1.) ), 0. );
    png1.close();
    pngwriter png2(iw, ih, 0, "dC.png");
    for (int i = 0; i < ih * iw; i++)
      png2.plot( i % iw, (ih - 1) - i / iw, std::max(0., std::min( -10.*gc_storage[i] * scalePNG, 1.) ), std::max(0., std::min( 10.*gc_storage[i] * scalePNG, 1.) ), 0. );
    png2.close();
    pngwriter png3(iw, ih, 0, "dB.png");
    for (int i = 0; i < ih * iw; i++)
      png3.plot( i % iw, (ih - 1) - i / iw, std::max(0., std::min( -300. / localGBMult * gb_storage[i] * scalePNG, 1.) ), std::max(0., std::min( 300. / localGBMult * gb_storage[i] * scalePNG, 1.) ), std::max(0., std::min( 300. / localGBMult * gb_storage[i + ih * iw] * scalePNG, 1.) ) );
    png3.close();
  }
#endif

  /*
    scaleToClip( gc_storage, clipvalue );
    scaleToClip( gy_storage, clipvalue );
    scaleToClip( gx_storage, clipvalue );
    scaleToClip( gb_storage, clipvalue );
  */

#ifdef _NAN_Output_Check_
  double run_step = std::min( _memory_T_, double(FistaTGVHuberB2_id::id[id].run) / double(FistaTGVHuberB2_id::id[id].run + 1) );
  double is_largest = 0; int num_above = 0; double av_gc = 0;

  check_gradUpdate( gc_storage, av_gc, is_largest, num_above, FistaTGVHuberB2_id::id[id].av_GC, "C" );
  update_correct_av_step_host(FistaTGVHuberB2_id::id[id].run, av_gc, FistaTGVHuberB2_id::id[id].av_GC, gc_storage, run_step, "C" );
  if (FistaTGVHuberB2_id::id[id].run % __debug_info_period__ == 0)
    std::cerr << "\nL " << id << " Output gradient to conf  " <<  num_above << " " << av_gc << "  vs  " << FistaTGVHuberB2_id::id[id].av_GC << " vs " << is_largest << " ";

  check_gradUpdate( gx_storage, av_gc, is_largest, num_above, FistaTGVHuberB2_id::id[id].av_GX, "X" ); // XXhere
  update_correct_av_step_host(FistaTGVHuberB2_id::id[id].run, av_gc, FistaTGVHuberB2_id::id[id].av_GX, gx_storage, run_step, "X" );
  if (FistaTGVHuberB2_id::id[id].run % __debug_info_period__ == 0)
    std::cerr << "\nL " << id << " Output gradient to gradX  " <<  num_above << " " << av_gc << "  vs  " << FistaTGVHuberB2_id::id[id].av_GX << " vs " << is_largest << "   ";

  check_gradUpdate( gy_storage, av_gc, is_largest, num_above, FistaTGVHuberB2_id::id[id].av_GY, "Y" );
  update_correct_av_step_host(FistaTGVHuberB2_id::id[id].run, av_gc, FistaTGVHuberB2_id::id[id].av_GY, gy_storage, run_step, "Y" );
  if (FistaTGVHuberB2_id::id[id].run % __debug_info_period__ == 0)
    std::cerr << "\nL " << id << " Output gradient to gradY  " <<  num_above << " " << av_gc << "  vs  " << FistaTGVHuberB2_id::id[id].av_GY << " vs " << is_largest << "   ";

  check_gradUpdate( gb_storage, av_gc, is_largest, num_above, FistaTGVHuberB2_id::id[id].av_GB, "B" );
  update_correct_av_step_host(FistaTGVHuberB2_id::id[id].run, av_gc, FistaTGVHuberB2_id::id[id].av_GB, gb_storage, run_step, "B" );
  if (FistaTGVHuberB2_id::id[id].run % __debug_info_period__ == 0)
    std::cerr << "\nL " << id << "  Output gradient to gradB  " <<  num_above << " " << av_gc << "  vs  " << FistaTGVHuberB2_id::id[id].av_GB << " vs " << is_largest << "   ";

  std::vector<float> vf( ingrad_c * iw * ih, 0 );
  cudaMemcpy( vf.data(), d_gxk1_temp.data(), ingrad_c * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );

  check_gradUpdate( vf, av_gc, is_largest, num_above, FistaTGVHuberB2_id::id[id].av_GI, "I" );
  update_correct_av_step_host(FistaTGVHuberB2_id::id[id].run, av_gc, FistaTGVHuberB2_id::id[id].av_GI, vf, run_step, "I" );
  if (FistaTGVHuberB2_id::id[id].run % __debug_info_period__ == 0)
    std::cerr << "\nL " << id << "  Output gradient to gradI  " <<  num_above << " " << av_gc << "  vs  " << FistaTGVHuberB2_id::id[id].av_GI << " vs " << is_largest << "\n\n";
  cudaMemcpy( d_gxk1_temp.data(), vf.data(), vf.size()*sizeof(float), cudaMemcpyHostToDevice );
#endif

  // copy back to float :
  std::vector<float> gc_temp( max(c_xyc, ingrad_c) * ih * iw, 0 );
  for (int i = 0; i < c_dims * ih * iw; i++)
    gc_temp[i] = (float) (gc_storage[i] * globalGMult);
  cudaMemcpy( d_gc.data(), gc_temp.data(), c_dims * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

  std::fill(gc_temp.begin(), gc_temp.end(), 0.0f);
  for (int i = 0; i < c_xyc * ih * iw; i++)
  {
    //std::cout << "i:" << i << " " << gx_storage[i] << " ";if (i==ih* iw-1) std::cout << "\n";
    gc_temp[i] = (float) (gx_storage[i] * globalGMult);
  }
  cudaMemcpy( d_gx.data(), gc_temp.data(), c_xyc * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

  std::fill(gc_temp.begin(), gc_temp.end(), 0.0f);
  for (int i = 0; i < c_xyc * ih * iw; i++)
    gc_temp[i] = (float) (gy_storage[i] * globalGMult);
  cudaMemcpy( d_gy.data(), gc_temp.data(), c_xyc * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

  std::fill(gc_temp.begin(), gc_temp.end(), 0.0f);
  for (int i = 0; i < ingrad_c * ih * iw; i++)
    gc_temp[i] = (float) (gb_storage[i] * globalGMult);
  cudaMemcpy( d_gb.data(), gc_temp.data(), ingrad_c * ih * iw * sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_gi.data(), d_gxk1_temp.data(), 3 * ingrad_c * ih * iw * sizeof(float), cudaMemcpyDeviceToDevice );

  thrust::transform(d_gx.begin(), d_gx.end(), d_gx.begin(), L * _1);
  thrust::transform(d_gy.begin(), d_gy.end(), d_gy.begin(), L * _1);

#ifdef clipvalue
  const float clip = clipvalue;
  auto ff = [ = ]  __device__ (float x) {return max(-clip, min(x, clip));};
  thrust::transform(d_gb.begin(), d_gb.end(), d_gb.begin(), ff);
  thrust::transform(d_gc.begin(), d_gc.end(), d_gc.begin(), ff);
  thrust::transform(d_gx.begin(), d_gx.end(), d_gx.begin(), ff);
  thrust::transform(d_gy.begin(), d_gy.end(), d_gy.begin(), ff);
#endif

  FistaTGVHuberB2_id::id[id].run++;
  return 0;
}

#undef beta
#undef __Huber1__
#undef __HDelta2__
#undef __HDeltaY1__
#undef __HDeltaY2__
#undef _NAN_Input_Check_
#undef _NAN_Output_Check_
#undef plotClipValue
#undef _pngWriting_
#undef COMMON_BLOCK_SIZE_2D_X
#undef COMMON_BLOCK_SIZE_2D_Y
#undef __HuberRegularizer__
#undef clipLocalGrad
#undef localGCMult
#undef localGXMult
#undef localGBMult
#undef globalGMult