#include <cuda_runtime.h>
#include "TVInpaintFista.h"

// produce png updates from the gradients
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
#include <iostream>

#include <cuda.h>
#include <iu/iucore.h>

// write out debug information about gradient sizes every __debug_info_period__ learning steps.
#define __debug_info_period__ 75

// print debug output
#define _NAN_Input_Check_
// not optional:
#define _NAN_Output_Check_

// total max .. 32 x 32 or 32 x 16 work same..
#define COMMON_BLOCK_SIZE_2D_X 32
#define COMMON_BLOCK_SIZE_2D_Y 16

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
// delta on huber in regularizer:
#define __HDelta1__ 0.1f
////////////////////////////////////

// Clip large gradients. not needed, even works a bit worse.
//#define clipvalue 3000.25f

// clip local gradients before adding to buffer to avoid overflows. (Double buffers make this more or less obsolete). Keep for safety.
#define clipLocalGrad 10000.0f
//#define clipLocalGrad 3000.0f

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

////////////////////////////////////////////////////
//
// If enabled gradient updates that are much larger ( scale_T * runningmean < |grad| < _kill_T_ * runningmean)
// than the running average  are scaled to be at max _scale_T_ larger than running average.
// Superlarge ones are ignored ( > _kill_T_ * runningmean )
// Why? Assume a simple optimization scheme as RMSProp it scales the gradients by the runnig mean.
// If one is much large than this running mean we end up in an explosion.
// This is happening very rarely, but the check solves all problems here, without negative influence.
//
// if gradient are _kill_T_ larger than runnig mean, ignore them all (See above rmsprop example).
#define _kill_T_  50.
//
// if gradient are _scale_T_ larger than runnig mean, rescale them to avoid too
// large updates due to the optimization algorithm. (See above rmsprop example)
#define _scale_T_ 10.
// memory / update rate of running mean
#define _memory_T_ 0.975
////////////////////////////////////////////////////


template<typename T>
bool is_infinite( const T &value )
{
  T max_value = std::numeric_limits<T>::max();
  T min_value = - max_value;

  return ! ( min_value <= value && value <= max_value );
}

template<typename T>
bool is_nan( const T &value )
{
  return value != value;
}

template<typename T>
bool is_valid( const T &value )
{
  return ! is_infinite(value) && ! is_nan(value);
}

// not used. For debugging only.
inline void scaleToClip( std::vector<double>& gc_storage, double clipVal )
{
  double maxVal = 0;
  for (int i = 0; i < gc_storage.size(); i++)
    maxVal = std::max( maxVal, std::abs (gc_storage[i]) );
  double scale = clipVal / maxVal;
  if (clipVal > maxVal)
    for (int i = 0; i < gc_storage.size(); i++)
      gc_storage[i] *= scale;
}

// at the end of the iterations, copy the double buffer back to the GPU arrays.
template <typename S>
void update_correct_av_step_host(int run, double av_gc, double& av_GC, std::vector<S>&tempc, double run_step, const char* XX )
{
  // 1st idea: rescale .. or ignore at all ? -> set to 0 //  // cutoff the large ones ?
  if ( run >= 10 && av_GC * _kill_T_ < av_gc ) // or never kill ? as this always fucks up everything ?
  {
    std::cerr << "Killed the update  step for " << XX[0] << ": " << av_gc << " vs " << av_GC << " = " << av_gc / av_GC << " \n";
    // if set to 0 all paraemters below are set to 0 instead of updated .. ? incredible
    for (int i = 0; i < tempc.size(); i++) tempc[i] = 0.0;

    av_GC = av_GC * run_step + (1. - run_step) * _scale_T_ * av_GC; // increase running mean at least
    return;
  }
  const double scale = _scale_T_;// or larger .. note that it appeared as if there never was an update on grad_store done before ?
  if ( run >= 10 && av_GC * scale < av_gc )
  {
    double mult = scale * av_GC / av_gc;
    std::cerr << "Corrected the step " << XX[0] << " " << mult << ": " << av_gc << " vs " << av_GC << " = " << av_gc / av_GC << " \n";
    for (int i = 0; i < tempc.size(); i++)
      tempc[i] *= mult;

    av_GC = av_GC * run_step + (1. - run_step) * av_gc * mult; // increase it nevertheless
  }
  else
  {
    av_GC = av_GC * run_step + (1. - run_step) * av_gc;
  }
}

// informs whether returned gradients are NAN or INF (or just very large). Sets those to 0.
template <typename S> void check_gradUpdate( std::vector<S>& tempc, double &av_gc, double& is_largest, int& num_above, double av_gc_current, const char* XX )
{
  const double maxTolerance = 10000000.;//std::numeric_limits<float>::max() / 100.f
  is_largest = 0; num_above = 0; av_gc = 0; int invlid = 0;
  for (int i = 0; i < tempc.size(); i++)
    if (is_valid( tempc[i]) ) {
      double tca = std::abs(tempc[i]);
      double step  = double(i) / double(i + 1);
      av_gc =  (1. - step) * tca + step * av_gc;
      is_largest = std::max( tca, is_largest );
      if ( tca > maxTolerance ) {tempc[i] = 0; num_above++;}
    }
    else
    {tempc[i] = 0; av_gc = av_gc_current * 1000.; invlid++;} //maybe better is to kill ALL

  if (invlid > 0) //tempc.size() << " "<<
    std::cerr << "Invalid gradient updates detected: " << XX[0] << " " << invlid << "\n";
}

// update gradients accumulated during 2 checkpoints to the double buffer & set gpu buffer to 0
inline void copy_and_set_to_0( std::vector<double>& gc_storage, std::vector<double>& gx_storage,
                               std::vector<double>& gy_storage, std::vector<double>& gb_storage,
                               iu::TensorGpu_32f& d_gb, iu::TensorGpu_32f& d_gc,
                               iu::TensorGpu_32f& d_gx, iu::TensorGpu_32f& d_gy,
                               int ic, int ingrad_c, int ih, int iw, float scale = 1. )
{
  // fetch and store and add
  std::vector<float> gc_temp( std::max(ic, ingrad_c) * ih * iw, 0 );
  double maxGCGrad(0);
  cudaMemcpy( gc_temp.data(), d_gc.data(), ic * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
  for (int i = 0; i < ic * ih * iw; i++)
  {
    gc_storage[i] = (double) (gc_temp[i]) + gc_storage[i] / scale;
    maxGCGrad = std::max( maxGCGrad, (double) std::abs(gc_temp[i])  );
  }
  thrust::fill(d_gc.begin(), d_gc.end(), 0.0f);

  maxGCGrad = 0; cudaMemcpy( gc_temp.data(), d_gx.data(), ic * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
  for (int i = 0; i < ic * ih * iw; i++)
  {
    gx_storage[i] = (double) (gc_temp[i]) + gx_storage[i] / scale;
    maxGCGrad = std::max( maxGCGrad, (double) std::abs(gc_temp[i])  );
  }
  thrust::fill(d_gx.begin(), d_gx.end(), 0.0f);

  maxGCGrad = 0; cudaMemcpy( gc_temp.data(), d_gy.data(), ic * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
  for (int i = 0; i < ic * ih * iw; i++)
  {
    gy_storage[i] = (double) (gc_temp[i]) + gy_storage[i] / scale;
    maxGCGrad = std::max( maxGCGrad, (double) std::abs(gc_temp[i])  );
  }
  thrust::fill(d_gy.begin(), d_gy.end(), 0.0f);

  maxGCGrad = 0; cudaMemcpy( gc_temp.data(), d_gb.data(), ingrad_c * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
  for (int i = 0; i < ingrad_c * ih * iw; i++)
  {
    gb_storage[i] = (double) (gc_temp[i]) + gb_storage[i] / scale;
    maxGCGrad = std::max( maxGCGrad, (double) std::abs(gc_temp[i])  );
  }
  thrust::fill(d_gb.begin(), d_gb.end(), 0.0f);
}


// just called once and once at all ..
__global__ void FistaInpaint_init_dc_db_dx05_dw( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv_05,
    iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb,
    iu::TensorGpu_32f::TensorKernelData d_gxk05, iu::TensorGpu_32f::TensorKernelData d_gxk1,
    short it , short n = 0, short channels = 1, float t = 1)
{

  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  if ( (w >= bv.W || h >= bv.H) )
    return;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  for ( short ch = 0; ch < channels; ch++ )
  {
    float df_dxk = d_gxk1(n, ch, h, w);
#ifdef __Huber1__
    const float diff = bv(n, ch, h, w) - xv_05(it, ch, h, w);
    const float sign = (diff < 0) ? -1. : 1.;
    if ( 1. + conf >= sign * diff )
    {
      // quadratic case EXACTLY: x_loc = (conf * b + x05) / (1.f+conf);
      d_gxk05(n, ch, h, w) = df_dxk / (1.f + conf);
      d_gc(n, 0, h, w)    += df_dxk * ( bv(n, ch, h, w) / (1.f + conf)  - ( conf * bv(n, ch, h, w) + xv_05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult; // actually add
      d_gb(n, ch, h, w)   += df_dxk * conf / (1.f + conf) * localGBMult;
    }
    else // c>diff (x,b) -> x+-c is valid
    {
      d_gxk05(n, ch, h, w) = df_dxk;
      d_gc(n, 0, h, w)    += sign * df_dxk * localGCMult; // actually add
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

    // gc huge here : adds b here .. all the time .. -> scale flow to be in 0/1 ? but what about conf then -> scale also ? and smooth ?
    d_gc(n, 0, h, w)    += df_dxk * ( bv(n, ch, h, w) / (1.f + conf)  - ( conf * bv(n, ch, h, w) + xv_05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult; // actually add
    d_gb(n, ch, h, w)   += df_dxk * conf / (1.f + conf) * localGBMult;
#else //L1
    const float diff = bv(n, ch, h, w) - xv_05(it, ch, h, w);
    const float sign = (diff < 0) ? -1. : 1.;

    if ( conf >= sign * diff )
    {
      d_gb(n, ch, h, w)    = df_dxk * localGBMult;
      d_gxk05(n, ch, h, w) = 0; // set to 0 always
    }
    else // c>diff (x,b) -> x+-c is valid
    {
      d_gxk05(n, ch, h, w) = df_dxk;
      d_gc(n, 0, h, w)    += sign * df_dxk * localGCMult;
      d_gb(n, ch, h, w)    = 0;
    }
#endif  // quadratic or L1 if not 
#endif  // Huber data term is used
  }
};

__global__ void FistaInpaint_end_2D_tex_bw( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is !! k-0.5 !! at it, at it-1 the one before that
    iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb,
    iu::TensorGpu_32f::TensorKernelData d_gwx, iu::TensorGpu_32f::TensorKernelData d_gwy, iu::TensorGpu_32f::TensorKernelData yv_0,
    iu::TensorGpu_32f::TensorKernelData d_gxk05_in, iu::TensorGpu_32f::TensorKernelData d_gxk05_out,
    iu::TensorGpu_32f::TensorKernelData d_gxk, iu::TensorGpu_32f::TensorKernelData d_gxk_m1,
    short it, short n = 0, short channels = 1, float t = 1, float nits = 1. )
{
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

  __shared__ float dx05_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!

  for ( short ch = 0; ch < channels; ch++ )
  {
    // onto 2 tiling .. alright ..
    short local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    short local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    short global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    short global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // read local_unrolled_x + blockDim.x * blockIdx.x, local_unrolled_y + blockDim.y * blockIdx.y -(1,1)
    // write to local id: local_unrolled_y, local_unrolled_x
    // load if inside ..
    if (ch > 0) __syncthreads(); // since y has to be loaded completely ..

    dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;
    // load 2
    if ( local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
      dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    __syncthreads(); // since y has to be loaded completely ..

    if ( !(w >= bv.W || h >= bv.H) )
    {
      /////////////////////////////// steps 5:
      //     y = y + (1/L)*laplace_W*y;
      float df_dy = dx05_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy_t = df_dy;//temporary it is only ..

      if ( h > 0 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
      if ( h < bv.H - 1 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;
      if ( w > 0 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x]) * wx_0;
      if ( w < bv.W - 1 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;
      ////////////////////////////////////////////////////////////////////////
      /////////////////////////////// steps 6 and 7 ///////////////////
      float df_dxk = d_gxk(n, ch, h, w) + (1. + t) * df_dy_t;
#ifdef clipLocalGrad
      df_dxk = max( -clipLocalGrad, min (df_dxk, clipLocalGrad) ); // sole grads go nuts .. still no idea why but this happens ..
#endif
      d_gxk(n, ch, h, w)    = df_dxk;
      d_gxk_m1(n, ch, h, w) = -t * df_dy_t;
      ////////////////////////////////////////////////////////////////////////
      // steps 2,3,4
      const float bb   = bv(n, ch, h, w);

#ifdef __Huber1__

      const float diff = bb - xv_m05(it, ch, h, w);
      const float sign = (diff < 0) ? -1. : 1.;
      if ( 1. + conf >= sign * diff )
      {
        // quadratic case EXACTLY:
        // x = (conf.*b+x05)./(1+conf); derivative is
        // df/dconf = df/dx *  b./(1+conf) - conf*b/*(1+conf*conf)
        // df/db    = df/dx * conf/(1+conf)
        // df/dy    = df/dx *  1/(1+conf)
        // df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
        d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
        d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
        d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf)  * localGBMult;
      }
      else // L1 case exactly
      {
        if ( conf >= sign * diff )
        {
          d_gxk05_out(n, ch, h, w) = 0; // set to 0 indeed, always
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
      // x = (conf.*b+x05)./(1+conf); derivative is
      // df/dconf = df/dx *  b./(1+conf) - conf*b/*(1+conf*conf)
      // df/db    = df/dx * conf/(1+conf)
      // df/dy    = df/dx *  1/(1+conf)
      // df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
      d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
      d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
      d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf)  * localGBMult;
#else // L1 data term
      const float diff = bb - xv_m05(it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      if ( conf >= sign * diff )
      {
        d_gxk05_out(n, ch, h, w) = 0; // set to 0 indeed, always
        d_gb(n, ch, h, w)       += df_dxk * localGBMult;
      }
      else
      {
        d_gxk05_out(n, ch, h, w) = df_dxk;
        d_gc(n, 0, h, w)       += sign * df_dxk * localGCMult;
      }
#endif // not quadratic -> L1
#endif // HUBER

      const float yk   = yv_0(n, ch, h, w);
      if (w > 0)
        atomicAdd( &(d_gwx(n, 0, h,   w - 1)), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 1][threadIdx.x  ]) * yk ); // dx05_prev[threadIdx.y][threadIdx.x+1]
      if (w < bv.W - 1)
        atomicAdd( &(d_gwx(n, 0, h,   w  )), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * yk ); //
      if (h > 0)
        atomicAdd( &(d_gwy(n, 0, h - 1, w)  ), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y  ][threadIdx.x + 1]) * yk ); //
      if (h < bv.H - 1)
        atomicAdd( &(d_gwy(n, 0, h,   w)  ), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * yk ); //
      ////////////////////////////////
    }
  }
};

// that is the main backward part, solved by ping pong on the variables whose neighbors (pixelwise) need to be kept at the current iteration (or last -- but consistent!)
__global__ void FistaInpaint_simple_nCalls_2D_tex_bw( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is !! k-0.5 !! at it, at it-1 the one before that
    iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb,
    iu::TensorGpu_32f::TensorKernelData d_gwx, iu::TensorGpu_32f::TensorKernelData d_gwy,
    iu::TensorGpu_32f::TensorKernelData d_gxk05_in, iu::TensorGpu_32f::TensorKernelData d_gxk05_out,
    iu::TensorGpu_32f::TensorKernelData d_gxk, iu::TensorGpu_32f::TensorKernelData d_gxk_m1,
    short it, short n = 0, short channels = 1, float t = 1, float nits = 1. )
{
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

  // each block holds ~ 48KB (my card at least) ..
  __shared__ float dx05_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!

  for ( short ch = 0; ch < channels; ch++ )
  {
    // onto 2 tiling .. alright ..
    short local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    short local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    short global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    short global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // read local_unrolled_x + blockDim.x * blockIdx.x, local_unrolled_y + blockDim.y * blockIdx.y -(1,1)
    // write to local id: local_unrolled_y, local_unrolled_x
    if (ch > 0) __syncthreads(); // since y has to be loaded completely ..

    dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    if ( local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
      dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    __syncthreads(); // since y has to be loaded completely ..
    // 2 explicit loads done ..

    if ( !(w >= bv.W || h >= bv.H) )
    {
      /////////////////////////////// steps 5:
      //     y = y + (1/L)*laplace_W*y;
      const float df_dy = dx05_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy_t = df_dy;//temporary

      if ( h > 0 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
      if ( h < bv.H - 1 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;
      if ( w > 0 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x]) * wx_0;
      if ( w < bv.W - 1 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;
      /////////////////////////////////////////////////////////////////
      /////////////////////////////// steps 6 and 7 ///////////////////
      float df_dxk = ( d_gxk(n, ch, h, w) + (1. + t) * df_dy_t );
#ifdef clipLocalGrad
      df_dxk = max( -clipLocalGrad, min (df_dxk, clipLocalGrad) );
#endif
      d_gxk(n, ch, h, w)    = df_dxk;
      d_gxk_m1(n, ch, h, w) = -t * df_dy_t;
      ////////////////////////////////////////////////////////////////////////
      // steps 2,3,4
      const float bb   = bv(n, ch, h, w);

#ifdef __Huber1__ // data term

      float diff = bb - xv_m05(it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      float yk(0);
      if ( 1. + conf >= sign * diff )
      {
        d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
        d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
        d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf) * localGBMult;

        const float xk  = (conf * bb + xv_m05(it,   ch, h, w)) / (1.f + conf);
        const float xkm = (conf * bb + xv_m05(it - 1, ch, h, w)) / (1.f + conf);
        yk  = xk + t * (xk - xkm);
      }
      else // l1 case of Huber data term
      {
        if ( conf >= sign * diff )
        {
          d_gxk05_out(n, ch, h, w) = 0; // set to 0 always
          d_gb(n, ch, h, w)       += df_dxk * localGBMult;
        }
        else
        {
          d_gxk05_out(n, ch, h, w) = df_dxk;
          d_gc(n, 0, h, w)       += sign * df_dxk * localGCMult;
        }
        const float xk   = ( conf <= sign * diff   ) ? bb - diff + sign * conf : bb;
        diff = bb - xv_m05(it - 1, ch, h, w);  sign = (diff < 0) ? -1. : 1.;
        yk   = xk + t * (xk - (( conf <= sign * diff   ) ? bb - diff + sign * conf : bb) );
      }

#else // not huber case any more /////////////////////////////////

#ifdef __quadratic__ // data term
      // x = (conf.*b+x05)./(1+conf); derivative is
      // df/db    = df/dx * conf/(1+conf)
      // df/dx05  = df/dx *  1/(1+conf)
      // df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
      d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
      d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
      d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf) * localGBMult;

      const float xk  = (conf * bb + xv_m05(it,   ch, h, w)) / (1.f + conf);
      const float xkm = (conf * bb + xv_m05(it - 1, ch, h, w)) / (1.f + conf);
      const float yk  = xk + t * (xk - xkm);
#else // L1 data term
      float diff = bb - xv_m05(it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      if ( conf >= sign * diff )
      {
        d_gxk05_out(n, ch, h, w) = 0; // set to 0 indeed, always
        d_gb(n, ch, h, w)       += df_dxk * localGBMult;
      }
      else
      {
        d_gxk05_out(n, ch, h, w) = df_dxk;
        d_gc(n, 0, h, w)       += sign * df_dxk * localGCMult;
      }

      const float xk   = ( conf <= sign * diff   ) ? bb - diff + sign * conf : bb;
      diff = bb - xv_m05(it - 1, ch, h, w);  sign = (diff < 0) ? -1. : 1.;
      const float yk   = xk + t * (xk - (( conf <= sign * diff   ) ? bb - diff + sign * conf : bb) );
#endif // quadratic or L1
#endif // huber or (quadratic or L1)
      if (w > 0)
        atomicAdd( &(d_gwx(0, 0, h,   w - 1)), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 1][threadIdx.x  ]) * yk ); // dx05_prev[threadIdx.y][threadIdx.x+1]
      if (w < bv.W - 1)
        atomicAdd( &(d_gwx(0, 0, h,   w  )), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * yk ); //
      if (h > 0)
        atomicAdd( &(d_gwy(0, 0, h - 1, w)  ), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y  ][threadIdx.x + 1]) * yk ); //
      if (h < bv.H - 1)
        atomicAdd( &(d_gwy(0, 0, h,   w)  ), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * yk ); //
      ////////////////////////////////
    }
  }
};


// that is the main backward part, solved by ping pong on the variables whose neighbors (pixelwise) need to be kept at the current iteration (or last -- but consistent!)
__global__ void FistaInpaint_simple_nCalls_2D_tex_bw_yk( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is !! k-0.5 !! at it, at it-1 the one before that
    iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb,
    iu::TensorGpu_32f::TensorKernelData d_gwx, iu::TensorGpu_32f::TensorKernelData d_gwy,
    iu::TensorGpu_32f::TensorKernelData d_gxk05_in, iu::TensorGpu_32f::TensorKernelData d_gxk05_out,
    iu::TensorGpu_32f::TensorKernelData d_gxk, iu::TensorGpu_32f::TensorKernelData d_gxk_m1, iu::TensorGpu_32f::TensorKernelData d_y, // new, now yk known from here on
    short it, short n = 0, short channels = 1, float t = 1, float nits = 1. )
{
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

// Tiling to use shared memory efficiently
#ifdef __HDelta1__
  __shared__ float y_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
#endif

// each block holds ~ 48KB (my card at least) ..
  __shared__ float dx05_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!

  for ( short ch = 0; ch < channels; ch++ )
  {
// step 0: compute and keep weights! from yk given as input !! -> need shared mem once, store 6 weights, continue as before..
    const float yk   = d_y(n, ch, h, w);
    short local_unrolled_x,  local_unrolled_y, global_unrolled_x, global_unrolled_y;
////////////////////////////////////////////////////////////////////////////////////////////////////////
    // onto 2 tiling .. alright ..
    local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    if (ch > 0) __syncthreads(); // since y has to be loaded completely ..
#ifdef __HDelta1__
    y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_y(n, ch, global_unrolled_y, global_unrolled_x) : 0;
#endif
    dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // load 2
    if ( local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
    {
#ifdef __HDelta1__
      y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_y(n, ch, global_unrolled_y, global_unrolled_x) : 0;
#endif
      dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;
    }
    __syncthreads(); // since y has to be loaded completely ..
////////////////////////////////////////////////////////////////////////////////////////////////////
// 2 explicit loads ..

    if ( !(w >= bv.W || h >= bv.H) )
    {
/////////////////////////////// steps 5:
//     y = y + (1/L)*laplace_W*y;
      const float df_dy = dx05_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy_t = df_dy;//temporary it is only ..

#ifdef __HDelta1__
      if ( h > 0 )
      {
        const float wx_2 = (h > 0 && w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h - 0.5f) : 0; // *L
        const float wy_nyy2_T2 = (yk - y_prev[threadIdx.y][threadIdx.x + 1]) * (yk - y_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
        const float wx_nyx2_T2 = (y_prev[threadIdx.y][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 2]) * (y_prev[threadIdx.y][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 2]) * wx_2;

        const float hub_weight = sqrtf( wx_nyx2_T2 + wy_nyy2_T2 );//sqrt(1/L)
        if ( hub_weight > __HDelta1__ ) // max(1.f,hub_weight/__HDelta1__)
        {
          const float nom = __HDelta1__ / hub_weight;//delta * A^{-1/2} -> 1/sqrt(1/L) =   -> sqrt(L)
          const float denom =  nom  / ( 2.f * hub_weight * hub_weight ); //delta/2 * A^{-3/2} -> sqrt(L)^3

          //(df/dxk - df/dxu) * 1/sqrt(A) * w_uk: ok
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * (wy_0 * nom); // -> sqrt(L)/L = sqrt(1/L)

          //////////////////////////////////////////////////////////////
          // temp is the derivative of the denominator wrt to yk, whose derivative we are computing
          // delta * A^{-3/2} * (y_k-y_u) * w_ku : ok
          const float temp = -denom * 2.0f * (yk  - y_prev[threadIdx.y][threadIdx.x + 1]) * wy_0; // -> sqrt(L)^3 * 1/L = sqrt(L)
          // (df/dxk - df/dxu) * w_ku * (y_k-y_u) * temp : ok
          df_dy_t -=  (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wy_0 * (yk  - y_prev[threadIdx.y][threadIdx.x + 1]) * temp; // 1/L * sqrt(L) = sqrt(1/L)
          // (df/dxt - df/dxu) * w_tu * (y_t-y_u) * temp : ok
          df_dy_t -=  (dx05_prev[threadIdx.y][threadIdx.x + 2] - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wx_2 * (y_prev[threadIdx.y][threadIdx.x + 2]  - y_prev[threadIdx.y][threadIdx.x + 1]) * temp; // sqrt(1/L)
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i+1,j - i,j) * y(i+1,j  - i,j  ) * w_(i,j / i+1,j) * temp
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i,j+1 - i,j) * y(i,j+1  - i,j  ) * w_(i,j / i,j+1) * temp
          //////////////////////////////////////////////////////////////
        }
        else
        {
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wy_0; // -> 1/L
        }
      }
      if ( w > 0 )
      {
        const float wy_2 = (w > 0 && h < bv.H - 1) ? tex2D<float>(dy, w - 0.5f, h + 0.5f) : 0;
        const float wx_nyx2_T1 = (yk - y_prev[threadIdx.y + 1][threadIdx.x]) * (yk - y_prev[threadIdx.y + 1][threadIdx.x]) * wx_0;
        const float wy_nyy2_T1 = (y_prev[threadIdx.y + 1][threadIdx.x] - y_prev[threadIdx.y + 2][threadIdx.x]) * (y_prev[threadIdx.y + 1][threadIdx.x] - y_prev[threadIdx.y + 2][threadIdx.x]) * wy_2;

        const float hub_weight = sqrtf( wx_nyx2_T1 + wy_nyy2_T1 );

        if ( hub_weight > __HDelta1__ ) // max(1.f,hub_weight/__HDelta1__)
        {
          const float nom = __HDelta1__ / hub_weight;//delta * A^{-1/2}
          const float denom =  nom  / ( 2.f * hub_weight * hub_weight ); //delta / 2 * A^{-3/2}

          //(df/dxk - df/dxl) * 1/sqrt(A) * wkl
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x])  * (wx_0 * nom);

          //////////////////////////////////////////////////////////////
          // delta * A^{-3/2} * (y_k-y_l) * w_kl : ok
          const float temp = -denom * 2.0f * (yk - y_prev[threadIdx.y + 1][threadIdx.x] ) * wx_0;
          //  (df/dxk - df/dxl) * w_lk * (y_k-y_l) * temp : ok
          df_dy_t -=  (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x]) * wx_0 * (yk  - y_prev[threadIdx.y + 1][threadIdx.x]) * temp;

          //  (df/dxb - df/dxl) * w_lb * (y_b-y_l) * temp : ok
          df_dy_t -=  (dx05_prev[threadIdx.y + 2][threadIdx.x] - dx05_prev[threadIdx.y + 1][threadIdx.x]) * wy_2 * (y_prev[threadIdx.y + 2][threadIdx.x] - y_prev[threadIdx.y + 1][threadIdx.x]) * temp;

          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i+1,j - i,j) * y(i+1,j  - i,j  ) * w_(i,j / i+1,j) * temp
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i,j+1 - i,j) * y(i,j+1  - i,j  ) * w_(i,j / i,j+1) * temp
          //////////////////////////////////////////////////////////////
        }
        else // standard case ..
        {
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x])  * wx_0;
        }
      }
// those 2 are special cases .. i have w as 0 if not present .. thus always 0 time something is 0
      {
        const float wx_nyx2_T0 = (yk - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * (yk - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;
        const float wy_nyy2_T0 = (yk - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * (yk - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;

        const float hub_weight = sqrtf( wx_nyx2_T0 + wy_nyy2_T0 );
        if ( hub_weight > __HDelta1__ )
        {
          const float nom = __HDelta1__ / hub_weight;// 1/sqrt(A)
          const float denom =  nom  / ( 2.f * hub_weight * hub_weight ); //1/2 * A^{-3/2}

          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * (wy_1 * nom); //(df/dxk - df/dxv) * 1/sqrt(A) * wv
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * (wx_1 * nom); //(df/dxk - df/dxh) * 1/sqrt(A) * wh

          //////////////////////////////////////////////////////////////
          // temp is the derivative of the denominator wrt to yk, whose derivative we are computing
          const float temp = denom * 2.0f * ( (y_prev[threadIdx.y + 2][threadIdx.x + 1] - yk) * wy_1 + (y_prev[threadIdx.y + 1][threadIdx.x + 2] - yk) * wx_1 );
          // A^{-3/2} * [ (y_v-y_k) * wv + (y_h-y_k) * wh ]

          // (df/dxk - df/dxv) * w_v * (y_k-y_v) * temp
          df_dy_t -=  (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1 * (yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * temp;
          // (df/dxk- df/dxh)  * w_h * (y_k-y_h) * temp
          df_dy_t -=  (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1 * (yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * temp;
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i+1,j - i,j) * y(i+1,j  - i,j  ) * w_(i,j / i+1,j) * temp
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i,j+1 - i,j) * y(i,j+1  - i,j  ) * w_(i,j / i,j+1) * temp
          //////////////////////////////////////////////////////////////

          // those lines  cover all others due to access to yk neighbors ..
          if (w < bv.W - 1) // - (df/dxk - df/dxh) * (yk-yh) * ( nom - wx_nyx2_T0      * denom )
            // - (df/dxk - df/dxv) *              (yk-yh)*(yk-yh)*wy_1 * (yk-yv) * denom
            d_gwx(n, 0, h,   w  ) = d_gwx(n, 0, h,   w  ) - localGXMult * ( yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2] ) *
                                    ( ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] ) * ( nom - wx_nyx2_T0 * denom ) -
                                      ( df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1] ) * ( yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2] ) * ( yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1] ) * denom * wy_1 );

          if (h < bv.H - 1) // - (df/dxk - df/dxv) * (yk-yv) * ( nom  - wy_nyy2_T0     * denom )
            // - (df/dxk - df/dxh) *              (yk-yv)*(yk-yv)*wx_1 * (yk-yh) * denom
            d_gwy(n, 0, h,   w  ) = d_gwy(n, 0, h,   w  ) - localGXMult * ( yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1] ) *
                                    ( ( df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1] )  * ( nom - wy_nyy2_T0 * denom ) -
                                      ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] )  * ( yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1] ) * ( yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2] ) * denom * wx_1 );
        }
        else
        {
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;
          // note w-1 is not called from w==bv.W [the above if w>0 addon]; so this should suffice
          if (w < bv.W - 1)
            d_gwx(n, 0, h,   w  ) = d_gwx(n, 0, h,   w  ) - localGXMult * (yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]);
          if (h < bv.H - 1)
            d_gwy(n, 0, h,   w  ) = d_gwy(n, 0, h,   w  ) - localGXMult * (yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]);
        }
      }
/////////////////////////
#else // no huber on it -> a lot simpler .. also no need for yk as input .. 

      if ( h > 0 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
      if ( h < bv.H - 1 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;
      if ( w > 0 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x]) * wx_0;
      if ( w < bv.W - 1 )
        df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;

      if (w > 0)
        atomicAdd( &(d_gwx(0, 0, h,   w - 1)), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 1][threadIdx.x  ]) * yk );
      if (w < bv.W - 1)
        atomicAdd( &(d_gwx(0, 0, h,   w  )), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * yk );
      if (h > 0)
        atomicAdd( &(d_gwy(0, 0, h - 1, w)  ), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y  ][threadIdx.x + 1]) * yk );
      if (h < bv.H - 1)
        atomicAdd( &(d_gwy(0, 0, h,   w)  ), - localGXMult * (dx05_prev[threadIdx.y + 1][threadIdx.x + 1] - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * yk );

#endif //end of huber or l2 regularization
////////////////////////////////////////////////////////////////////////
/////////////////////////////// steps 6 and 7 ///////////////////
      float df_dxk = ( d_gxk(n, ch, h, w) + (1. + t) * df_dy_t );
#ifdef clipLocalGrad
      df_dxk = max( -clipLocalGrad, min (df_dxk, clipLocalGrad) );
#endif
      d_gxk(n, ch, h, w)    = df_dxk; // TODO: WRITTEN JUST FOR DEBUGGING -> if runnig can just go .. (does not matter..)
      d_gxk_m1(n, ch, h, w) = -t * df_dy_t;
////////////////////////////////////////////////////////////////////////
// steps 2,3,4
      const float bb   = bv(n, ch, h, w);

#ifdef __Huber1__

      float diff = bb - xv_m05(it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      if ( 1. + conf >= sign * diff )
      {
        d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
        d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
        d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf) * localGBMult;
      }
      else // l1 case
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

#else // quadratic or L1 data term:

#ifdef __quadratic__
// x = (conf.*b+x05)./(1+conf); derivative is
// df/db    = df/dx * conf/(1+conf)
// df/dx05  = df/dx *  1/(1+conf)
// df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
      d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
      d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
      d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf) * localGBMult;
#else

      float diff = bb - xv_m05(it, ch, h, w);
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
#endif // quadratic
#endif // huber 
////////////////////////////////
    }
  }
};


__global__ void FistaInpaint_end_2D_tex_bw_yk( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is !! k-0.5 !! at it, at it-1 the one before that
    iu::TensorGpu_32f::TensorKernelData d_gc, iu::TensorGpu_32f::TensorKernelData d_gb,
    iu::TensorGpu_32f::TensorKernelData d_gwx, iu::TensorGpu_32f::TensorKernelData d_gwy, iu::TensorGpu_32f::TensorKernelData yv_0,
    iu::TensorGpu_32f::TensorKernelData d_gxk05_in, iu::TensorGpu_32f::TensorKernelData d_gxk05_out,
    iu::TensorGpu_32f::TensorKernelData d_gxk, iu::TensorGpu_32f::TensorKernelData d_gxk_m1,
    short it, short n = 0, short channels = 1, float t = 1, float nits = 1. )
{
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < bv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

#ifdef __HDelta1__
  __shared__ float y_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!
#endif

  __shared__ float dx05_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!

  for ( short ch = 0; ch < channels; ch++ )
  {
    const float yk   = yv_0(n, ch, h, w);
    short local_unrolled_x, local_unrolled_y, global_unrolled_x, global_unrolled_y;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // onto 2 tiling ..
    local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    if (ch > 0) __syncthreads(); // since y has to be loaded completely ..
#ifdef __HDelta1__
    y_prev   [local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv_0(n, ch, global_unrolled_y, global_unrolled_x) : 0;
#endif
    dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;
    // load 2
    if ( local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
    {
#ifdef __HDelta1__
      y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? yv_0(n, ch, global_unrolled_y, global_unrolled_x) : 0;
#endif
      dx05_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < bv.H && global_unrolled_x >= 0 && global_unrolled_x < bv.W) ? d_gxk05_in(n, ch, global_unrolled_y, global_unrolled_x) : 0;
    }
    __syncthreads(); // since y has to be loaded completely ..

////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( !(w >= bv.W || h >= bv.H) )
    {
/////////////////////////////// steps 5:      y = y + (1/L)*laplace_W*y;
      const float df_dy = dx05_prev[threadIdx.y + 1][threadIdx.x + 1];
      float df_dy_t = df_dy;

#ifdef __HDelta1__
      if ( h > 0 )
      {
        const float wx_2 = (h > 0 && w < bv.W - 1) ? tex2D<float>(dx, w + 0.5f, h - 0.5f) : 0; // *L
        const float wy_nyy2_T2 = (yk - y_prev[threadIdx.y][threadIdx.x + 1]) * (yk - y_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
        const float wx_nyx2_T2 = (y_prev[threadIdx.y][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 2]) * (y_prev[threadIdx.y][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 2]) * wx_2;

        const float hub_weight = sqrtf( wx_nyx2_T2 + wy_nyy2_T2 );
        if ( hub_weight > __HDelta1__ ) // max(1.f,hub_weight/__HDelta1__)
        {
          const float nom = __HDelta1__ / hub_weight;//delta * A^{-1/2}
          const float denom =  nom  / ( 2.f * hub_weight * hub_weight ); //delta/2 * A^{-3/2}

          //(df/dxk - df/dxu) * 1/sqrt(A) * w_uk: ok
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * (wy_0 * nom);

          //////////////////////////////////////////////////////////////
          // delta * A^{-3/2} * (y_k-y_u) * w_ku : ok
          const float temp = -denom * 2.0f * (yk  - y_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
          // (df/dxk - df/dxu) * w_ku * (y_k-y_u) * temp : ok
          df_dy_t -=  (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wy_0 * (yk  - y_prev[threadIdx.y][threadIdx.x + 1]) * temp;
          // (df/dxt - df/dxu) * w_tu * (y_t-y_u) * temp : ok
          df_dy_t -=  (dx05_prev[threadIdx.y][threadIdx.x + 2] - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wx_2 * (y_prev[threadIdx.y][threadIdx.x + 2]  - y_prev[threadIdx.y][threadIdx.x + 1]) * temp;
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i+1,j - i,j) * y(i+1,j  - i,j  ) * w_(i,j / i+1,j) * temp
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i,j+1 - i,j) * y(i,j+1  - i,j  ) * w_(i,j / i,j+1) * temp
          //////////////////////////////////////////////////////////////
        }
        else
        {
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
        }
      }
      if ( w > 0 )
      {
        const float wy_2 = (w > 0 && h < bv.H - 1) ? tex2D<float>(dy, w - 0.5f, h + 0.5f) : 0;
        const float wx_nyx2_T1 = (yk - y_prev[threadIdx.y + 1][threadIdx.x]) * (yk - y_prev[threadIdx.y + 1][threadIdx.x]) * wx_0;
        const float wy_nyy2_T1 = (y_prev[threadIdx.y + 1][threadIdx.x] - y_prev[threadIdx.y + 2][threadIdx.x]) * (y_prev[threadIdx.y + 1][threadIdx.x] - y_prev[threadIdx.y + 2][threadIdx.x]) * wy_2;

        const float hub_weight = sqrtf( wx_nyx2_T1 + wy_nyy2_T1 );

        if ( hub_weight > __HDelta1__ ) // max(1.f,hub_weight/__HDelta1__)
        {
          const float nom = __HDelta1__ / hub_weight;//delta * A^{-1/2}
          const float denom =  nom  / ( 2.f * hub_weight * hub_weight ); //delta / 2 * A^{-3/2}

          //(df/dxk - df/dxl) * 1/sqrt(A) * wkl
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x])  * (wx_0 * nom);

          //////////////////////////////////////////////////////////////
          // delta * A^{-3/2} * (y_k-y_l) * w_kl : ok
          const float temp = -denom * 2.0f * (yk - y_prev[threadIdx.y + 1][threadIdx.x] ) * wx_0;
          //  (df/dxk - df/dxl) * w_lk * (y_k-y_l) * temp : ok
          df_dy_t -=  (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x]) * wx_0 * (yk  - y_prev[threadIdx.y + 1][threadIdx.x]) * temp;

          //  (df/dxb - df/dxl) * w_lb * (y_b-y_l) * temp : ok
          df_dy_t -=  (dx05_prev[threadIdx.y + 2][threadIdx.x] - dx05_prev[threadIdx.y + 1][threadIdx.x]) * wy_2 * (y_prev[threadIdx.y + 2][threadIdx.x] - y_prev[threadIdx.y + 1][threadIdx.x]) * temp;
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i+1,j - i,j) * y(i+1,j  - i,j  ) * w_(i,j / i+1,j) * temp
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i,j+1 - i,j) * y(i,j+1  - i,j  ) * w_(i,j / i,j+1) * temp
          //////////////////////////////////////////////////////////////
        }
        else // standard case ..
        {
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x])  * wx_0;
        }
      }
// those 2 are special cases ..
      {
        const float wx_nyx2_T0 = (yk - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * (yk - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;
        const float wy_nyy2_T0 = (yk - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * (yk - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;

        const float hub_weight = sqrtf( wx_nyx2_T0 + wy_nyy2_T0 );
        if ( hub_weight > __HDelta1__ )
        {
          const float nom = __HDelta1__ / hub_weight;// 1/sqrt(A)
          const float denom =  nom  / ( 2.f * hub_weight * hub_weight ); //1/2 * A^{-3/2}

          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * (wy_1 * nom); //(df/dxk - df/dxv) * 1/sqrt(A) * wv
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * (wx_1 * nom); //(df/dxk - df/dxh) * 1/sqrt(A) * wh

          //////////////////////////////////////////////////////////////
          // temp is the derivative of the denominator wrt to yk, whose derivative we are computing
          const float temp = denom * 2.0f * ( (y_prev[threadIdx.y + 2][threadIdx.x + 1] - yk) * wy_1 + (y_prev[threadIdx.y + 1][threadIdx.x + 2] - yk) * wx_1 );
          // A^{-3/2} * [ (y_v-y_k) * wv + (y_h-y_k) * wh ]

          // (df/dxk - df/dxv) * w_v * (y_k-y_v) * temp
          df_dy_t -=  (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1 * (yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * temp;
          // (df/dxk- df/dxh)  * w_h * (y_k-y_h) * temp
          df_dy_t -=  (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1 * (yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * temp;
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i+1,j - i,j) * y(i+1,j  - i,j  ) * w_(i,j / i+1,j) * temp
          /// pattern appears to in triangle T(i,j/ i+1,j / i,j+1) : be df/dx^0.5 (i,j+1 - i,j) * y(i,j+1  - i,j  ) * w_(i,j / i,j+1) * temp
          //////////////////////////////////////////////////////////////

          // those lines should cover all others due to access to yk neighbors ..
          // note w-1 is not called from w==bv.W [the above if w>0 addon]; so this should suffice : wx_nyx2_T0 = wx_1 * (yk-yh)*(yk-yh)
          if (w < bv.W - 1) // - (df/dxk - df/dxh) * (yk-yh) * ( nom - wx_nyx2_T0      * denom )
            // - (df/dxk - df/dxv) *              (yk-yh)*(yk-yh)*wy_1 * (yk-yv) * denom
            d_gwx(n, 0, h,   w  ) = d_gwx(n, 0, h,   w  ) - localGXMult * ( ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] ) * ( yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2] ) * ( nom - wx_nyx2_T0 * denom ) -
                                    ( df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1] ) * ( yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2] ) * ( yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2] ) * ( yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1] ) * denom * wy_1 );
          if (h < bv.H - 1) // - (df/dxk - df/dxv) * (yk-yv) * ( nom  - wy_nyy2_T0     * denom )
            // - (df/dxk - df/dxh) *              (yk-yv)*(yk-yv)*wx_1 * (yk-yh) * denom
            d_gwy(n, 0, h,   w  ) = d_gwy(n, 0, h,   w  ) - localGXMult * ( ( df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1] ) * ( yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1] ) * ( nom - wy_nyy2_T0 * denom ) -
                                    ( df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2] ) * ( yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1] ) * ( yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1] ) * ( yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2] ) * denom * wx_1 );
        }
        else
        {
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;
          df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;
          // note w-1 is not called from w==bv.W [the above if w>0 addon]; so this should suffice
          if (w < bv.W - 1)
            d_gwx(n, 0, h,   w  ) = d_gwx(n, 0, h,   w  ) - localGXMult * (yk  - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]);
          if (h < bv.H - 1)
            d_gwy(n, 0, h,   w  ) = d_gwy(n, 0, h,   w  ) - localGXMult * (yk  - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]);
        }
      }
/////////////////////////

#else // no huber on it -> a lot simpler .. a lot .. !!! also no need for yk as input .. etc .. sigh .. 

      df_dy_t -= (df_dy - dx05_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
      df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;
      df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x]) * wx_0;
      df_dy_t -= (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;

// todo only write for ch=0 above for loop only for ch=0 ..
      if (w > 0) // df/dxk - df/dxl * yk
        atomicAdd( &(d_gwx(n, 0, h,   w - 1)), - localGXMult * (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x  ]) * yk );
      if (w < bv.W - 1) // df/dxk - df/dxh * yk
        atomicAdd( &(d_gwx(n, 0, h,   w  )), - localGXMult * (df_dy - dx05_prev[threadIdx.y + 1][threadIdx.x + 2]) * yk ); //
      if (h > 0)     //  df/dxk - df/dxu * yk
        atomicAdd( &(d_gwy(n, 0, h - 1, w)  ), - localGXMult * (df_dy - dx05_prev[threadIdx.y  ][threadIdx.x + 1]) * yk ); //
      if (h < bv.H - 1) // df/dxk - df/dxv * yk
        atomicAdd( &(d_gwy(n, 0, h,   w)  ), - localGXMult * (df_dy - dx05_prev[threadIdx.y + 2][threadIdx.x + 1]) * yk ); //

#endif // case of Huber on regularizer .. or not .. 

////////////////////////////////////////////////////////////////////////
/////////////////////////////// steps 6 and 7 ///////////////////
      float df_dxk = d_gxk(n, ch, h, w) + (1. + t) * df_dy_t;
#ifdef clipLocalGrad
      df_dxk = max( -clipLocalGrad, min (df_dxk, clipLocalGrad) ); // sole grads go nuts .. still no idea why but this happens ..
#endif
      d_gxk(n, ch, h, w)    = df_dxk;
      d_gxk_m1(n, ch, h, w) = -t * df_dy_t;
////////////////////////////////////////////////////////////////////////
// steps 2,3,4
      const float bb   = bv(n, ch, h, w);

#ifdef __Huber1__

      const float diff = bb - xv_m05(it, ch, h, w);
      const float sign = (diff < 0) ? -1. : 1.;
      if ( 1. + conf >= sign * diff )
      {
// quadratic case EXACTLY:
// x = (conf.*b+x05)./(1+conf); derivative is
// df/dconf = df/dx *  b./(1+conf) - conf*b/*(1+conf*conf)
// df/db    = df/dx * conf/(1+conf)
// df/dy    = df/dx *  1/(1+conf)
// df/dconf = df/dx *  b./(1+conf) - (conf*b+xv_m05(it, ch, h, w) )/*((1+conf)*(1+conf))
        d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
        d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
        d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf)  * localGBMult;
      }
      else // L1 case exactly
      {
        if ( conf >= sign * diff )
        {
          d_gxk05_out(n, ch, h, w) = 0; // set to 0 always
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
      d_gc(n, 0, h, w)         += df_dxk * ( bb / (1.f + conf)  - ( conf * bb + xv_m05(it, ch, h, w) ) / ((1.f + conf) * (1.f + conf)) ) * localGCMult;
      d_gxk05_out(n, ch, h, w)  = df_dxk / (1.f + conf);
      d_gb(n, ch, h, w)        += df_dxk * conf / (1.f + conf)  * localGBMult;
#else
      const float diff = bb - xv_m05(it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      if ( conf >= sign * diff )
      {
        d_gxk05_out(n, ch, h, w) = 0; // set to 0 always
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
__global__ void FistaInpaint_simple_get_yk(
  cudaTextureObject_t cf,
  iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv_m05, // xv_m05 is k-0.5; at iteration it, at it-1 the one before that
  iu::TensorGpu_32f::TensorKernelData yv_out_k, short it, short n = 0, short channels = 1, float t = 1 )
{

  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  for ( short ch = 0; ch < channels; ch++ )
  {
    if ( !(w >= bv.W || h >= bv.H) )
    {
////////////////////////////////////////////////////////////////////////
// steps 2,3,4
      const float bb   = bv(n, ch, h, w);

#ifdef __Huber1__

      float diff = bb - xv_m05(it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;
      float yk(0);
      if ( 1. + conf >= sign * diff )
      {
        const float xk  = (conf * bb + xv_m05(it,   ch, h, w)) / (1.f + conf);
        const float xkm = (conf * bb + xv_m05(it - 1, ch, h, w)) / (1.f + conf);
        yk  = xk + t * (xk - xkm);
      }
      else // l1 case
      {
        const float xk   = ( conf <= sign * diff   ) ? bb - diff + sign * conf : bb;
        diff = bb - xv_m05(it - 1, ch, h, w);  sign = (diff < 0) ? -1. : 1.;
        yk   = xk + t * (xk - (( conf <= sign * diff   ) ? bb - diff + sign * conf : bb) );
      }

#else // not huber case any more /////////////////////////////////

#ifdef __quadratic__
      const float xk  = (conf * bb + xv_m05(it,   ch, h, w)) / (1.f + conf);
      const float xkm = (conf * bb + xv_m05(it - 1, ch, h, w)) / (1.f + conf);
      const float yk  = xk + t * (xk - xkm);
#else

      float diff = bb - xv_m05(it, ch, h, w);
      float sign = (diff < 0) ? -1. : 1.;

      const float xk   = ( conf <= sign * diff   ) ? bb - diff + sign * conf : bb;
      diff = bb - xv_m05(it - 1, ch, h, w);  sign = (diff < 0) ? -1. : 1.;
      const float yk   = xk + t * (xk - (( conf <= sign * diff   ) ? bb - diff + sign * conf : bb) );

#endif // quadratic
#endif // huber 

      yv_out_k(n, ch, h, w) = yk;

    }
  }
};

///////////////////////////////////////////////

// forward kernel 1
__global__ void FistaInpaint_simple_nCalls_2D_tex_clean( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv, iu::TensorGpu_32f::TensorKernelData yv, iu::TensorGpu_32f::TensorKernelData yv_out,
    short n = 0, short channels = 1, float t = 1)
{

  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < xv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < xv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

#ifdef __HDelta1__
  const float wx_2 = (h > 0 && w < xv.W - 1) ? tex2D<float>(dx, w + 0.5f, h - 0.5f) : 0; // *L
  const float wy_2 = (w > 0 && h < xv.H - 1) ? tex2D<float>(dy, w - 0.5f, h + 0.5f) : 0; // stored as wx_0
#endif
// this one holds y of previous iteration of the current block of ids PLUS one more to the right (forward differences)
// huber needs to adjust the weights according to the |sqrt(w) nabla x|_2 norm ..

  __shared__ float y_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!

  for (short ch = 0; ch < channels; ch++)
  {
    // onto 2 tiling ..
    short local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    short local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    short global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    short global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // read local_unrolled_x + blockDim.x * blockIdx.x, local_unrolled_y + blockDim.y * blockIdx.y -(1,1)
    // write to local id: local_unrolled_y, local_unrolled_x
    if (ch > 0) __syncthreads(); // since y has to be loaded completely ..
    y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;
    // load 2
    if (local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
      y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    __syncthreads(); // since y has to be loaded completely ..

    if ( !(w >= xv.W || h >= xv.H) )
    {
      const float x_   = xv(n, ch, h, w);
      const float b    = bv(n, ch, h, w); // as in the right hand side ..
      float x_loc      = b;
      //     y = y + (1/L)*laplace_W*y;
      float yc = y_prev[threadIdx.y + 1][threadIdx.x + 1];
      float x05 = yc;
#ifdef __HDelta1__
      if ( h > 0 )
      {
        // 1st compute |sqrt(W)|_delta .. could use a sqrt kernel on the weights .. or
        // use ||\sqrt{W} \nabla y|| = \sqrt{W_x(\nablay)^2_x + W_y(\nablay)^2_y}
        // W_x(\nablay)^2_x + W_y(\nablay)^2_y < delta^2
        // weight is then W * delta/(max(delta, sqrt(W_x(\nablay)^2_x + W_y(\nablay)^2_y < delta^2) ))
        float hub_weight = sqrtf( (yc - y_prev[threadIdx.y][threadIdx.x + 1]) * (yc - y_prev[threadIdx.y][threadIdx.x + 1]) * wy_0 +
                                  (y_prev[threadIdx.y][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 2]) * (y_prev[threadIdx.y][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 2]) * wx_2 );
        x05 -= (yc - y_prev[threadIdx.y][threadIdx.x + 1]) * (wy_0 / max(1.f, hub_weight / __HDelta1__));
      }
      if ( w > 0 )
      {
        float hub_weight = sqrtf( (yc - y_prev[threadIdx.y + 1][threadIdx.x]) * (yc - y_prev[threadIdx.y + 1][threadIdx.x]) * wx_0 +
                                  (y_prev[threadIdx.y + 1][threadIdx.x] - y_prev[threadIdx.y + 2][threadIdx.x]) * (y_prev[threadIdx.y + 1][threadIdx.x] - y_prev[threadIdx.y + 2][threadIdx.x]) * wy_2 );

        x05 -= (yc - y_prev[threadIdx.y + 1][threadIdx.x]) * (wx_0 / max(1.f, hub_weight / __HDelta1__));
      }
      // those 2 are special cases..
      {
        float hub_weight = sqrtf( (yc - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * (yc - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1 +
                                  (yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * (yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1 );

        x05 -= (yc - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * (wy_1 / max(1.f, hub_weight / __HDelta1__));
        x05 -= (yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * (wx_1 / max(1.f, hub_weight / __HDelta1__));
      }
#else
      if ( h > 0 )
        x05 -= (yc - y_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
      if ( h < xv.H - 1 )
        x05 -= (yc - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;
      if ( w > 0 )
        x05 -= (yc - y_prev[threadIdx.y + 1][threadIdx.x]) * wx_0;
      if ( w < xv.W - 1 )
        x05 -= (yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;
#endif

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
      yc  = x05 - conf;
      x05 = x05 + conf;
      if (yc > b)  x_loc = yc;
      if (x05 < b) x_loc = x05;

#endif

#endif // huber or not 
      //     t_=t
      //     t=(1+sqrt(1+4*t_^2))/2;
      //     y = x + (t_-1)/t * (x-x_); % that is y over-relaxed
      yv_out(0, ch, h, w) = x_loc + t * (x_loc - x_); //
      xv(n, ch, h, w) = x_loc;
    }
  }
}

__global__ void FistaInpaint_simple_nCalls_2D_tex_clean_writexk05( cudaTextureObject_t dx, cudaTextureObject_t dy, cudaTextureObject_t cf,
    iu::TensorGpu_32f::TensorKernelData bv, iu::TensorGpu_32f::TensorKernelData xv,
    iu::TensorGpu_32f::TensorKernelData yv, iu::TensorGpu_32f::TensorKernelData yv_out,
    iu::TensorGpu_32f::TensorKernelData xv_k05_out, short it, short n = 0, short channels = 1, float t = 1)
{ //2*it+1, samples,       ingrad_c,      step
  short w = blockIdx.x * blockDim.x + threadIdx.x;
  short h = blockIdx.y * blockDim.y + threadIdx.y;

  const float conf = tex2D<float>( cf, w + 0.5, h + 0.5); // * L;

  const float wx_0 = (w > 0 )     ? tex2D<float>(dx, w - 0.5f, h + 0.5f) : 0;// *L
  const float wx_1 = (w < xv.W - 1) ? tex2D<float>(dx, w + 0.5f, h + 0.5f) : 0;
  const float wy_0 = (h > 0 )     ? tex2D<float>(dy, w + 0.5f, h - 0.5f) : 0;
  const float wy_1 = (h < xv.H - 1) ? tex2D<float>(dy, w + 0.5f, h + 0.5f) : 0;

#ifdef __HDelta1__
  const float wx_2 = (h > 0 && w < xv.W - 1) ? tex2D<float>(dx, w + 0.5f, h - 0.5f) : 0; // *L
  const float wy_2 = (w > 0 && h < xv.H - 1) ? tex2D<float>(dy, w - 0.5f, h + 0.5f) : 0;
#endif

  __shared__ float y_prev[COMMON_BLOCK_SIZE_2D_Y + 2][COMMON_BLOCK_SIZE_2D_X + 2]; // max KB was 48 for a kernel !!!

  for (short ch = 0; ch < channels; ch++)
  {
    // onto 2 tiling ..
    short local_unrolled_x  = (blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    short local_unrolled_y  = (blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    short global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    short global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;

    // read local_unrolled_x + blockDim.x * blockIdx.x, local_unrolled_y + blockDim.y * blockIdx.y -(1,1)
    // write to local id: local_unrolled_y, local_unrolled_x
    if (ch > 0) __syncthreads(); // since y has to be loaded completely ..

    y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    local_unrolled_x  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) % (COMMON_BLOCK_SIZE_2D_X + 2);
    local_unrolled_y  = ( blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x) / (COMMON_BLOCK_SIZE_2D_X + 2);
    global_unrolled_x = local_unrolled_x + blockIdx.x * blockDim.x - 1;
    global_unrolled_y = local_unrolled_y + blockIdx.y * blockDim.y - 1;
    // load 2
    if (local_unrolled_x < (COMMON_BLOCK_SIZE_2D_X + 2) && local_unrolled_y < (COMMON_BLOCK_SIZE_2D_Y + 2) )
      y_prev[local_unrolled_y][local_unrolled_x]   = (global_unrolled_y >= 0 && global_unrolled_y < xv.H && global_unrolled_x >= 0 && global_unrolled_x < xv.W) ? yv(n, ch, global_unrolled_y, global_unrolled_x) : 0;

    __syncthreads(); // since y has to be loaded completely ..

    if ( !(w >= xv.W || h >= xv.H) )
    {
      const float x_   = xv(n, ch, h, w);
      const float b    = bv(n, ch, h, w); // as in the right hand side ..
      float x_loc      = b;
      //     y = y + (1/L)*laplace_W*y;
      float yc = y_prev[threadIdx.y + 1][threadIdx.x + 1];
      float yt = yc;

#ifdef __HDelta1__
      if ( h > 0 ) // w < xv.W-1 ) FUCK ..
      {
        // 1st compute |sqrt(W)|_delta .. could use a sqrt kernel on the weights .. or
        // use ||\sqrt{W} \nabla y|| = \sqrt{W_x(\nablay)^2_x + W_y(\nablay)^2_y}
        // W_x(\nablay)^2_x + W_y(\nablay)^2_y < delta^2
        // weight is then W * delta/(max(delta, sqrt(W_x(\nablay)^2_x + W_y(\nablay)^2_y < delta^2) ))
        float hub_weight = sqrtf( (yc - y_prev[threadIdx.y][threadIdx.x + 1]) * (yc - y_prev[threadIdx.y][threadIdx.x + 1]) * wy_0 +
                                  (y_prev[threadIdx.y][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 2]) * (y_prev[threadIdx.y][threadIdx.x + 1] - y_prev[threadIdx.y][threadIdx.x + 2]) * wx_2 );
        yt -= (yc - y_prev[threadIdx.y][threadIdx.x + 1]) * (wy_0 / max(1.f, hub_weight / __HDelta1__));
      }
      if ( w > 0 )
      {
        float hub_weight = sqrtf( (yc - y_prev[threadIdx.y + 1][threadIdx.x]) * (yc - y_prev[threadIdx.y + 1][threadIdx.x]) * wx_0 +
                                  (y_prev[threadIdx.y + 1][threadIdx.x] - y_prev[threadIdx.y + 2][threadIdx.x]) * (y_prev[threadIdx.y + 1][threadIdx.x] - y_prev[threadIdx.y + 2][threadIdx.x]) * wy_2 );

        yt -= (yc - y_prev[threadIdx.y + 1][threadIdx.x]) * (wx_0 / max(1.f, hub_weight / __HDelta1__));
      }
      // those 2 are special cases ..
      {
        float hub_weight = sqrtf( (yc - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * (yc - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1 +
                                  (yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * (yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1 );

        yt -= (yc - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * (wy_1 / max(1.f, hub_weight / __HDelta1__));
        yt -= (yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * (wx_1 / max(1.f, hub_weight / __HDelta1__));
      }
#else
      if ( h > 0 )
        yt -= (yc - y_prev[threadIdx.y][threadIdx.x + 1]) * wy_0;
      if ( h < xv.H - 1 )
        yt -= (yc - y_prev[threadIdx.y + 2][threadIdx.x + 1]) * wy_1;
      if ( w > 0 )
        yt -= (yc - y_prev[threadIdx.y + 1][threadIdx.x]) * wx_0;
      if ( w < xv.W - 1 )
        yt -= (yc - y_prev[threadIdx.y + 1][threadIdx.x + 2]) * wx_1;

#endif

      xv_k05_out(it, ch, h, w) = yt;

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
      //     t_=t
      //     t=(1+sqrt(1+4*t_^2))/2;
      //     y = x + (t_-1)/t * (x-x_); % that is y over-relaxed
      yv_out(0, ch, h, w) = x_loc + t * (x_loc - x_); //
      xv(n, ch, h, w) = x_loc;
    }
  }
}

// dx,dy: tensors/edge weights in x and y direction. d_c: confidence, d_b rhs, eg. u_hat;
// d_i: initial solution used at iteration 0. simplest is d_b or solution from lower resolution
// d_out: output. ic: channels(# of rhs/channels of b), iw: width ih: height of inputs
// its: iterations to run.
// id: we can run multiple inpainters, eg. for different resolutions. To identify the buffers, etc.
// we need an id, that is specified by the user. Its a number between 0 and 9 so far..
int TVInpaintFista::forward( float *d_x, float *d_y, float *d_c, float *d_b, float *d_i, float *d_out,
                             int ic, int ih, int iw, int its, int id )
{
  const int in = 1;

  if (static_cast<size_t> (id) >= TVInpaintFista::id.size())
  {
    std::cerr << "Cannot use an id that is negative or larger/equal than the maximal one:" << id <<"/" << TVInpaintFista::id.size() << "\n";
    return 1;
  }

  iu::TensorGpu_32f d_db(d_b,     in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_dout(d_out, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_din(d_i,    in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  //thrust::fill(d_dout.begin(), d_dout.end(), 0.0f);
  double L =  1. / 8.; // 1/Lipshitz for the TV case ..

  iu::TensorGpu_32f y0_temp( 1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x, even iterations
  iu::TensorGpu_32f y1_temp( 1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x, odd  iterations

  iu::ImageGpu_32f_C1 tex_dx(d_x, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_dy(d_y, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_cf(d_c, iw, ih, sizeof(float) * iw, false);

  using namespace thrust::placeholders;
  thrust::transform(tex_dx.begin(), tex_dx.end(), tex_dx.begin(), L * _1);
  thrust::transform(tex_dy.begin(), tex_dy.end(), tex_dy.begin(), L * _1);

  dim3 dimBlock(COMMON_BLOCK_SIZE_2D_X, COMMON_BLOCK_SIZE_2D_Y);
  dim3 dimGrid( std::ceil( ( d_db.width()   ) / static_cast<float>(dimBlock.x)) ,
                std::ceil( ( d_db.height()  ) / static_cast<float>(dimBlock.y)) );

  cudaMemcpy(   d_dout.data(), d_din.data(), in * ic * ih * iw * sizeof(float),  cudaMemcpyDeviceToDevice); // init value ..
  cudaMemcpy(  y0_temp.data(), d_din.data(), in * ic * ih * iw * sizeof(float),  cudaMemcpyDeviceToDevice);
  cudaMemcpy(  y1_temp.data(), d_din.data(), in * ic * ih * iw * sizeof(float),  cudaMemcpyDeviceToDevice);

  TVInpaintFista::id[id].iterations.resize( std::floor( sqrt( its ) ), 0);
  TVInpaintFista::id[id].maxInterIts = 0;

  for (int i = 0; i < TVInpaintFista::id[id].iterations.size(); i++)
  {
    TVInpaintFista::id[id].iterations[i] = std::floor(i * sqrt(its) * 0.5) * 2;
    if (i > 0)
    {
      TVInpaintFista::id[id].maxInterIts = std::max(TVInpaintFista::id[id].maxInterIts, TVInpaintFista::id[id].iterations[i] - TVInpaintFista::id[id].iterations[i - 1] );
    }

    if (TVInpaintFista::id[id].intermediate_y[ i ])
    {
      free( TVInpaintFista::id[id].intermediate_y[ i ] );
      TVInpaintFista::id[id].intermediate_y[ i ] = NULL;
    }
    TVInpaintFista::id[id].intermediate_y[ i ] = (float*) calloc( in * ic * ih * iw, sizeof(float) );
    if (TVInpaintFista::id[id].intermediate_x[ i ])
    {
      free( TVInpaintFista::id[id].intermediate_x[ i ] );
      TVInpaintFista::id[id].intermediate_x[ i ] = NULL;
    }
    TVInpaintFista::id[id].intermediate_x[ i ] = (float*) calloc( in * ic * ih * iw, sizeof(float) );
  }
  TVInpaintFista::id[id].maxInterIts = std::max(TVInpaintFista::id[id].maxInterIts, its - TVInpaintFista::id[id].iterations[TVInpaintFista::id[id].iterations.size() - 1] );

  TVInpaintFista::id[id].stepSizes.resize( its, 0);
  double t(1), t_(1);
  for (int i = 0; i < TVInpaintFista::id[id].stepSizes.size(); i++)
  {
    t_ = t;
    t  = (1. + sqrt(1. + 4.*t * t)) / 2.;
    TVInpaintFista::id[id].stepSizes[i] = std::max( 0.0, (t_ - 1.) / t);
  }

  int start_it = 0;
  for (int samples = 0; samples < in; samples++)
  {
    float step(1);

    for ( int it = 0; it < its / 2; it++ )
    {
      if ( start_it < TVInpaintFista::id[id].iterations.size() && TVInpaintFista::id[id].iterations[start_it] == 2 * it && TVInpaintFista::id[id].intermediate_y[ start_it ] != NULL )
      {
        cudaMemcpy( TVInpaintFista::id[id].intermediate_y[ start_it ], y0_temp.data(), in * ic * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
        cudaMemcpy( TVInpaintFista::id[id].intermediate_x[ start_it ],  d_dout.data(), in * ic * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
        start_it++;
      }

      //t_ = t;
      //t  = (1.+sqrt(1.+4.*t*t))/2.; // precomputed
      step = TVInpaintFista::id[id].stepSizes[2 * it];
      FistaInpaint_simple_nCalls_2D_tex_clean <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), d_db, d_dout, y0_temp, y1_temp, samples, ic, step );
      //t_ = t;
      //t  = (1.+sqrt(1.+4.*t*t))/2.       ; // precomputed
      step = TVInpaintFista::id[id].stepSizes[2 * it + 1];
      FistaInpaint_simple_nCalls_2D_tex_clean <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), d_db, d_dout, y1_temp, y0_temp, samples, ic, step );
    }
  }
  return 0;
}


// dx,dy: tensors/edge weights in x and y direction. d_c: confidence, d_b rhs, eg. u_hat;
// d_inGrad: input gradient, ie. we compute f(wx,wy,c,b,i0) in the layer and then its dLoss/df
// d_outGradX, d_outGradY, d_outGradC, d_outGradB, d_outGradI, gradients of our function wrt its input:
// X/Y: edge weights, C: confidence, b: rhs, I: initial solution for f(..)
// ic: how many input channels has our rhs b, iw: width ih: height of inputs 
// its: iterations to run.
// id: we can run multiple inpainters, eg. for different resolutions. To idnetify the buffers, etc.
// we need an id, that is specified by the user. It is a number between 0 and 9 so far..
int TVInpaintFista::backward( float *d_x, float *d_y, float *d_c, float *d_b,
                              float *d_inGrad, float *d_outGradX, float *d_outGradY, float *d_outGradC, float *d_outGradB, float *d_outGradI,
                              int ic, int ih, int iw, int its, int id )
{
  const int cc = 1;
  const int in = 1;
  const int tc = 1;
  const int samples = 0;  
  int lastIt = its;  // loop over y's; 1st create all x in between,

  if (static_cast<size_t> (id) >= TVInpaintFista::id.size())
  {
    std::cerr << "Cannot use an id that is negative or larger than the maximal one:" << id <<"/" << TVInpaintFista::id.size() << "\n";
    return 1;
  }
  //std::cout << "sizes: " << in <<" " << ic << " " << ih << " " << iw << " " << ingrad_c << " " << its << "\n";

  const double L =  1. / 8.;
  iu::ImageGpu_32f_C1 tex_dx(d_x, iw, ih, sizeof(float) * iw, false); // bool ext_data_pointer=true
  iu::ImageGpu_32f_C1 tex_dy(d_y, iw, ih, sizeof(float) * iw, false);
  iu::ImageGpu_32f_C1 tex_cf(d_c, iw, ih, sizeof(float) * iw, false);
  using namespace thrust::placeholders;
  thrust::transform(tex_dx.begin(), tex_dx.end(), tex_dx.begin(), L * _1);
  thrust::transform(tex_dy.begin(), tex_dy.end(), tex_dy.begin(), L * _1);

  iu::TensorGpu_32f d_gi(d_outGradI, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_db(d_b,        in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gx(d_outGradX, in, tc, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gy(d_outGradY, in, tc, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gc(d_outGradC, in, cc, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gb(d_outGradB, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_gin(d_inGrad,  in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);

  thrust::fill(d_gi.begin(), d_gi.end(), 0.0f);
  thrust::fill(d_gx.begin(), d_gx.end(), 0.0f);
  thrust::fill(d_gy.begin(), d_gy.end(), 0.0f);
  thrust::fill(d_gc.begin(), d_gc.end(), 0.0f);
  thrust::fill(d_gb.begin(), d_gb.end(), 0.0f);

#ifdef _NAN_Input_Check_
  std::vector<float> tempv( ic * ih * iw, 0);
  cudaMemcpy(tempv.data(), d_gin.data(), tempv.size()*sizeof(float),  cudaMemcpyDeviceToHost);
  int is_nan = 0;
  double maxinG = 0;
  for (int i = 0; i < tempv.size(); i++)
  {
    if (!is_valid(tempv[i]) ) {     is_nan = 1; tempv[i] = 0;}
    maxinG = std::max( maxinG, (double)std::abs(tempv[i]) );
  }
  if ( is_nan )
  {
    std::cerr << "Input gradient to Fista is inf or nan!\n";
    cudaMemcpy(d_gin.data(), tempv.data(), tempv.size()*sizeof(float),  cudaMemcpyHostToDevice);
  }

#ifdef _pngWriting_
  pngwriter png4(iw, ih, 0, "dIn.png");
  for (int i = 0; i < ih * iw; i++)
    png4.plot( i % iw, (ih - 1) - i / iw, std::max(0.f, std::min( -tempv[i], 1.f) ), std::max(0.f, std::min( tempv[i], 1.f) ), std::max(0.f, std::min( tempv[i + ih * iw], 1.f) ) );
  png4.close();
#endif

#endif

  iu::TensorGpu_32f y0_temp( 1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x, even iterations
  iu::TensorGpu_32f y1_temp( 1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x, odd iterations

  iu::TensorGpu_32f d_dout(         1, ic, ih, iw, iu::TensorGpu_32f::NCHW);  // will hold x variable
  iu::TensorGpu_32f d_gxk05_0_temp( 1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for dx/dx^k+0.5 .. need ping pong
  iu::TensorGpu_32f d_gxk05_1_temp( 1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for dx/dx^k+0.5 .. need ping pong
  iu::TensorGpu_32f d_gxk1_temp(    1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for dx/dx^k  ..
  iu::TensorGpu_32f d_gxk0_temp(    1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for dx/dx^k-1  .. yes i need both of them ..
  cudaMemcpy( d_gxk0_temp.data(), d_gin.data(), in * ic * ih * iw * sizeof(float), cudaMemcpyDeviceToDevice );

  thrust::fill(d_gxk1_temp.begin(), d_gxk1_temp.end(), 0.0f);
  thrust::fill(d_gxk0_temp.begin(), d_gxk0_temp.end(), 0.0f);

  thrust::fill(d_gxk05_0_temp.begin(), d_gxk05_0_temp.end(), 0.0f);
  thrust::fill(d_gxk05_1_temp.begin(), d_gxk05_1_temp.end(), 0.0f);

  iu::TensorGpu_32f d_yk_temp( 1, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for yk needed to get weights for huber on regularizer
  thrust::fill(d_yk_temp.begin(), d_yk_temp.end(), 0.0f);

  dim3 dimBlock(COMMON_BLOCK_SIZE_2D_X, COMMON_BLOCK_SIZE_2D_Y);
  dim3 dimGrid( std::ceil( ( d_db.width()   ) / static_cast<float>(dimBlock.x)) ,
                std::ceil( ( d_db.height()  ) / static_cast<float>(dimBlock.y)) );

  // backwards works like this :
  // take last y, for how many iterations: its-TVInpaintFista::id[id].iterations[]
  // generate memory for x storage: how many: lastIt-nextLastIt
  iu::TensorGpu_32f d_xk05( TVInpaintFista::id[id].maxInterIts, ic, ih, iw, iu::TensorGpu_32f::NCHW); // temporary for x+0.5
  // then i do call the kernel-Forward for k its etc ..

  std::vector<double> gc_storage( cc * ih * iw, 0 );
  std::vector<double> gx_storage( tc * ih * iw, 0 );
  std::vector<double> gy_storage( tc * ih * iw, 0 );
  std::vector<double> gb_storage( ic * ih * iw, 0 );

  float nIts(1.);
  for ( int outer = TVInpaintFista::id[id].iterations.size() - 1; outer >= 0; outer--)
  {
    int nextLastIt = TVInpaintFista::id[id].iterations[ outer ];
    // init y/x to kickstart process:
    cudaMemcpy( y0_temp.data(), TVInpaintFista::id[id].intermediate_y[ outer ], in * ic * ih * iw * sizeof(float), cudaMemcpyHostToDevice ); // write on other 1st ..
    cudaMemcpy( y1_temp.data(), TVInpaintFista::id[id].intermediate_y[ outer ], in * ic * ih * iw * sizeof(float), cudaMemcpyHostToDevice ); // write on other 1st ..
    cudaMemcpy(  d_dout.data(), TVInpaintFista::id[id].intermediate_x[ outer ], in * ic * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

    // build d_xk05 .. for the small subpart of iterations ..
    for ( int it = 0; it < (lastIt - nextLastIt) / 2; it++ )
    {
      float step = TVInpaintFista::id[id].stepSizes[ nextLastIt + 2 * it    ];
      // i need x AND y, xk and yk
      FistaInpaint_simple_nCalls_2D_tex_clean_writexk05 <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), d_db, d_dout, y0_temp, y1_temp, d_xk05, 2 * it, samples, ic, step );

      step = TVInpaintFista::id[id].stepSizes[ nextLastIt + 2 * it + 1 ];

      FistaInpaint_simple_nCalls_2D_tex_clean_writexk05 <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(), d_db, d_dout, y1_temp, y0_temp, d_xk05, 2 * it + 1, samples, ic, step );
    }
    std::vector<float> dgb(10, 0);
    //////////////////////////////////////////////
    // now parse again, backward this time, aggregating gradients.
    // need to ping pong with df/dx^(k+0.5) -> xtra kernel to init this one ..
    //
    // now can do special kernel to fill-in 1st part of df/dx05 df/dc df/db ....  df/dw?
    // then backwards kernel
    // here the d_gxk05_0_temp, d_gxk05_1_temp, d_gxk0_temp, d_gxk1_temp grow large and are differences of large numbers often
    for ( int it = (lastIt - nextLastIt) / 2 - 1; it >= 0; it-- )
    {
      float step = TVInpaintFista::id[id].stepSizes[ nextLastIt + 2 * it + 1 ];
      nIts += 1.;
      // i need x AND y, xk and yk
      // df/dy intern. extern: df/dxk, df/dxk-1, d_gxk0_temp, d_gxk1_temp
      //d_xk05 load
      if ( outer == TVInpaintFista::id[id].iterations.size() - 1  && it == ((lastIt - nextLastIt) / 2 - 1) )
        FistaInpaint_init_dc_db_dx05_dw <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(),
            d_db, d_xk05, d_gc, d_gb, d_gxk05_1_temp, d_gin, 2 * it + 1, samples, ic, step );
      else
      {
        FistaInpaint_simple_get_yk <<< dimGrid, dimBlock>>>( tex_cf.getTexture(), d_db, d_xk05, d_yk_temp, 2 * it + 1, samples, ic, step );

        FistaInpaint_simple_nCalls_2D_tex_bw_yk <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(),
            d_db, d_xk05, d_gc, d_gb, d_gx, d_gy, d_gxk05_0_temp, d_gxk05_1_temp, d_gxk0_temp, d_gxk1_temp, d_yk_temp, 2 * it + 1, samples, ic, step, nIts ); //d_dout, y0_temp, y1_temp
      }

      nIts += 1.;      step = TVInpaintFista::id[id].stepSizes[ nextLastIt + 2 * it    ];
      if (it > 0) // so 2 its this here not used still totally off ..
      {
        FistaInpaint_simple_get_yk <<< dimGrid, dimBlock>>>( tex_cf.getTexture(), d_db, d_xk05, d_yk_temp, 2 * it, samples, ic, step );

        FistaInpaint_simple_nCalls_2D_tex_bw_yk <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(),
            d_db, d_xk05, d_gc, d_gb, d_gx, d_gy, d_gxk05_1_temp, d_gxk05_0_temp, d_gxk1_temp, d_gxk0_temp, d_yk_temp, 2 * it, samples, ic, step, nIts ); //
      }
      else
      {
        cudaMemcpy( y0_temp.data(), TVInpaintFista::id[id].intermediate_y[ outer ], in * ic * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

        FistaInpaint_end_2D_tex_bw_yk <<< dimGrid, dimBlock>>>( tex_dx.getTexture(), tex_dy.getTexture(), tex_cf.getTexture(),
            d_db, d_xk05, d_gc, d_gb, d_gx, d_gy, y0_temp, d_gxk05_1_temp, d_gxk05_0_temp, d_gxk1_temp, d_gxk0_temp, 2 * it, samples, ic, step, nIts ); //
      }
    }
    copy_and_set_to_0( gc_storage, gx_storage, gy_storage, gb_storage, d_gb, d_gc, d_gx, d_gy, tc, ic, ih, iw, 1. );
    /////////////////////////////////////////
    lastIt    = nextLastIt;// 'one down'
  }
#ifdef _pngWriting_
  /*
    pngwriter png(iw,ih,0,"test.png");
    for(int i =0;i< ih* iw;i++)
      png.plot( i%iw, (ih-1) - i/iw, std::min( gx_storage[i]/1000., 1.), std::min( gy_storage[i]/1000., 1.), std::min( gc_storage[i]/1000., 1.) );
    png.close();
  */
  const double scalePNG = 1. / plotClipValue;
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
#endif

#ifdef _NAN_Output_Check_
  double run_step = std::min( _memory_T_, double(TVInpaintFista::id[id].run) / double(TVInpaintFista::id[id].run + 1) );
  double is_largest = 0; int num_above = 0; double av_gc = 0;

  check_gradUpdate( gc_storage, av_gc, is_largest, num_above, TVInpaintFista::id[id].av_GC, "C" );
  update_correct_av_step_host(TVInpaintFista::id[id].run, av_gc, TVInpaintFista::id[id].av_GC, gc_storage, run_step, "C" );
  if (TVInpaintFista::id[id].run % __debug_info_period__ == 1)
    std::cerr << "\nOutput gradient to conf  " <<  num_above << " " << av_gc << "  vs  " << TVInpaintFista::id[id].av_GC << " vs " << is_largest << " ";

  check_gradUpdate( gx_storage, av_gc, is_largest, num_above, TVInpaintFista::id[id].av_GX, "X" );
  update_correct_av_step_host(TVInpaintFista::id[id].run, av_gc, TVInpaintFista::id[id].av_GX, gx_storage, run_step, "X" );
  if (TVInpaintFista::id[id].run % __debug_info_period__ == 1)
    std::cerr << "\nOutput gradient to gradX  " <<  num_above << " " << av_gc << "  vs  " << TVInpaintFista::id[id].av_GX << " vs " << is_largest << "   ";

  check_gradUpdate( gy_storage, av_gc, is_largest, num_above, TVInpaintFista::id[id].av_GY, "Y" );
  update_correct_av_step_host(TVInpaintFista::id[id].run, av_gc, TVInpaintFista::id[id].av_GY, gy_storage, run_step, "Y" );
  if (TVInpaintFista::id[id].run % __debug_info_period__ == 1)
    std::cerr << "\nOutput gradient to gradY  " <<  num_above << " " << av_gc << "  vs  " << TVInpaintFista::id[id].av_GY << " vs " << is_largest << "   ";

  check_gradUpdate( gb_storage, av_gc, is_largest, num_above, TVInpaintFista::id[id].av_GB, "B" );
  update_correct_av_step_host(TVInpaintFista::id[id].run, av_gc, TVInpaintFista::id[id].av_GB, gb_storage, run_step, "B" );
  if (TVInpaintFista::id[id].run % __debug_info_period__ == 1)
    std::cerr << "\nOutput gradient to gradB  " <<  num_above << " " << av_gc << "  vs  " << TVInpaintFista::id[id].av_GB << " vs " << is_largest << "   ";

  std::vector<float> vf( ic * iw * ih, 0 );
  cudaMemcpy( vf.data(), d_gxk1_temp.data(), ic * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );

  check_gradUpdate( vf, av_gc, is_largest, num_above, TVInpaintFista::id[id].av_GI, "I" );
  update_correct_av_step_host(TVInpaintFista::id[id].run, av_gc, TVInpaintFista::id[id].av_GI, vf, run_step, "I" );
  if (TVInpaintFista::id[id].run % __debug_info_period__ == 1)
    std::cerr << "\nOutput gradient to gradI  " <<  num_above << " " << av_gc << "  vs  " << TVInpaintFista::id[id].av_GI << " vs " << is_largest << "\n\n";
  cudaMemcpy( d_gxk1_temp.data(), vf.data(), vf.size()*sizeof(float), cudaMemcpyHostToDevice );
#endif

  // copy back:
  std::vector<float> gc_temp( ic * ih * iw, 0 );
  for (int i = 0; i < cc * ih * iw; i++)
    gc_temp[i] = (float) (gc_storage[i] * globalGMult);
  cudaMemcpy( d_gc.data(), gc_temp.data(), cc * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

  std::fill(gc_temp.begin(), gc_temp.end(), 0.0f);
  for (int i = 0; i < tc * ih * iw; i++)
    gc_temp[i] = (float) (gx_storage[i] * globalGMult);
  cudaMemcpy( d_gx.data(), gc_temp.data(), tc * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

  std::fill(gc_temp.begin(), gc_temp.end(), 0.0f);
  for (int i = 0; i < tc * ih * iw; i++)
    gc_temp[i] = (float) (gy_storage[i] * globalGMult);
  cudaMemcpy( d_gy.data(), gc_temp.data(), tc * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

  std::fill(gc_temp.begin(), gc_temp.end(), 0.0f);
  for (int i = 0; i < ic * ih * iw; i++)
    gc_temp[i] = (float) (gb_storage[i] * globalGMult);
  cudaMemcpy( d_gb.data(), gc_temp.data(), ic * ih * iw * sizeof(float), cudaMemcpyHostToDevice );

  // well i start with x=y=b .. so: -- if starting from 'some' init value -> add this accordingly.
  thrust::transform(d_gi.begin(), d_gi.end(), d_gxk1_temp.begin(), d_gi.begin(), thrust::plus<float>());
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

  TVInpaintFista::id[id].run++;
  return 0;
}