#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <vector>
#include <limits>

#include <cuda.h>

#include <iu/iucore.h>

#include <iostream>     // cout
// #include <math.h>       // acos
// #include <float.h>      // DBL_MAX


////////////////////////////////////////////////////
//
// If gradient updates are much larger ( scale_T * runningmean < |grad| < _kill_T_ * runningmean)
// than the running average, returned gradients are scaled to be at max _scale_T_ larger than running average.
// Superlarge ones are ignored ( > _kill_T_ * runningmean )
//
// Why? Assume a simple optimization scheme like RMSProp. RMSProp scales the gradients by the running mean.
// If one gradient is much large than this running mean an explosion of values happens.
// Ie. the last gradient step dominates learned values.
// This is happening very rarely, but the check solves all problems here, without negative influence.
//
// If gradient are _kill_T_ larger than runnig mean, ignore them all (See above rmsprop example).
#define _kill_T_  35.
//
// If gradient are _scale_T_ larger than runnig mean, rescale them to avoid too
// large updates due to the optimization algorithm. (See above rmsprop example)
#define _scale_T_ 10.
// memory / update rate of running mean
#define _memory_T_ 0.95
#define _kill_but_scale_
////////////////////////////////////////////////////


/// upper clipping range .. this is so large it never happens .. should not happen
#define _maxTolerance_ 5000000.


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
  return value != value;// True if NAN
}

template<typename T>
bool is_valid( const T &value )
{
  return ! is_infinite(value) && ! is_nan(value);
}

inline void printVector( iu::TensorGpu_32f& dev_vec, int T, int C, int N, int M, const char* XX)
{
  std::vector< float > host_vec(T * C * N * M);
  cudaMemcpy( host_vec.data(), dev_vec.data(), host_vec.size()*sizeof(float), cudaMemcpyDeviceToHost );

  // now print
  std::cout << "Vector " << XX[0] << "  " << T << "x" << C << "x" << N << "x" << M << "\n";
  for (int t = 0; t < T; t++)
  {
    std::cout << "[";
    for (int c = 0; c < C; c++)
    {
      std::cout << "[";
      for (int n = 0; n < N; n++)
      {
        std::cout << "[";
        for (int m = 0; m < M; m++)
          std::cout << host_vec[ t * (C * N * M) + c * (N * M) + n * (M) + m] << " ";
        std::cout << "]\n";
      }
      std::cout << "]\n";
    }
    std::cout << "]\n";
  }
}

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

template <typename T>
struct sqroot_
{
  __host__ __device__
  T operator()(const T& x) const {
    return sqrt(x);
  }
};

struct sqroot
{
  __host__ __device__
  float operator()(const float& x) const {
    return sqrtf(x);
  }
};

template <typename S>
void update_correct_av_step_host(int run, double av_gc, double& av_GC, std::vector<S>&tempc, double run_step, const char* XX )
{
  if ( run >= 10 && av_GC * _kill_T_ < av_gc )
  {
    std::cerr << "Killed the update  step for " << XX[0] << ": " << av_gc << " vs " << av_GC << " = " << av_gc / av_GC << " \n";
    //thrust::fill(grad_store.begin(), grad_store.end(), 0.0f);
    for (int i = 0; i < tempc.size(); i++)
      tempc[i] = 0.0;

    av_GC = av_GC * run_step + (1. - run_step) * _scale_T_ * av_GC; // increase running mean
    return;
  }
  const double scale = _scale_T_;
  if ( run >= 10 && av_GC * scale < av_gc )
  {
    double mult = scale * av_GC / av_gc;
    std::cerr << "Corrected the step " << XX[0] << " " << mult << ": " << av_gc << " vs " << av_GC << " = " << av_gc / av_GC << " \n";
    for (int i = 0; i < tempc.size(); i++)
      tempc[i] *= mult;

    // later: cudaMemcpy( grad_store.data(), tempc.data(), tempc.size()*sizeof(float), cudaMemcpyHostToDevice );
    av_GC = av_GC * run_step + (1. - run_step) * av_gc * mult;
  }
  else
  {
    av_GC = av_GC * run_step + (1. - run_step) * av_gc;
  }
}

// imho the problem occurs BEFORE this here fires .. but why ? must check inside adam/rmsprop updates ..
// clips the updates at maxTolerance ..
template <typename S> void check_gradUpdate( std::vector<S>& tempc, double &av_gc, double& is_largest, int& num_above, double av_gc_current, const char* XX )
{
  const double maxTolerance = _maxTolerance_;
  is_largest = 0; num_above = 0; av_gc = 0; int invlid = 0;
  for (int i = 0; i < tempc.size(); i++)
    if (is_valid( tempc[i]) ) {
      double tca = std::abs(tempc[i]);
      is_largest = std::max( tca, is_largest );
      if ( tca > maxTolerance ) {if (tempc[i] > 0) tempc[i] = maxTolerance; else tempc[i] = -maxTolerance; num_above++; tca = maxTolerance;}
      double step  = double(i) / double(i + 1);
      av_gc =  (1. - step) * tca + step * av_gc;
    }
    else
    {
      tempc[i] = 0;
      av_gc = av_gc_current * 1000.;
      invlid++;
    }

  if (invlid > 0)
    std::cerr << "Invalid gradient updates detected: " << XX[0] << " " << invlid << "\n";
  if (num_above > 0)
    std::cerr << "Gradient update clipped: " << XX[0] << " " << num_above << "\n";
}


inline void copy_and_set_to_0( std::vector<double>& gc_storage, std::vector<double>& gx_storage,
                               std::vector<double>& gy_storage, std::vector<double>& gb_storage,
                               iu::TensorGpu_32f& d_gb, iu::TensorGpu_32f& d_gc,
                               iu::TensorGpu_32f& d_gx, iu::TensorGpu_32f& d_gy,
                               int c_dims, int ingrad_c, int ih, int iw, float scale = 1., int dgx_ch = 1, int dgy_ch = 1 )
{
  std::vector<float> gc_temp( max(dgy_ch, max(dgx_ch, ingrad_c) ) * ih * iw, 0 );// ic:
  double maxGCGrad(0);
  cudaMemcpy( gc_temp.data(), d_gc.data(), c_dims * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
  for (int i = 0; i < c_dims * ih * iw; i++)
  {
    gc_storage[i] = (double) (gc_temp[i]) + gc_storage[i] / scale;
    maxGCGrad = std::max( maxGCGrad, (double) std::abs(gc_temp[i])  );
  }
  thrust::fill(d_gc.begin(), d_gc.end(), 0.0f);
  maxGCGrad = 0; cudaMemcpy( gc_temp.data(), d_gx.data(), dgx_ch * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
  for (int i = 0; i < dgx_ch * ih * iw; i++)
  {
    gx_storage[i] = (double) (gc_temp[i]) + gx_storage[i] / scale;
    maxGCGrad = std::max( maxGCGrad, (double) std::abs(gc_temp[i])  );
  }
  thrust::fill(d_gx.begin(), d_gx.end(), 0.0f);

  maxGCGrad = 0; cudaMemcpy( gc_temp.data(), d_gy.data(), dgy_ch * ih * iw * sizeof(float), cudaMemcpyDeviceToHost );
  for (int i = 0; i < dgy_ch * ih * iw; i++)
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

#undef _kill_T_
#undef _scale_T_
#undef _memory_T_
#undef _maxTolerance_