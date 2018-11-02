#include "quadFit.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <iu/iucore.h>

/// the result of the quad fit is a value that needs to be  added to the input flow value

// feature vectors img1 and img2 of 1st/2nd image. flow: initial guess that gets refined
// qf: the output displacement. quarter: cheap way from thread id to pixel location. 
// d: # of channels of the features 
__global__ void kQuadFitFlow(iu::TensorGpu_32f::TensorKernelData img1,
                         iu::TensorGpu_32f::TensorKernelData img2,
                         iu::TensorGpu_32f::TensorKernelData flow,
                         iu::TensorGpu_32f::TensorKernelData qf, 
                         iu::TensorGpu_32f::TensorKernelData quarter,
                         int d   )
{
  // start here ..
  int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= quarter.length_)
      return;

  // get pixel location h,w
  short n, dum, h, w;
	quarter.coords(t_idx, &n, &dum, &h, &w);

  // find displaced pixel
  float idx = w+flow(n, 0, h, w);
  float idy = h+flow(n, 1, h, w);

  // cannot fit, not in image .. 
  if (idx > img1.W-1 || idx < 0 || idy > img1.H-1 || idy<0)
  {
    return;
  }

#ifdef __nonsense__
  // that is a problem .. if oob -> score 0 and flow = ? why would that be outside ? 
  idx = min(static_cast<float>(img1.W - 1), max(static_cast<float>(0), idx));
  idy = min(static_cast<float>(img1.H - 1), max(static_cast<float>(0), idy));
#endif

  const int iidx = floor(idx+0.5);
  const int iidy = floor(idy+0.5);

  const int iidx_m = max(iidx - 1, 0 );
  const int iidy_m = max(iidy - 1, 0 );
  const int iidx_p = min(iidx + 1, img1.W-1 );
  const int iidy_p = min(iidy + 1, img1.H-1 );

  //////// lookup 5-stencil of integral cost values 
  float cost00(0);
  float cost0p(0);//01
  float costp0(0);//10
  float cost0m(0);//0,-1
  float costm0(0);//-1,0

  // scalar product
  for (int cc=0;cc<d;cc++)
  {
    const float I1 = img1(n, cc, h, w);
  	cost00 +=  I1 * img2(n, cc, iidy,   iidx);
  	cost0p +=  I1 * img2(n, cc, iidy,   iidx_p);
  	costp0 +=  I1 * img2(n, cc, iidy_p, iidx);
  	cost0m +=  I1 * img2(n, cc, iidy,   iidx_m);
  	costm0 +=  I1 * img2(n, cc, iidy_m, iidx);
  }
  ////////////////////////

  // fit the cost (idx+0,idy+0, fx+dx, fy+dy) .. 
  float ax = 0.5*( cost0p+cost0m -2.*cost00);
  float ay = 0.5*( costp0+costm0 -2.*cost00);
  float bx = 0.5*( cost0p-cost0m );
  float by = 0.5*( costp0-costm0 );
  float fx(0); 
  float fy(0);

  // positive definite case ?
  int validX = (ax<0 && ax*ax>=0.25*bx*bx);
  int validY = (ay<0 && ay*ay>=0.25*by*by);

  if (! (iidx < img2.W-1 && iidx > 0 ) )
    validX = 0;
  if (! (iidy < img2.H-1 && iidy > 0 ) )  
    validY = 0;

  // compute the fit == parameters of quadratic per dimension: 
  if (validX)  // a>0.5*bx -> still < 1
    fx = -0.5*bx/ax;// a>0.5*bx -> still fx < 1
    // simpler : all 0 !
    /*
  else 
    if(bx>0)
      fx = 1.0;
    else 
      if(bx>0)
        fx = -1.0;
        */
  if (validY)
    fy = -0.5*by/ay;// a>0.5*bx -> still fx < 1
/*
  else 
    if(by>0)
      fy = 1.0;
    else 
      if(by>0)
        fy = -1.0;
*/
#ifdef  __return__add_to_input__
  //fx-(idx-iidx) = iidx+fx - idx = bestfit - idx ; hence find best overall flow by adding back the input !
  if (iidx < img2.W-1 && iidx > 0 )
    qf(n, 0, h, w) = fx-(idx-iidx);/// RETURN INTEGER FLOW PLUS DISPLACEMENT OR ONLY DISPLACEMENT
  else
    fx = 0;
  if (iidy < img2.H-1 && iidy > 0 )
    qf(n, 1, h, w) = fy-(idy-iidy);/// RETURN INTEGER FLOW PLUS DISPLACEMENT OR ONLY DISPLACEMENT
  else
    fy = 0;
#else
  if (iidx < img2.W-1 && iidx > 0 )
    qf(n, 0, h, w) = fx;//+iidx;/// RETURN INTEGER FLOW PLUS DISPLACEMENT OR ONLY DISPLACEMENT
  else
    fx = 0;
  if (iidy < img2.H-1 && iidy > 0 )
    qf(n, 1, h, w) = fy;//+iidy;/// RETURN INTEGER FLOW PLUS DISPLACEMENT OR ONLY DISPLACEMENT
  else
    fy = 0;
#endif

  float score = ax*fx*fx + bx*fx + cost00 + ay*fy*fy + by*fy;
  qf(n, 2, h, w) = score;

}

// outputg gradients: imgG1 , imgG2 of feature vectors of 1st / 2nd image.
// feature vectors img1 and img2 of 1st/2nd image. flow: initial guess that gets refined
// inG: the backpropagated gradient. 
// d: # of channels of the features 
__global__ void kQuadFitFlowGrad( iu::TensorGpu_32f::TensorKernelData  imgG1,
                                   iu::TensorGpu_32f::TensorKernelData imgG2,
                                   iu::TensorGpu_32f::TensorKernelData img1,
                                   iu::TensorGpu_32f::TensorKernelData img2,
                                   iu::TensorGpu_32f::TensorKernelData flow,
                                   iu::TensorGpu_32f::TensorKernelData inG,
                                   iu::TensorGpu_32f::TensorKernelData quarter, int d )
{
  int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= quarter.length_)
      return;

  short n, h, w, dum;
	quarter.coords(t_idx, &n, &dum, &h, &w); // get pixel location

  // find displaced pixel
  float idx = w+flow(n, 0, h, w);
  float idy = h+flow(n, 1, h, w);

  if (idx > img1.W-1 || idx < 0 || idy > img1.H-1 || idy<0)
  {
    return;
  }

#ifdef __nonsense__  
  // that is a problem .. if oob -> score 0 and flow = ? keep as is .. 
  idx = min(static_cast<float>(img1.W - 1), max(static_cast<float>(0), idx));
  idy = min(static_cast<float>(img1.H - 1), max(static_cast<float>(0), idy));
#endif

  const int iidx = floor(idx+0.5);
  const int iidy = floor(idy+0.5);

  const int iidx_m = max(iidx - 1, 0 );
  const int iidy_m = max(iidy - 1, 0 );
  const int iidx_p = min(iidx + 1, img1.W-1 );
  const int iidy_p = min(iidy + 1, img1.H-1 );

  // lookup 5-stencil of integral cost values 
  float cost00(0);
  float cost0p(0);//01
  float costp0(0);//10
  float cost0m(0);//0,-1
  float costm0(0);//-1,0

  for (int cc=0;cc<d;cc++) // over channels of feature descriptor
  {
    const float I1 = img1(n, cc, h, w);
  	cost00 +=  I1 * img2(n, cc, iidy,   iidx);
  	cost0p +=  I1 * img2(n, cc, iidy,   iidx_p);
  	costp0 +=  I1 * img2(n, cc, iidy_p, iidx);
  	cost0m +=  I1 * img2(n, cc, iidy,   iidx_m);
  	costm0 +=  I1 * img2(n, cc, iidy_m, iidx);
  }
  /////////////////////
  // the fit cost (idx+0,idy+0, fx+dx, fy+dy) .. 
  //const float c00 = cost00;
  float ax = 0.5*( cost0p+cost0m -2.*cost00);
  float ay = 0.5*( costp0+costm0 -2.*cost00);
  float bx = 0.5*( cost0p-cost0m );
  float by = 0.5*( costp0-costm0 );
  float fx(0),fy(0);

  // positive definit case per dimension?   
  int validX = (ax<0 && ax*ax>=0.25*bx*bx);
  int validY = (ay<0 && ay*ay>=0.25*by*by);
  if (! (iidx < img2.W-1 && iidx > 0 ) )
    validX = 0;
  if (! (iidy < img2.H-1 && iidy > 0 ) )  
    validY = 0;


  // compute the fit == parameters of quadratic per dimension:     
  if (ax<0 && validX)  // a>0.5*bx -> still < 1
    fx = -0.5*bx/ax;// a>0.5*bx -> still fx < 1
  else
  { 
    /*
    ax=0;
    if(bx>0)
      fx = 1.0;
    else 
      if(bx<0)
        fx = -1.0;
   */
  }
  if (ay<0 && validY)
    fy = -0.5*by/ay;// a>0.5*bx -> still fx < 1
  else 
  {
    /*
    ay=0;
    if(by>0)
      fy = 1.0;
    else 
      if(by<0)
        fy = -1.0;
   */
  }

  //float score = ax*fx*fx + bx*fx + cost00 + ay*fy*fy + by*fy;

  // derivative needs: 
  const float dfx_dbx = validX ? -1./(2.*ax) : 0.;
  const float dfx_dax = validX ?     -fx/ax  : 0.;
  const float dfy_dby = validY ? -1./(2.*ay) : 0.;
  const float dfy_day = validY ?     -fy/ay  : 0.;

  const float ds_dax = validX ? fx*fx     : 0.; // score / dax etc
  const float ds_dbx = validX ? fx        : 0.;
  const float ds_dc  = (validX || validY) ? 1. : 0.;
  const float ds_day = validY ? fy*fy     : 0.; // score / dax etc
  const float ds_dby = validY ? fy        : 0.;

  // dax_dCost = 1(cost0p)  1 cost0m -2(cost00)
  // dbx_dCost = 1(cost0p) -1 cost0m
  // day_dCost = 1(costp0)  1 costm0 -2(cost00)
  // dby_dCost = 1(costp0) -1 costm0
  // dC/dIx = I_(1-x)

  float dFdfx   = inG(n, 0, h, w); // fx
  float dFdfy   = inG(n, 1, h, w); // fy
  float dFds    = inG(n, 2, h, w); // score    
  
  for (int cc=0;cc<d;cc++)
  {
    const float I1    = img1(n, cc,    h,    w);
  	const float I2    = img2(n, cc, iidy, iidx);
  	const float I2_0p = img2(n, cc, iidy, iidx_p);
  	const float I2_0m = img2(n, cc, iidy, iidx_m);
  	const float I2_p0 = img2(n, cc, iidy_p, iidx);    
  	const float I2_m0 = img2(n, cc, iidy_m, iidx);    

    if (validX || validY)
    {
      atomicAdd( &imgG2( n, cc, iidy,   iidx  ), 0.5 * I1 * ( -2.*dFdfx * dfx_dax -2.* dFdfy * dfy_day - 2. * dFds * ds_dax - 2. * dFds * ds_day + 2. * dFds * ds_dc) );
      atomicAdd( &imgG2( n, cc, iidy,   iidx_p), 0.5 * I1 * ( dFdfx * dfx_dax + dFdfx * dfx_dbx + dFds * ds_dax + dFds * ds_dbx ) );
      atomicAdd( &imgG2( n, cc, iidy,   iidx_m), 0.5 * I1 * ( dFdfx * dfx_dax - dFdfx * dfx_dbx + dFds * ds_dax - dFds * ds_dbx ) );
      atomicAdd( &imgG2( n, cc, iidy_p, iidx  ), 0.5 * I1 * ( dFdfy * dfy_day + dFdfy * dfy_dby + dFds * ds_day + dFds * ds_dby) );
      atomicAdd( &imgG2( n, cc, iidy_m, iidx  ), 0.5 * I1 * ( dFdfy * dfy_day - dFdfy * dfy_dby + dFds * ds_day - dFds * ds_dby) );

      // deriv wrt I00: 
      imgG1( n, cc, h,   w  ) =  0.5* ( I2   * ( -2.* dFdfx * dfx_dax -2.* dFdfy * dfy_day - 2. * dFds * ds_dax - 2. * dFds * ds_day + 2. * dFds * ds_dc) + 
                                                       I2_0p * ( dFdfx * dfx_dax + dFdfx * dfx_dbx + dFds * ds_dax + dFds * ds_dbx ) +
                                                       I2_0m * ( dFdfx * dfx_dax - dFdfx * dfx_dbx + dFds * ds_dax - dFds * ds_dbx ) +
                                                       I2_p0 * ( dFdfy * dfy_day + dFdfy * dfy_dby + dFds * ds_day + dFds * ds_dby ) +
                                                       I2_m0 * ( dFdfy * dfy_day - dFdfy * dfy_dby + dFds * ds_day - dFds * ds_dby ) ); 

    }
    else // only score at c00 has gradient .. passing through as is .. a,b==0
    {
      imgG1( n, cc,    h,      w) = I2 * dFds;
      atomicAdd( &imgG2( n, cc, iidy,   iidx),  I1 * dFds );
    }
  }  
}


// inputs  : d_input0: feature vector from 1st image, d_input1: feature vector from 2nd image, d_dxdy: flow, 
// outputs : d_outF: local flow displacement, best fit flow is d_dxdy+out!
// other   : in: batch size, ic: channel, ih: height of image, iw: width of image
int FlowQuadFitting::forward( float *d_input0, float *d_input1, float *d_dxdy, float *d_outF,
                              int in, int ic, int ih, int iw )
{
  iu::TensorGpu_32f d_img0(d_input0, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_img1(d_input1, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_flow(d_dxdy,   in,  2, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_qf  (d_outF,   in,  3, ih, iw, true, iu::TensorGpu_32f::NCHW);// flow and scores as well
  iu::TensorGpu_32f dummy( in, 1, ih, iw, iu::TensorGpu_32f::NCHW);// cheap coding effort .. :)
  thrust::fill(d_qf.begin(), d_qf.end(), 0.0f);

  dim3 threadsPerBlock(480);
	dim3 numBlocks(
			std::ceil(
					(dummy.samples() * dummy.width() * dummy.channels() * dummy.height())
							/ static_cast<float>(threadsPerBlock.x)));
		kQuadFitFlow<<<numBlocks, threadsPerBlock>>>( d_img0, d_img1, d_flow, d_qf, dummy, ic );

    cudaDeviceSynchronize();

    return 0;
}

// inputs  : d_input0: feature vector from 1st image, d_input1: feature vector from 2nd image, d_dxdy: flow, 
// d_inGrad: the backpropagated gradient.
// outputs : d_outGrad0: the gradients wrt. features of first image, 
//           d_outGrad1: the gradient wrt. features of second image
// other:  in: batch size, ic: channel, ih: height of image, iw: width of image
int FlowQuadFitting::backward( float *d_input0, float* d_input1, float *d_dxdy,
                               float *d_inGrad,
                               float* d_outGradIm0, float* d_outGradIm1,
                               int in, int ic, int ih, int iw )
{
  iu::TensorGpu_32f d_img0(d_input0, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_img1(d_input1, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_flow(d_dxdy  , in,  2, ih, iw, true, iu::TensorGpu_32f::NCHW);

  iu::TensorGpu_32f d_inG(d_inGrad, in,   3, ih, iw, true, iu::TensorGpu_32f::NCHW);

  iu::TensorGpu_32f d_imgG0(d_outGradIm0,  in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_imgG1(d_outGradIm1,  in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);

  iu::TensorGpu_32f dummy( in, 1, ih, iw, iu::TensorGpu_32f::NCHW);// cheap coding effort .. :)

  thrust::fill( d_imgG0.begin(), d_imgG0.end(),  0.0f);
  thrust::fill( d_imgG1.begin(), d_imgG1.end(),  0.0f);

  dim3 threadsPerBlock(480);
	dim3 numBlocks(
			std::ceil(
					(dummy.samples() * dummy.width() * dummy.channels() * dummy.height())
							/ static_cast<float>(threadsPerBlock.x)));

		kQuadFitFlowGrad<<<numBlocks, threadsPerBlock>>>(d_imgG0, d_imgG1, d_img0, d_img1, d_flow, d_inG, dummy, ic );

    cudaDeviceSynchronize();

    return 0;
}