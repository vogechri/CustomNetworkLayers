//#include "flow_src/flowmatch.h"
#include "medimax.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <iu/iucore.h>

// as in enforce sum to 1 .. 
//#define __Normalization__

//kMedianScoreGrad<<<numBlocks, threadsPerBlock>>>(d_outG, d_in, d_inG );
//kMedianScore<<<numBlocks, threadsPerBlock>>>( d_out, d_in ); // i need as well .. ?

/// look up correlation with a forward lookup instead of simple one ..
__global__ void kMedianScore(iu::TensorGpu_32f::TensorKernelData medianFlowScore, iu::TensorGpu_32f::TensorKernelData probVol )
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= medianFlowScore.length_) return;

    short n, h, w, dummy;//d, 
  	medianFlowScore.coords( t_idx, &n, &dummy, &h, &w ); // this one has 2 channels .. so trick it ?

	  float score =0;
    float w2 = 0;//float( probVol.W )/ 2.;
  	for (int c = 0; c < probVol.W; ++c) // it is NCHW
  	{
	    float cScore = probVol(n, h, w, c);
      if ( score < 0.5 && score + cScore >= 0.5) // transition between c and c+1
      {
        // alpha should be within [-0.5, 0.5] ..
        //float alpha = ( 1. - (score + cScore) - score ) / ( 2. * cScore ) ; // exactlocation of transition.. alpha = B-A =
        float alpha = ( 1. - 2. * score - cScore ) / ( 2. * cScore ); // exactlocation of transition.. alpha = B-A =
        medianFlowScore(n, 0, h, w) = ( float(c) - w2 + alpha); // yes, transition can occur at -0.5 of course if p(0) == 1! (not perfect -- i know)
        medianFlowScore(n, 1, h, w) = cScore;        // ok ..
        //  printf("%dx%d: %.1f -- %.3f / %.3f\n", h,w, float(c) - w2 + alpha, cScore, score);
        break; // does this exist ?
	    }
	    score = score + cScore;
  	}
}

__global__ void kMedianScoreGrad(iu::TensorGpu_32f::TensorKernelData probVolG, iu::TensorGpu_32f::TensorKernelData medianFlowScoreGrad, iu::TensorGpu_32f::TensorKernelData probVol )
{
  int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= medianFlowScoreGrad.length_)  return;

  short n, h, w, dummy;// dummy is 2 -- no ? d, 
	medianFlowScoreGrad.coords( t_idx, &n, &dummy, &h, &w ); // this one has 2 channels .. so trick it ?

	float score = 0;
	float alpha = 0;
	int crossingFound = 0;

  float pmid= 1.;
	for (int c = 0; c < probVol.W; ++c) // it is NCHW
 	{
	  float cScore = probVol(n, h, w, c);
    //if ( crossingFound ==1 )
    //  probVolG(n,h,w,c) = medianFlowScoreGrad(n, 0, h, w) / (2.*pmid)*0 - 1./(probVol.W-1); // how does cScore==pmid change if i add eps to any other? goes down by eps/n 

 	  if ( crossingFound==0 && score < 0.5 && score + cScore >= 0.5) // transition between c and c+1
    {
		  crossingFound = 1;
      pmid = cScore;
	    alpha = ( 1. - 2. * score - cScore ) / ( 2. * cScore ) ; // exactlocation of transition..
		  // 0: flow grad .. (median style)  1: conf grad: as usual just 1 if at spot ; 0:else
      // actually the latter summand should be divided by (sum_pi)^2 -- which is one
#ifdef __Normalization__
		  probVolG(n,h,w,c) = medianFlowScoreGrad(n, 0, h, w) * alpha / (-pmid) + (1-pmid) * medianFlowScoreGrad(n, 1, h, w); // assumes that sum pi == 1 !
#else
		  probVolG(n,h,w,c) = medianFlowScoreGrad(n, 0, h, w) * alpha / (-pmid) + medianFlowScoreGrad(n, 1, h, w); // assumes that sum pi == 1 is FUCKING FIXED and handled from outside 
#endif
      break;
	  }
	  score = score + cScore;
  }

  score = 0;
	for (int c = 0; c < probVol.W; ++c) // it is NCHW
 	{
	  float cScore = probVol(n, h, w, c);
#ifdef __Normalization__
 	  if ( score + cScore < 0.5) // transition between c and c+1
      probVolG(n,h,w,c) = - medianFlowScoreGrad(n, 0, h, w) / (2.*pmid)  - pmid * medianFlowScoreGrad(n, 1, h, w);// actually latter summand should be divided by (sum_pi)^2 -- which is one
    else
 	  if ( score >= 0.5) // transition between c and c+1
      probVolG(n,h,w,c) = + medianFlowScoreGrad(n, 0, h, w) / (2.*pmid)  - pmid * medianFlowScoreGrad(n, 1, h, w);// actually latter summand should be divided by (sum_pi)^2 -- which is one
#else
 	  if ( score + cScore < 0.5) // transition between c and c+1
      probVolG(n,h,w,c) = - medianFlowScoreGrad(n, 0, h, w) / (2.*pmid);// sum to 1 doen outside -- no need to do here 
    else
 	  if ( score >= 0.5) // transition between c and c+1
      probVolG(n,h,w,c) = + medianFlowScoreGrad(n, 0, h, w) / (2.*pmid);// sum to 1 doen outside -- no need to do here 
#endif
      //break;
	  score = score + cScore;
  }
}

__global__ void kMedianScoreGrad_alt(iu::TensorGpu_32f::TensorKernelData probVolG, iu::TensorGpu_32f::TensorKernelData medianFlowScoreGrad, iu::TensorGpu_32f::TensorKernelData probVol )
{
  int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= medianFlowScoreGrad.length_)  return;

  short n, d, h, w, dummy;// dummy is 2 -- no ? 
	medianFlowScoreGrad.coords( t_idx, &n, &dummy, &h, &w ); // this one has 2 channels .. so trick it ?

	float score = 0;
	float alpha = 0;
	int crossingFound = 0;
	for (int c = 0; c < probVol.W; ++c) // it is NCHW
 	{
	  float cScore = probVol(n, h, w, c);
 	  if ( crossingFound==0 && score < 0.5 && score + cScore >= 0.5) // transition between c and c+1
    {
		  crossingFound = 1;
	    alpha = ( 1. - 2. * score - cScore ) / ( 2. * cScore ) ; // exactlocation of transition..
		  // 0: flow grad .. (median style)  1: conf grad: as usual just 1 if at spot ; 0:else
		  probVolG(n,h,w,c) = medianFlowScoreGrad(n, 0, h, w) * 2. * alpha + medianFlowScoreGrad(n, 1, h, w);
          //medianFlowScore(n, 0, h, w) = float(c)+alpha; // yes, transition can occur at -0.5 of course if p(0) == 1! (not perfect -- i know)
          //medianFlowScore(n, 1, h, w) = cScore;        // ok ..
	  }
	  else
	  {
	    if (crossingFound==0) // add
		  {
		    probVolG(n,h,w,c) =   medianFlowScoreGrad(n, 0, h, w) * probVol(n,h,w,c);
		  }
      else // sub
  		{
	  	  probVolG(n,h,w,c) = - medianFlowScoreGrad(n, 0, h, w) * probVol(n,h,w,c);
		  }
	  }

	  score = score + cScore;
  }
}

// ? and now .. ? input is 4Dimensional .. so i do split always ??? -- maybe better ? yes .. ! so always u and v part split ?
int MediMax::forward( float *d_vol0, float *d_outFS0, int in, int ih, int iw, int numDisp )
{
  int ic = numDisp;
  iu::TensorGpu_32f d_in(d_vol0, in, ih, iw, numDisp, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_out(d_outFS0,  in, 2, ih, iw, true, iu::TensorGpu_32f::NCHW); // 1st median, then score

  if ( MediMax::run++ <= 0 )
  {
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    std::cout <<"using "<<properties.multiProcessorCount<<" multiprocessors with -- max threads per processor: " <<properties.maxThreadsPerMultiProcessor<<std::endl;
    std::cout << "Inputs : " << in << " " << ic << " " << ih << " " << iw << "\n";
  }

//#define _debug_
#ifdef _debug_
  std::vector<float> h_rhs( in*ih*iw*numDisp, 0); // huge .. 
 	cudaMemcpy(h_rhs.data(), d_vol0.data(), h_rhs.size()*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = in*ih*iw/2; i < in*ih*iw/2 + numDisp; ++i)
       std::cout <<  h_rhs[i] << " ";
  std::cout << "\n";
#endif

  thrust::fill(d_out.begin(), d_out.end(), 0.0f);

  dim3 threadsPerBlock(480);
	dim3 numBlocks(
			std::ceil(
					(d_out.samples() * d_out.height() * d_out.width() ) //* d_out.channels())
							/ static_cast<float>(threadsPerBlock.x)));

    kMedianScore<<<numBlocks, threadsPerBlock>>>( d_out, d_in ); // i need as well .. ?
    cudaDeviceSynchronize();

#ifdef _debug_
  std::vector<float> o_rhs( in*ih*iw, 0); // huge .. 
 	cudaMemcpy(o_rhs.data(), d_outFS0.data(), o_rhs.size()*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = in*ih*iw/2; i < in*ih*iw/2 + iw; ++i)
       std::cout <<  o_rhs[i] << " ";
  std::cout << "\n";
#endif

    return 0;
}

int MediMax::backward( float *d_vol0, float *d_inGrad, float *d_outGrad, int in, int ih, int iw, int numDisp )
{
  iu::TensorGpu_32f d_in(d_vol0,      in, ih, iw, numDisp, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_inG(d_inGrad,   in,  2, ih, iw, true, iu::TensorGpu_32f::NCHW);
  iu::TensorGpu_32f d_outG(d_outGrad, in, ih, iw, numDisp, true, iu::TensorGpu_32f::NCHW);

  thrust::fill(d_outG.begin(), d_outG.end(), 0.0f);

  dim3 threadsPerBlock(480);
	dim3 numBlocks(
			std::ceil(
					(d_inG.samples() * d_inG.height() * d_inG.width() ) //* d_outG0.channels())
							/ static_cast<float>(threadsPerBlock.x)));

    kMedianScoreGrad<<<numBlocks, threadsPerBlock>>>(d_outG, d_inG, d_in );
    cudaDeviceSynchronize();
    return 0;
}
