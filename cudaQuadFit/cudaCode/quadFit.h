#pragma once

// This layer does the quad-fitting given the (integral, or continuos) intial flow  
// and feature vectors that are looked up at location (x+flow) via nearest neighbour interpolation. 
// For the 5-stencil of the displaced flow and at the integral location of the 
// pixel for which we have the motion vector.
//  
// returns only the continuous DISPLACEMENT! not adding the original flow and the computed cost at the location.
// This can be done after calling this function. 
//
// There is no gradient wrt. the input flow: such a gradient does not make sense. 
// Changing the location slightly, the fit remains constant. So a 0 gradient is correct. 
class FlowQuadFitting
{
public:
	FlowQuadFitting();
	~FlowQuadFitting();

  // inputs  : d_im0: feature vector from 1st image, d_im1 feature vector from 2nd image, d_dxdy: flow, 
  // outputs : output: local flow displacement, best fit flow is d_dxdy+out!
  // other   : in: batch size, ic: channel, ih: height of image, iw: width of image
  static int forward(float *d_im0, float *d_im1, float *d_dxdy, float *output,
                     int in, int ic, int ih, int iw );

  // inputs  : d_im0: feature vector from 1st image, d_im1 feature vector from 2nd image, d_dxdy: flow, 
  // d_inGrad: the backpropagated gradient.
  // outputs : d_outGrad0: the gradients wrt. features of first image, 
  //           d_outGrad1: the gradient wrt. features of second image
  // other:  in: batch size, ic: channel, ih: height of image, iw: width of image
  static int backward( float *d_im0, float *d_im1, float *d_dxdy, float *d_inGrad,
                       float *d_outGrad0, float *d_outGrad1,
                       int in, int ic, int ih, int iw );

private:
	// no copies!
	FlowQuadFitting(FlowQuadFitting const&);
	void operator=(FlowQuadFitting const&);
};
