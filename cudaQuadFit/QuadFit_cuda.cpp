#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// compile with python setup.py install
class FlowQuadFitting
{
public:

	static int forward( float *d_im0, float *d_im1, float *d_dxdy, float *output,
                      int in, int ic, int ih, int iw );

  static int backward( float *d_im0, float *d_im1, float *d_dxdy, float *d_inGrad,
                       float *d_outGrad0, float *d_outGrad1,
                       int in, int ic, int ih, int iw );
};

//////////////////////////////////////////////////
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
//////////////////////////////////////////////////

std::vector<at::Tensor> flowQuadFitting_dummy( )
{
	std::cout << "Input: not present just dummy:\n";
	return std::vector<at::Tensor> {};
}

void flowQuadFitting_dummy_only( )
{
	std::cout << "Dummy call:\n";
	return;
}

// inputs  : featureImage0: feature vector from 1st image, 
//					 featureImage1: feature vector from 2nd image, 
//					 flow: initial flow between both images, 
std::vector<at::Tensor> flowQuadFitting_forward(at::Tensor featureImage0, 
																								at::Tensor featureImage1,
																								at::Tensor flow)
{
	CHECK_INPUT(featureImage0);
	CHECK_INPUT(featureImage1);
	CHECK_INPUT(flow);

	const int batches   = featureImage0.size(0);
	const int channel   = featureImage0.size(1);
	const int height    = featureImage0.size(2);
	const int width     = featureImage0.size(3);

	// if (batches != 1)
	// {
	// 		std::cout << "Cuda Function call flowQuadFitting::forward failed\n\
	// 		 The function can only handle a batchsize of one\n"; return {};
	// }
	//std::cout << "FW -- Input: " << batches << " " << channel << " " << height << " " << width <<"\n";

	if ( flow.type().scalarType() == at::ScalarType::Float )
	{
		at::Tensor flowUpdate = at::CUDA(at::kFloat).zeros({batches, 2, height, width});
		CHECK_INPUT(flowUpdate);

		int result = FlowQuadFitting::forward( (float*) ( featureImage0.data<float>() ), 
																					 (float*) ( featureImage1.data<float>() ),
  		                                     (float*) ( flow.data<float>() ),
																		       (float*) ( flowUpdate.data<float>() ),
		                                 			 batches, channel, height, width );
		CHECK_INPUT(flowUpdate);
		if (result)
		{
			std::cout << "Cuda Function call FlowQuadFitting::forward failed"; return {};
		}
		return {flowUpdate};
	}
	std::cout << "Input/Outout type to cuda operator must be single/float";
	return {};
}


// d_iGf   : the backpropagated gradient until here.
// inputs  : featureImage0: feature vector from 1st image, 
//					 featureImage1: feature vector from 2nd image, 
//					 flow: initial flow between both images, 
std::vector<at::Tensor> flowQuadFitting_backward( at::Tensor d_iGf, 
																									at::Tensor featureImage0, 
																									at::Tensor featureImage1,
																								  at::Tensor flow)
{
	CHECK_INPUT(featureImage0);
	CHECK_INPUT(featureImage1);
	CHECK_INPUT(flow);
	CHECK_INPUT(d_iGf);

	const int batches   = featureImage0.size(0);
	const int channel   = featureImage0.size(1);
	const int height    = featureImage0.size(2);
	const int width     = featureImage0.size(3);

	// cannot happen since forward is checking this already
	// if (batches != 1)
	// {
	// 		std::cout << "Cuda Function call tvInpaint::backward failed\n\
	// 		 The function can only handle a batchsize of one\n"; return {};
	// }
	//std::cout << "Input: " << batches << " " << channel << " " << height << " " << width <<"\n";

	if ( flow.type().scalarType() == at::ScalarType::Float )
	{
		auto d_oFeat0 = at::zeros_like( featureImage0 );
		auto d_oFeat1 = at::zeros_like( featureImage1 );

		int result = FlowQuadFitting::backward((float*) ( featureImage0.data<float>() ), 
																					 (float*) ( featureImage1.data<float>() ),
  		                                     (float*) ( flow.data<float>() ),
																					 (float*) ( d_iGf.data<float>() ), 
																					 (float*) ( d_oFeat0.data<float>() ), (float*) ( d_oFeat1.data<float>() )
																					 batches, channel, height, width );
		if (result)
		{
			std::cout << "Cuda Function call FlowQuadFitting::backward failed"; return {};
		}
		return {d_oFeat0, d_oFeat1};
	}
	std::cout << "Input/Outout type to cuda operator must be single/float";
	return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward",    &flowQuadFitting_forward,    "flowQuadFitting forward (CUDA)");
	m.def("backward",   &flowQuadFitting_backward,   "flowQuadFitting backward (CUDA)");
	m.def("dummy",      &flowQuadFitting_dummy,      "flowQuadFitting dummy");
	m.def("dummy_only", &flowQuadFitting_dummy_only, "flowQuadFitting dummy2");
}
