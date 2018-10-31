#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// compile with python setup.py install

// instead use my theano version cuda file .. 
class TVInpaint
{
public:

	static int forward(float *d_x, float *d_y, float *d_c, float *d_b, float *d_init, float *d_solution,
	                   int ic, int ih, int iw, int its = 10000, int id = 0);

	static int backward( float *d_x, float *d_y, float *d_c, float *d_b, float *d_inGrad,
	                     float *d_outGradX, float *d_outGradY, float *d_outGradC, float *d_outGradB, float *d_outGradInit,
	                     int ic, int ih, int iw, int its = 10000, int id = 0);
};

//////////////////////////////////////////////////
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
//////////////////////////////////////////////////

std::vector<at::Tensor> tvInpaint_dummy( )
{
	std::cout << "Input: not present just dummy:\n";
	return std::vector<at::Tensor> {};
}

void tvInpaint_dummy_only( )
{
	std::cout << "Dummy call:\n";
	return;
}

// dx,dy: tensors/edge weights in x and y direction. d_c: confidence, d_b rhs, eg. u_hat;
// d_i: initial solution used at iteration 0. simplest is d_b or solution from lower resolution
// d_out: output. ic: channels(# of rhs/channels of b), iw: width ih: height of inputs
// its: iterations to run.
// id: we can run multiple inpainters, eg. for different resolutions. To identify the buffers, etc.
// we need an id, that is specified by the user. Its usually a unique number between 0 and 9.
std::vector<at::Tensor> tvInpaint_forward(at::Tensor d_x, at::Tensor d_y,
at::Tensor d_c, at::Tensor d_b, at::Tensor d_i, int its, int id)
{
	CHECK_INPUT(d_x);
	CHECK_INPUT(d_y);
	CHECK_INPUT(d_c);
	CHECK_INPUT(d_b);
	CHECK_INPUT(d_i);

	const int batches   = d_b.size(0);
	const int channel   = d_b.size(1);
	const int height    = d_b.size(2);
	const int width     = d_b.size(3);

	if (batches != 1)
	{
			std::cout << "Cuda Function call tvInpaint::forward failed\n\
			 The function can only handle a batchsize of one\n"; return {};
	}
	//std::cout << "FW -- Input: " << batches << " " << channel << " " << height << " " << width <<"\n";

	if ( d_x.type().scalarType() == at::ScalarType::Float )
	{
		//auto solution = at::zeros_like( unaries );
		//at::Tensor solution = at::CUDA(at::kFloat).zeros({batches, 2, height, width});
		at::Tensor solution = at::zeros(torch::CUDA(at::kFloat), {1, channel, height, width});
		CHECK_INPUT(solution);

		int result = TVInpaint::forward( (float*) ( d_x.data<float>() ), (float*) ( d_y.data<float>() ),
		                                 (float*) ( d_c.data<float>() ), (float*) ( d_b.data<float>() ),
																		 (float*) ( d_i.data<float>() ), (float*) ( solution.data<float>() ),
		                                 channel, height, width, its, id );
		CHECK_INPUT(solution);
		if (result)
		{
			std::cout << "Cuda Function call TVInpaint::forward failed"; return {};
		}
		return {solution};
	}
	std::cout << "Input/Outout type to cuda operator must be single/float";
	return {};
}

// d_x,d_y: tensors/edge weights in x and y direction. d_c: confidence, d_b rhs, eg. u_hat;
// d_iGf: input gradient wrt f, ie. we compute f(wx,wy,c,b,i0) in the layer and then its dLoss/df
// d_oGx, d_oGy, d_oGc, d_oGb, d_oGi, gradients of our function wrt its input:
// X/Y: edge weights, C: confidence, b: rhs, I: initial solution for f(..)
// ic: channels, iw: width ih: height of inputs
// ingrad_c: how many input channels has our rhs b.
// its: iterations to run.
// id: we can run multiple inpainters, eg. for different resolutions. To identify the buffers, etc.
// we need an id, that is specified by the user. It is a number between 0 and 9 so far..
std::vector<at::Tensor> tvInpaint_backward( at::Tensor d_iGf, at::Tensor d_x, at::Tensor d_y,
				at::Tensor d_c, at::Tensor d_b, int its, int id)
{
	CHECK_INPUT(d_x);
	CHECK_INPUT(d_y);
	CHECK_INPUT(d_c);
	CHECK_INPUT(d_b);
	CHECK_INPUT(d_iGf);

	const int batches   = d_b.size(0);
	const int channel   = d_b.size(1);
	const int height    = d_b.size(2);
	const int width     = d_b.size(3);

	// cannot happen since forward is checking this already
	if (batches != 1)
	{
			std::cout << "Cuda Function call tvInpaint::backward failed\n\
			 The function can only handle a batchsize of one\n"; return {};
	}
	//std::cout << "Input: " << batches << " " << channel << " " << height << " " << width <<"\n";

	if ( d_b.type().scalarType() == at::ScalarType::Float )
	{
		auto d_oGx = at::zeros_like( d_x );
		auto d_oGy = at::zeros_like( d_y );
		auto d_oGc = at::zeros_like( d_c );
		auto d_oGb = at::zeros_like( d_b );
		auto d_oGi = at::zeros_like( d_b );

		int result = TVInpaint::backward((float*) ( d_x.data<float>() ), (float*) ( d_y.data<float>() ),
		                                 (float*) ( d_c.data<float>() ), (float*) ( d_b.data<float>() ),
																		 (float*) ( d_iGf.data<float>() ), 
																		 (float*) ( d_oGx.data<float>() ), (float*) ( d_oGy.data<float>() ),
																		 (float*) ( d_oGc.data<float>() ), (float*) ( d_oGb.data<float>() ),
																		 (float*) ( d_oGi.data<float>() ), channel, height, width, its, id );
		if (result)
		{
			std::cout << "Cuda Function call TVInpaint::forward failed"; return {};
		}
		return {d_oGx, d_oGy, d_oGc, d_oGb, d_oGi};
	}
	std::cout << "Input/Outout type to cuda operator must be single/float";
	return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward",    &tvInpaint_forward,    "tvInpaint forward (CUDA)");
	m.def("backward",   &tvInpaint_backward,   "tvInpaint backward (CUDA)");
	m.def("dummy",      &tvInpaint_dummy,      "tvInpaint dummy");
	m.def("dummy_only", &tvInpaint_dummy_only, "tvInpaint dummy2");
}
