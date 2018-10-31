#pragma once
#include <vector>

///
#define __uniqueIds__ 10

class idStruct {
public:
	idStruct(int iteration = 100)
	{
		maxInterIts = 0;
		stepSizes.resize(iteration, 0);
		intermediate_x.resize(iteration, NULL);
		intermediate_y.resize(iteration, NULL);
		iterations.resize(iteration, 0);
		run = 0; av_GC = 0; av_GY = 0; av_GX = 0; av_GB = 0; av_GI = 0;
	}
	~idStruct()
	{
		free_buf();
	}

	void free_buf ()
	{
		for (int i = 0; i < intermediate_x.size(); i++)
			if (intermediate_x[i] != NULL )
			{
				free (intermediate_x[i]);
				intermediate_x[i] = NULL;
			}
		for (int i = 0; i < intermediate_y.size(); i++)
			if (intermediate_y[i] != NULL )
			{
				free (intermediate_y[i]);
				intermediate_y[i] = NULL;
			}
	}

	int run;      // counting the number of calls of FW/BW path.
	double av_GC; // running mean of gradient norm wrt. confidence
	double av_GY; // running mean of gradient norm wrt. edge-weight in y-direction
	double av_GX; // running mean of gradient norm wrt. edge-weight in x-direction
	double av_GB; // running mean of gradient norm wrt. rhs, eg u_hat
	double av_GI; // running mean of gradient norm wrt. input
	std::vector<int>    iterations;     // the fixpoints with intermediate storage
	std::vector<float*> intermediate_x; // checkpointing variables
	std::vector<float*> intermediate_y; // checkpointing variables
	std::vector<float>  stepSizes;      // just precompute ..
	int maxInterIts; // maximal number of iterations run here.
};



class TVInpaintFista
{
public:

	typedef int intType;

	TVInpaintFista();
	~TVInpaintFista();


// dx,dy: tensors/edge weights in x and y direction. d_c: confidence, d_b rhs, eg. u_hat;
// d_b rhs, eg. u_hat; d_i: initial solution, eg. d_b or from other hirarchy level.
// d_solution: output. 
// ic: channels(# of rhs/channels of b), iw: width ih: height of inputs
// its: iterations to run.
// id: we can run multiple inpainters, eg. for different resolutions. To identify the buffers, etc.
// we need an id, that is specified by the user. Its a number between 0 and 9 so far..
	static int forward(float *d_dx, float *d_dy, float *d_c, float *d_b, float *d_i, float *d_solution,
	                   int ic, int ih, int iw, int its = 10000, int id = 0);

// dx,dy: tensors/edge weights in x and y direction. d_c: confidence, d_b rhs, eg. u_hat;
// d_inGrad: input gradient, ie. we compute f(wx,wy,c,b,i0) in the layer and then its dLoss/df
// d_outGradX, d_outGradY, d_outGradC, d_outGradB, d_outGradI, gradients of our function wrt its input:
// X/Y: edge weights, C: confidence, b: rhs, I: initial solution for f(..)
// ic: how many input channels has our rhs b, iw: width ih: height of inputs
// its: iterations to run.
// id: we can run multiple inpainters, eg. for different resolutions. To identify the buffers, etc.
// we need an id, that is specified by the user. It is a number between 0 and 9 so far..
	static int backward( float *d_dx, float *d_dy, float *d_c, float *d_b,
	                     float *d_inGrad,
	                     float *d_outGradX, float *d_outGradY, float *d_outGradC, float *d_outGradB, float *d_outGradI,
	                     int ic, int ih, int iw, int its = 10000, int id = 0);

private:
	// no copies!
	TVInpaintFista(TVInpaintFista const&);
	void operator=(TVInpaintFista const&);

	static std::vector<idStruct> id;
};

// we use 10 buffers .. ie. we cna run 10 independent instances.
std::vector <idStruct> TVInpaintFista::id = std::vector <idStruct>(__uniqueIds__);
#undef __uniqueIds__