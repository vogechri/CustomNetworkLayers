#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// compile with python setup.py install

// instead use my theano version cuda file .. good enough
class MediMax
{
  public:
static int forward ( float *d_vol, float *d_med, int in, int ic, int ih, int iw );
static int backward( float *d_vol, float *d_inGrad, float *d_outGrad, int in, int ic, int ih, int iw );
};

//////////////////////////////////////////////////
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
//////////////////////////////////////////////////

std::vector<at::Tensor> median_dummy( ) //at::Tensor unaries ) 
{
  std::cout << "Input: not present just dummy:\n";
  return std::vector<at::Tensor> {};
}

void median_dummy_only( ) //at::Tensor unaries ) 
{
  std::cout << "Dummy call:\n";
  return;
}

std::vector<at::Tensor> median_forward(at::Tensor in_vol) 
{
  //std::cout << "FW -- Input: (its/id):" << its << "/" << id <<"\n";//return{};
  //return std::vector<at::Tensor> {unaries};

  CHECK_INPUT(in_vol);
 
  const int batches   = in_vol.size(0);
  const int channel   = in_vol.size(3);
  const int height    = in_vol.size(1);
  const int width     = in_vol.size(2);

  //std::cout << "FW -- Input: " << batches << " " << channel << " " << height << " " << width <<"\n";
  
  if ( in_vol.type().scalarType() == at::ScalarType::Float )
  {
    //auto solution = at::zeros_like( unaries );
    //at::Tensor solution = at::CUDA(at::kFloat).zeros({batches, 2, height, width});
    at::Tensor solution = at::zeros(torch::CUDA(at::kFloat), {batches, 2, height, width}); // 2: as in argmedian and its 'score'
    CHECK_INPUT(solution);
    
    int result = MediMax::forward( (float*)   ( in_vol.data<float>() ),
                                   (float*)   ( solution.data<float>() ), 
                                   batches, height, width, channel );
    CHECK_INPUT(solution);
    if (result)
    {
      std::cout << "Cuda Function call SKNLSE_id::forward failed";return {};
    }
    //std::cout << "Forward run finished\n";
    return {solution};
  }
  std::cout << "Input/Outout type to cuda operator must be single/float";
  return {};
}

std::vector<at::Tensor> median_backward(
    at::Tensor in_grad,
    at::Tensor in_vol ) {

  //std::cout << "BW -- Input: (its/id):" << its << "/" << id <<"\n";

  //return std::vector<at::Tensor> {unaries, pairwise}; //, 0, 0

  CHECK_INPUT(in_vol);

  const int batches   = in_vol.size(0);
  const int channel   = in_vol.size(3);
  const int height    = in_vol.size(1);
  const int width     = in_vol.size(2);

  //std::cout << "Input: " << batches << " " << channel << " " << height << " " << width <<"\n";

  if ( in_vol.type().scalarType() == at::ScalarType::Float )
  {
    auto outGrad = at::zeros_like( in_vol );

  	int result = MediMax::backward( (float*)   ( in_vol.data<float>() ),
                								    (float*)   ( in_grad.data<float>() ), 
                								    (float*)   ( outGrad.data<float>() ), 
                                     batches, height, width, channel );
    if (result)
    {
      std::cout << "Cuda Function call SKNLSE_id::forward failed";return {};
    }
    return {outGrad};
  }
  std::cout << "Input/Outout type to cuda operator must be single/float";
  return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",    &median_forward,    "Median forward (CUDA)");
  m.def("backward",   &median_backward,   "Median backward (CUDA)");
  m.def("dummy",      &median_dummy,      "Median dummy");
  m.def("dummy_only", &median_dummy_only, "Median dummy2");
}
