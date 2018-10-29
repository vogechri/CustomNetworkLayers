#import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
#torch.utils.cpp_extension.include_paths(), 

######################################################################
##############  compile with python setup.py install ##############
######################################################################

# note that python MUST use gcc vf version >4.8 ! else this will segfault
# this indeed uses the respective version to compile ! so could manually set here if wanted..
#os.environ["CC"] = "g++-4.9" 
#os.environ["CXX"] = "g++-4.9"

setup(
    name='median_cuda',
    include_dirs=[ os.path.join(os.environ['IMAGEUTILITIES_ROOT'], 'include'), '/home/vogechri/.local/include/'],
    ext_modules=[
        CUDAExtension('median_cuda_ext', [
            'median_cuda.cpp',
            'cudaCode/medimax.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
})
