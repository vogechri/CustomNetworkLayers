# CustomNetworkLayers

###################################################################
#                                                                 #
#        Learning Energy Based Inpainting for Optical Flow        #
#      Christoph Vogel, Patrick Knoebelreiter and Thomas Pock     #
#                          ACCV 2018                              #
#                                                                 #
# Copyright 2018 Graz University of Technology (Christoph Vogel)  #
#                                                                 #
###################################################################

About:
The repository holds several custom network layers. 
Some of which were used in my recent optical flow project: 
Learning Energy Based Inpainting for Optical Flow.

The additional and necessary libraries
 - ImageUtilities
 - CUDA Programming model 
are not included.

To download those packages follow the links:
https://developer.nvidia.com/cuda-zone
https://github.com/VLOGroup/imageutilities
and read the licensing and installation information provided there.

=====================================================================

DISCLAIMER:
This software has been rewritten for the sake of providing an implementation 
in a recent language. Therefore, the results produced by the code may differ
from those presented in the paper [1]. 
Results are also always subject to the training procedure, training set, etc.

=====================================================================

IMPORTANT:
If you use this software you should cite the following in any resulting publication:

    [1] Learning Energy Based Inpainting for Optical Flow
        C. Vogel, P. Knoebelreiter and T. Pock
        In ACCV, Perth, Australia, December 2018


INSTALLING & RUNNING

1.	Download and install PyTorch from https://pytorch.org/
    and similarly acquire ImageUtilities from 
    https://github.com/VLOGroup/imageutilities. 
    Compile ImageUtilities and if desired create a system variable pointing 
    to the installation path 
    of the Image Utilities.

2.	Compile the custom layers by changing to the respective directory
    and execute 'python setup.py install' on the command line.
    Make sure that PyTorch and ImageUtilities are installed. 
    Also verify the installation path to the ImageUtilities library 
    in the file setup.py.
    
CONTENT

1. The folder 'cudaTV' contains the Optimization Layer from the ACCV paper. 
   More precisely this is the TV version. Contained is a PyTorch layer:
   'TVInpainting.py' that can be used directly after compilation succeeds. 
   Note the 'id' parameter. This one ensures that the buffers are assigned to 
   the correct instantiation of the layer. This allows one to have multiple of 
   these layers executed, for instance to build a hierarchical scheme as was 
   done for optical flow computation. For now the ids are limited to 0..9 and 
   each layer should have a different id. With a trivial change in the code (setting
   __uniqueIds__ in TVInpaintFista.h to a different value) one can have more layers.
2. The folder 'cudaTGV' contains the TGV version of the optimization layer 
   used in the ACCV paper.
   
   
