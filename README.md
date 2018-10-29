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
    
