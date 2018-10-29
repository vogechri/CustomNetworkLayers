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
The repository holds several custom network layers. Some of which were used in my recent optical flow project: Learning Energy Based Inpainting for Optical Flow.

The additional and optional library
 - ImageUtilities
is not included.

To download that package follow the link:
https://github.com/VLOGroup/imageutilities
and read the licensing information provided there.


==========================================================================
DISCLAIMER:
This software has been rewritten for the sake of providing an implementation 
in a recent language. Therefore, the results produced by the code may differ
from those presented in the paper [1]. 
Results are also always subject to the training procedure, training set, etc.
==========================================================================

IMPORTANT:
If you use this software you should cite the following in any resulting publication:

    [1] An Evaluation of Data Costs for Optical Flow
        C. Vogel, S. Roth and K. Schindler
        In GCPR, Saarbruecken, Germany, September 2013


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
    
