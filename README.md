# MD-Lab
Implements PIMD simulation of fictitious identical particles based on the quadratic scaling MD algorithm (https://arxiv.org/abs/2305.18025). No third party libraries apart from OpenCL are used, a working POSIX environment is required for compilation.

OpenCL for parallelized PIMD simulation is now supported (check out the paper https://arxiv.org/abs/2404.02628). To use OpenCL, compile the project with makefiles that end with "_cl" suffix ("_nocl" suffix is for single thread no OpenCL compilation, "_pocl" is for POCL implementation of OpenCL), the system default implementation of OpenCL will be used.

This page only includes the source code of the library itself. For tests and applications to quantum many body systems such as trapped particles and UEG see the release section https://github.com/xiongyunuo/MD-Lab/releases.

Public data folder is for openly accessible results for some of my research papers.
