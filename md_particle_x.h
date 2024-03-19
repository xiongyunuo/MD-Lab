#ifndef MD_PARTICLE_X_H
#define MD_PARTICLE_X_H

#define MD_DIMENSION_X 2
#define MD_NHC_LENGTH_X 8
#define MD_kB_X 1.0
#define MD_hBar_X 1.0

#include <math.h>

#ifndef M_PI
#define M_PI 3.141592653589793
//23846264338327950288
#endif

#ifdef __GNUC__
  #define MD_UNUSED_X(x) UNUSED_ ## x __attribute__((__unused__))
#else
  #define MD_UNUSED_X(x) UNUSED_ ## x
#endif

#ifdef MD_USE_OPENCL_X
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif
#ifdef MD_DOUBLE_PREC_X
typedef cl_double md_num_t_x;
#else
typedef cl_float md_num_t_x;
#endif
#else
typedef double md_num_t_x;
#endif

typedef struct {
  md_num_t_x m;
  md_num_t_x x[MD_DIMENSION_X];
  md_num_t_x v[MD_DIMENSION_X];
  md_num_t_x f[MD_DIMENSION_X];
} md_particle_t_x;

typedef struct {
  int f;
  md_num_t_x theta[MD_NHC_LENGTH_X];
  md_num_t_x vtheta[MD_NHC_LENGTH_X];
  md_num_t_x Q[MD_NHC_LENGTH_X];
} md_nhc_t_x;

enum { MD_ND_RECT_BOX_X };

typedef struct {
  int type;
  void *box;
} simulation_box_t_x;

typedef struct {
  md_num_t_x L[MD_DIMENSION_X];
} nd_rect_t_x;

typedef struct {
  int i, j;
} md_index_pair_t_x;

typedef struct {
  int N;
  md_particle_t_x *particles;
  int fc;
  int Nf;
  int *fs;
  int *f0s;
  md_nhc_t_x *nhcs;
  simulation_box_t_x *box;
  md_num_t_x T;
  int pcount_in, pcount_ex;
  md_index_pair_t_x *pair_in;
  md_index_pair_t_x *pair_ex;
#ifdef MD_USE_OPENCL_X
  cl_mem particles_mem;
  cl_mem fs_mem;
  cl_mem f0s_mem;
  cl_mem nhcs_mem;
  cl_mem pair_in_mem;
  cl_mem pair_ex_mem;
  cl_context context;
  cl_command_queue queue;
  cl_kernel rf_kernel;
  cl_kernel cpf_kernel;
  char kname[64];
  cl_mem forces;
  cl_kernel cpf_kernel2;
  cl_kernel uVV1_kernel;
  cl_kernel uVV2_kernel;
  cl_kernel pb_kernel;
  cl_kernel cpe_kernel;
  char kname2[64];
  cl_kernel add_kernel;
  cl_kernel ct_kernel;
  cl_kernel cpc_kernel;
  cl_kernel aden_kernel;
  int points;
  cl_mem pc_mem;
#endif
} md_simulation_t_x;

#endif