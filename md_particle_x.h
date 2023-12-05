#ifndef MD_PARTICLE_X_H
#define MD_PARTICLE_X_H

#define MD_DIMENSION_X 3
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

typedef double md_num_t_x;

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
  int N;
  md_particle_t_x *particles;
  int fc;
  int Nf;
  int *fs;
  md_nhc_t_x *nhcs;
  simulation_box_t_x *box;
  md_num_t_x T;
} md_simulation_t_x;

#endif