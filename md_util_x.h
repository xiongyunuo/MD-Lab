#ifndef MD_UTIL_X_H
#define MD_UTIL_X_H

#include "md_particle_x.h"
#include <stdio.h>

extern int md_pimd_mode_x;
void md_set_pimd_mode_x(int pimd_mode);

#define MD_MAX_NAME_LENGTH_X 1024
#define MD_STRINGIFY2_X(X) #X
#define MD_STRINGIFY_X(X) MD_STRINGIFY2_X(X)

typedef char md_name_t_x[MD_MAX_NAME_LENGTH_X+1];

typedef struct {
  int N;
  md_name_t_x *names;
  md_num_t_x *values;
} md_attr_pair_t_x;

md_attr_pair_t_x *md_alloc_attr_pair_x(void);
md_attr_pair_t_x *md_read_attr_pair_x(FILE *in, md_attr_pair_t_x *attr);
md_num_t_x md_get_attr_value_x(md_attr_pair_t_x *attr, const char *name, int *found);

#ifdef MD_USE_OPENCL_X
extern cl_uint md_num_platforms_x;
extern cl_int *md_num_devices_x;
extern cl_int *md_num_queues_x;
extern cl_platform_id *md_platforms_x;
extern cl_context *md_contexts_x;
extern cl_device_id **md_devices_x;
extern cl_command_queue **md_command_queues_x;
extern cl_device_id **md_command_devices_x;
extern cl_program *md_programs_x;
extern int md_que_cur_i_x, md_que_cur_j_x;

int md_get_command_queue_x(int *i, int *j, cl_context *context, cl_device_id *device, cl_command_queue *queue);
int md_roundup_x(int a, int b);
int md_get_work_size_x(cl_kernel kernel, cl_device_id device, int d, int *size, size_t *global, size_t *local);

#ifdef MD_CL_FAST_MATH_X
#define MD_CL_MATH_FLAGS_X " -cl-fast-relaxed-math"
#else
#define MD_CL_MATH_FLAGS_X ""
#endif

#ifdef MD_DOUBLE_PREC_X
#define MD_CL_COMPILER_OPTIONS_X "-DMD_DOUBLE_PREC_X " "-DMD_DIMENSION_X=" MD_STRINGIFY_X(MD_DIMENSION_X) " -DMD_NHC_LENGTH_X=" MD_STRINGIFY_X(MD_NHC_LENGTH_X) MD_CL_MATH_FLAGS_X
#else
#define MD_CL_COMPILER_OPTIONS_X "-DMD_DIMENSION_X=" MD_STRINGIFY_X(MD_DIMENSION_X) " -DMD_NHC_LENGTH_X=" MD_STRINGIFY_X(MD_NHC_LENGTH_X) MD_CL_MATH_FLAGS_X
#endif

#define MD_MAX_QUEUE_SIZE_X 5000

cl_int md_update_queue_x(cl_command_queue queue);

#endif

#define MD_MIN_X(a, b) (((a)<(b))?(a):(b))

typedef struct {
  int cpu_count;
  int gpu_count;
  int acc_count;
  unsigned int seed;
  const char *cl_path;
} md_init_config_x;

int md_init_setup_x(FILE *out, void *config);

#ifdef MD_USE_OPENCL_X
cl_int md_simulation_to_context_x(md_simulation_t_x *sim, cl_context context);
cl_event md_simulation_sync_queue_x(md_simulation_t_x *sim, cl_command_queue queue, cl_int *err);
cl_int md_simulation_clear_cache_x(md_simulation_t_x *sim);
#endif

md_simulation_t_x *md_simulation_sync_host_x(md_simulation_t_x *sim, int read_only);

md_simulation_t_x *md_alloc_simulation_x(int N, int f, int fc, int box_type);
simulation_box_t_x *md_alloc_box_x(int box_type);
int md_init_particle_face_center_3d_lattice_pos_x(md_simulation_t_x *sim, md_num_t_x *center, md_num_t_x *length, md_num_t_x fluc, md_num_t_x eps);
int md_init_maxwell_vel_x(md_simulation_t_x *sim, md_num_t_x T);
int md_init_particle_mass_x(md_simulation_t_x *sim, md_num_t_x m, int start, int end);
int md_init_nhc_x(md_simulation_t_x *sim, md_num_t_x Q, int start, int end, md_num_t_x fp);
int md_init_nd_rect_box_x(simulation_box_t_x *box, md_num_t_x *L);
int md_fprint_particle_pos_x(FILE *out, md_simulation_t_x *sim);
int md_fprint_particle_vel_x(FILE *out, md_simulation_t_x *sim);
int md_fprint_particle_force_x(FILE *out, md_simulation_t_x *sim);
int md_fprint_nhcs_x(FILE *out, md_simulation_t_x *sim);
int md_fprint_sim_box_x(FILE *out, simulation_box_t_x *box);
void md_set_seed_x(unsigned int seed);
md_num_t_x md_random_uniform_x(md_num_t_x a, md_num_t_x b);
md_num_t_x md_random_gaussian_x(void);
md_num_t_x md_get_quick_Q_x(md_num_t_x T, md_num_t_x omega);
int md_fread_particle_pos_x(FILE *in, md_simulation_t_x *sim);
int md_fread_particle_vel_x(FILE *in, md_simulation_t_x *sim);
int md_fread_nhcs_x(FILE *in, md_simulation_t_x *sim);

md_num_t_x md_laguerre_poly_x(int n, md_num_t_x x);
md_num_t_x md_coulomb_Cn_x(int n, md_num_t_x Z);
md_num_t_x md_coulomb_En_x(int n, md_num_t_x Z, md_num_t_x ke);
md_num_t_x md_r_magnitude_x(md_num_t_x *r);
md_num_t_x md_coulomb_rho_b_x(int nmax, md_num_t_x Z, md_num_t_x ke, md_num_t_x beta, md_num_t_x *r, md_num_t_x *r2);
md_num_t_x md_coulomb_diag_rho_b_x(int nmax, md_num_t_x Z, md_num_t_x ke, md_num_t_x beta, md_num_t_x *r);
md_num_t_x md_coulomb_wave_Ak_x(int k, md_num_t_x eta);
md_num_t_x md_coulomb_wave_Bk_x(int k, md_num_t_x eta, md_num_t_x rho);
md_num_t_x md_coulomb_wave_func_x(md_num_t_x eta, md_num_t_x rho, int kmax);
void md_log_gamma_x(md_num_t_x re, md_num_t_x im, md_num_t_x *res, md_num_t_x *res2);
md_num_t_x md_1F1_integrand_x(md_num_t_x u);
md_num_t_x md_1F1_im_integrand_x(md_num_t_x u);
void md_1F1_x(md_num_t_x a, md_num_t_x ia, md_num_t_x b, md_num_t_x ib, md_num_t_x z, md_num_t_x iz, md_num_t_x *res, md_num_t_x *res2);
md_num_t_x md_coulomb_wave_C_x(md_num_t_x eta);
md_num_t_x md_coulomb_wave_F0_x(md_num_t_x eta, md_num_t_x rho, int kmax);
md_num_t_x md_coulomb_wave_func_star_x(md_num_t_x eta, md_num_t_x rho, int kmax);
md_num_t_x md_coulomb_wave_dF0_x(md_num_t_x eta, md_num_t_x rho, int kmax);
md_num_t_x md_coulomb_wave_integrand_x(md_num_t_x k);
md_num_t_x md_coulomb_wave_diag_integrand_x(md_num_t_x k);
md_num_t_x md_romberg_integral_x(md_num_t_x (*f)(md_num_t_x), md_num_t_x a, md_num_t_x b, int max_steps, md_num_t_x acc);
md_num_t_x md_coulomb_rho_c_x(int kmax, md_num_t_x Z, md_num_t_x ke, md_num_t_x beta, md_num_t_x *r, md_num_t_x *r2);
md_num_t_x md_coulomb_diag_rho_c_x(int kmax, md_num_t_x Z, md_num_t_x ke, md_num_t_x beta, md_num_t_x *r);
md_num_t_x md_coulomb_free_rho_x(md_num_t_x ke, md_num_t_x beta, md_num_t_x *r, md_num_t_x *r2);

typedef md_num_t_x (*md_pa_func_t_x)(md_num_t_x *r1, md_num_t_x *r2, md_num_t_x beta, int N, md_num_t_x *params);

typedef struct {
  md_num_t_x beta, ka, Z;
  int ucount, ducount;
  md_num_t_x *r;
  md_num_t_x *u;
  md_num_t_x *A;
  md_num_t_x *r2;
  md_num_t_x *du;
  md_num_t_x *dA;
  int Npar;
  md_num_t_x *params;
  md_pa_func_t_x pa_func;
} md_pa_table_t_x;

md_num_t_x md_pa_delta_x(md_num_t_x *r1, md_num_t_x *r2, md_num_t_x beta, int N, md_num_t_x *params);

md_pa_table_t_x *md_read_pa_table_x(FILE *in);
md_num_t_x md_calc_pa_pair_energy_x(md_pa_table_t_x *pa, md_num_t_x *r, md_num_t_x *r2);
md_num_t_x md_calc_pa_pair_U_x(md_pa_table_t_x *pa, md_num_t_x *r, md_num_t_x *r2);
void md_calc_pa_deri_1_x(md_pa_table_t_x *pa, md_num_t_x *r, md_num_t_x *r2, md_num_t_x *dr, md_num_t_x rincre);
void md_calc_pa_deri_2_x(md_pa_table_t_x *pa, md_num_t_x *r, md_num_t_x *r2, md_num_t_x *dr, md_num_t_x rincre);

int md_simulation_copy_to_x(md_simulation_t_x *sim, md_simulation_t_x *sim2);

#endif