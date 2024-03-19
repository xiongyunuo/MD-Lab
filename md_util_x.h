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
void md_init_maxwell_vel_x(md_simulation_t_x *sim, md_num_t_x T);
void md_init_particle_mass_x(md_simulation_t_x *sim, md_num_t_x m, int start, int end);
void md_init_nhc_x(md_simulation_t_x *sim, md_num_t_x Q, int start, int end, md_num_t_x fp);
void md_init_nd_rect_box_x(simulation_box_t_x *box, md_num_t_x *L);
int md_fprint_particle_pos_x(FILE *out, md_simulation_t_x *sim);
int md_fprint_particle_vel_x(FILE *out, md_simulation_t_x *sim);
int md_fprint_particle_force_x(FILE *out, md_simulation_t_x *sim);
int md_fprint_nhcs_x(FILE *out, md_simulation_t_x *sim);
int md_fprint_sim_box_x(FILE *out, simulation_box_t_x *box);
void md_set_seed_x(unsigned int seed);
md_num_t_x md_random_uniform_x(md_num_t_x a, md_num_t_x b);
md_num_t_x md_random_gaussian_x(void);
md_num_t_x md_get_quick_Q_x(md_num_t_x T, md_num_t_x omega);

#endif