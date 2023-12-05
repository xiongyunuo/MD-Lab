#ifndef MD_UTIL_X_H
#define MD_UTIL_X_H

#include "md_particle_x.h"
#include <stdio.h>

#define MD_MAX_NAME_LENGTH_X 1024
#define MD_STRINGIFY2_X(X) #X
#define MD_STRINGIFY_X(X) MD_STRINGIFY2_X(X)

typedef char md_name_t_x[MD_MAX_NAME_LENGTH_X+1];

typedef struct {
  int N;
  md_name_t_x *names;
  md_num_t_x *values;
} md_attr_pair_t_x;

md_attr_pair_t_x *md_alloc_attr_pair_x();
md_attr_pair_t_x *md_read_attr_pair_x(FILE *in, md_attr_pair_t_x *attr);
md_num_t_x md_get_attr_value_x(md_attr_pair_t_x *attr, const char *name, int *found);

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
md_num_t_x md_random_gaussian_x();
md_num_t_x md_get_quick_Q_x(md_num_t_x T, md_num_t_x omega);

#endif