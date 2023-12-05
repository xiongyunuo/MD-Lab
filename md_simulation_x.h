#ifndef MD_SIMULATION_X_H
#define MD_SIMULATION_X_H

#include "md_particle_x.h"

extern md_num_t_x md_trap_frequency_x;
extern md_num_t_x md_trap_center_x[MD_DIMENSION_X];
extern md_num_t_x md_gaussian_strength_x;
extern md_num_t_x md_gaussian_range_x;
extern md_num_t_x md_coulomb_strength_x;
extern md_num_t_x md_coulomb_truc_x;
extern int md_coulomb_n_sum_x;

typedef void (*md_pair_force_t_x)(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
typedef void (*md_trap_force_t_x)(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);

md_num_t_x md_minimum_image_x(md_num_t_x a, md_num_t_x L);
md_num_t_x md_minimum_image_distance_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L);
md_num_t_x md_distance_x(md_num_t_x *p1, md_num_t_x *p2);
void md_LJ_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_harmonic_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_harmonic_trap_frequency_x(md_num_t_x f);
void md_set_harmonic_trap_center_x(md_num_t_x *center);
void md_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_gaussian_force_strength_x(md_num_t_x g);
void md_set_guassian_force_range_x(md_num_t_x s);
void md_coulomb_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_coulomb_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_coulomb_force_strength_x(md_num_t_x g);
void md_set_coulomb_truc_x(md_num_t_x k);
void md_set_coulomb_n_sum_x(int n);
md_num_t_x md_calc_3d_ueg_madelung_x(md_num_t_x *L);
void md_coulomb_3d_ewald_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);

void md_reset_force_x(md_simulation_t_x *sim);
void md_calc_pair_force_x(md_simulation_t_x *sim, md_pair_force_t_x pf);
void md_calc_nhc_force_x(md_simulation_t_x *sim, int f0, int f2, int i, md_num_t_x *res);
void md_update_nhc_VV3_1_x(md_simulation_t_x *sim, md_num_t_x h);
void md_update_nhc_VV3_2_x(md_simulation_t_x *sim, md_num_t_x h);
void md_periodic_boundary_x(md_simulation_t_x *sim);
int md_periodic_image_count_x(md_num_t_x x, md_num_t_x L);
void md_periodic_boundary_count_x(md_simulation_t_x *sim, int *count);
void md_periodic_boundary_recover_x(md_simulation_t_x *sim, int *count);

#endif