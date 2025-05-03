#ifndef MD_SIMULATION_X_H
#define MD_SIMULATION_X_H

#include "md_particle_x.h"

#ifdef MD_USE_OPENCL_X

#define MD_MAX_INTER_PARAMETERS_X 15

typedef struct {
  md_num_t_x params[MD_MAX_INTER_PARAMETERS_X];
} md_inter_params_t_x;

#endif

extern md_num_t_x md_trap_frequency_x;
extern md_num_t_x md_trap_center_x[MD_DIMENSION_X];
extern md_num_t_x md_trap_frequency_2_x[MD_DIMENSION_X];
extern md_num_t_x md_gaussian_strength_x;
extern md_num_t_x md_gaussian_range_x;
extern md_num_t_x md_coulomb_strength_x;
extern md_num_t_x md_coulomb_truc_x;
extern int md_coulomb_n_sum_x;
extern md_num_t_x md_hubbard_trap_strength_x;
extern md_num_t_x md_hubbard_trap_frequency_x;
extern md_num_t_x md_he_eps_x;
extern md_num_t_x md_he_A_x;
extern md_num_t_x md_he_alpha_x;
extern md_num_t_x md_he_C6_x;
extern md_num_t_x md_he_C8_x;
extern md_num_t_x md_he_C10_x;
extern md_num_t_x md_he_D_x;
extern md_num_t_x md_he_rm_x;
extern md_num_t_x md_kelbg_beta_x;
extern md_num_t_x md_kelbg_mu_x;
extern int md_kelbg_P_x;

typedef void (*md_pair_force_t_x)(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
typedef void (*md_trap_force_t_x)(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);

md_num_t_x md_minimum_image_x(md_num_t_x a, md_num_t_x L);
md_num_t_x md_minimum_image_distance_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L);
md_num_t_x md_minimum_image_distance_2_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, int *w1, int *w2);
md_num_t_x md_distance_x(md_num_t_x *p1, md_num_t_x *p2);
void md_LJ_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_harmonic_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_neg_harmonic_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_harmonic_trap_frequency_x(md_num_t_x f);
void md_set_harmonic_trap_center_x(md_num_t_x *center);
void md_set_harmonic_trap_frequency_2_x(md_num_t_x *f);
void md_harmonic_trap_force_2_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_periodic_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_gaussian_force_strength_x(md_num_t_x g);
void md_set_guassian_force_range_x(md_num_t_x s);
void md_coulomb_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_coulomb_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_coulomb_force_strength_x(md_num_t_x g);
void md_set_coulomb_truc_x(md_num_t_x k);
void md_set_coulomb_n_sum_x(int n);
md_num_t_x md_calc_3d_ueg_madelung_x(md_num_t_x *L);
void md_coulomb_3d_ewald_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_coulomb_3d_ewald_force_R_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_coulomb_3d_ewald_force_NI_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_hubbard_trap_strength_x(md_num_t_x h);
void md_set_hubbard_trap_frequency_x(md_num_t_x k);
void md_hubbard_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_helium_parameters_x(md_num_t_x eps, md_num_t_x A, md_num_t_x alpha, md_num_t_x C6, md_num_t_x C8, md_num_t_x C10, md_num_t_x D, md_num_t_x rm);
void md_periodic_helium_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_set_kelbg_parameters_x(md_num_t_x beta, md_num_t_x mu, int P);
void md_periodic_kelbg_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);

int md_reset_force_x(md_simulation_t_x *sim);
int md_calc_pair_force_x(md_simulation_t_x *sim, md_pair_force_t_x pf);
void md_calc_nhc_force_x(md_simulation_t_x *sim, int f0, int f2, int i, md_num_t_x *res);
int md_update_nhc_VV3_1_x(md_simulation_t_x *sim, md_num_t_x h);
int md_update_nhc_VV3_2_x(md_simulation_t_x *sim, md_num_t_x h);
int md_periodic_boundary_x(md_simulation_t_x *sim);
int md_periodic_image_count_x(md_num_t_x x, md_num_t_x L);
int md_periodic_boundary_count_x(md_simulation_t_x *sim, int *count);
int md_periodic_boundary_recover_x(md_simulation_t_x *sim, int *count);
int md_simulation_rescale_velocity_x(md_simulation_t_x *sim, md_num_t_x T);

#ifdef MD_USE_OPENCL_X
void md_get_pair_force_info_x(md_pair_force_t_x pf, const char *prefix, const char *postfix, char *dest, md_inter_params_t_x *params);
void md_get_trap_force_info_x(md_trap_force_t_x tf, const char *prefix, const char *postfix, char *dest, md_inter_params_t_x *params);
#endif

#endif