#ifndef MD_STATISTICS_X_H
#define MD_STATISTICS_X_H

#include "md_particle_x.h"
#include "md_simulation_x.h"

#define MD_MAX_STATS_COUNT 10000

typedef struct {
  int N;
  int count;
  md_num_t_x e;
  md_num_t_x *es;
  md_num_t_x T;
  md_num_t_x *Ts;
  int points;
  md_num_t_x *fx;
  md_num_t_x **fxs;
#ifdef MD_USE_OPENCL_X
  cl_mem e_mem;
  cl_mem T_mem;
  cl_mem fx_mem;
  cl_context context;
  cl_command_queue queue;
#endif
} md_stats_t_x;

typedef struct {
  int N;
  md_num_t_x *qs;
  md_num_t_x *qis;
} md_sk_sphere_t_x;

md_sk_sphere_t_x *md_alloc_sk_sphere_3d_x(md_num_t_x q0, md_num_t_x qincre, md_num_t_x qmax);

#ifdef MD_USE_OPENCL_X
cl_int md_stats_to_context_x(md_stats_t_x *stats, cl_context context);
#endif

md_stats_t_x *md_stats_sync_host_x(md_stats_t_x *stats);

typedef md_num_t_x (*md_pair_energy_t_x)(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
typedef md_num_t_x (*md_trap_energy_t_x)(md_num_t_x *p1, md_num_t_x *L, md_num_t_x m);

md_stats_t_x *md_alloc_stats_x(int points);
md_stats_t_x *md_update_stats_x(md_stats_t_x *stats);
md_stats_t_x *md_finalize_stats_x(md_stats_t_x *stats);
md_num_t_x md_LJ_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_harmonic_trap_energy_x(md_num_t_x *p1, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_neg_harmonic_trap_energy_x(md_num_t_x *p1, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_harmonic_trap_energy_2_x(md_num_t_x *p1, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_periodic_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_3d_ewald_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_3d_ewald_energy_R_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_3d_ewald_energy_R2_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_3d_ewald_energy_NI_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_hubbard_trap_energy_x(md_num_t_x *p1, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_periodic_helium_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_periodic_kelbg_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_periodic_kelbg_beta_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);

typedef struct {
  int mode;
  int N;
  md_num_t_x *r;
  md_num_t_x *u;
} md_potential_table_t_x;

extern md_potential_table_t_x *md_cur_ptable_x;

md_potential_table_t_x *md_create_potential_table_x(md_num_t_x rmax, md_num_t_x dr, md_pair_energy_t_x pe);
md_potential_table_t_x *md_create_trilinear_potential_table_x(md_num_t_x rmax, int N, md_pair_energy_t_x pe);
void md_set_potential_table_x(md_potential_table_t_x *table);
md_num_t_x md_ptable_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_periodic_ptable_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
void md_ptable_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);
void md_periodic_ptable_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m);

int md_calc_pair_energy_x(md_simulation_t_x *sim, md_pair_energy_t_x pe, md_stats_t_x *stats);
int md_calc_temperature_x(md_simulation_t_x *sim, md_stats_t_x *stats);
int md_calc_kinetic_x(md_simulation_t_x *sim, md_stats_t_x *stats);
int md_calc_nhc_kinetic_x(md_simulation_t_x *sim, md_stats_t_x *stats);
int md_calc_pair_correlation_x(md_simulation_t_x *sim, md_stats_t_x *stats, md_num_t_x rmax, int image);
void md_normalize_distribution_x(int N, md_num_t_x *dis, md_num_t_x rmax, md_num_t_x norm);
void md_normalize_2d_distribution_x(int N, md_num_t_x *dis, md_num_t_x rmin, md_num_t_x rmax);
md_num_t_x md_2d_fourier_transform_x(int N, md_num_t_x *dis, md_num_t_x rmin, md_num_t_x rmax, md_num_t_x q);
md_num_t_x md_calc_structure_factor_x(int N, md_num_t_x *dis, md_num_t_x rmax, md_num_t_x den, md_num_t_x q);
int md_calc_Sk_structure_x(md_simulation_t_x *sim, md_stats_t_x *stats, md_num_t_x q0, md_num_t_x qincre);

#ifdef MD_USE_OPENCL_X
void md_get_pair_energy_info_x(md_pair_energy_t_x pe, const char *prefix, const char *postfix, char *dest, md_inter_params_t_x *params);
void md_get_trap_energy_info_x(md_trap_energy_t_x te, const char *prefix, const char *postfix, char *dest, md_inter_params_t_x *params);
#endif

#endif