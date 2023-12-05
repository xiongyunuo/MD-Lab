#ifndef MD_STATISTICS_X_H
#define MD_STATISTICS_X_H

#include "md_particle_x.h"

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
} md_stats_t_x;

typedef md_num_t_x (*md_pair_energy_t_x)(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
typedef md_num_t_x (*md_trap_energy_t_x)(md_num_t_x *p1, md_num_t_x *L, md_num_t_x m);

md_stats_t_x *md_alloc_stats_x(int points);
md_stats_t_x *md_update_stats_x(md_stats_t_x *stats);
void md_finalize_stats_x(md_stats_t_x *stats);
md_num_t_x md_LJ_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_harmonic_trap_energy_x(md_num_t_x *p1, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
md_num_t_x md_coulomb_3d_ewald_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x m);
void md_calc_pair_energy_x(md_simulation_t_x *sim, md_pair_energy_t_x pe, md_stats_t_x *stats);
void md_calc_temperature_x(md_simulation_t_x *sim, md_stats_t_x *stats);
void md_calc_pair_correlation_x(md_simulation_t_x *sim, md_stats_t_x *stats, md_num_t_x rmax, int image);
void md_normalize_distribution_x(int N, md_num_t_x *dis, md_num_t_x rmax, md_num_t_x norm);
void md_normalize_2d_distribution_x(int N, md_num_t_x *dis, md_num_t_x rmin, md_num_t_x rmax);
md_num_t_x md_2d_fourier_transform_x(int N, md_num_t_x *dis, md_num_t_x rmin, md_num_t_x rmax, md_num_t_x q);

#endif