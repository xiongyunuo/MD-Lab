#ifndef MD_PIMD_X_H
#define MD_PIMD_X_H

#include "md_particle_x.h"
#include "md_statistics_x.h"
#include "md_simulation_x.h"
#include <stdio.h>

#define MD_PIMD_INDEX_X(l,j,P) (((l)-1)*(P)+(j)-1)

typedef struct {
  md_simulation_t_x *sim;
  int N, P;
  int ENk_count;
  md_num_t_x **ENk;
  md_num_t_x **ENk2;
  md_num_t_x *VBN;
  md_num_t_x vi;
  md_num_t_x omegaP;
  md_num_t_x beta;
} md_pimd_t_x;

md_pimd_t_x *md_alloc_pimd_x(md_simulation_t_x *sim, int N, int P, md_num_t_x vi, md_num_t_x T);
void md_pimd_change_temperature_x(md_pimd_t_x *pimd, md_num_t_x T);
int md_pimd_next_index_x(int l, int j, int N2, int k, int P);
int md_pimd_prev_index_x(int l, int j, int N2, int k, int P);
void md_pimd_init_particle_uniform_pos_x(md_pimd_t_x *pimd, md_num_t_x *center, md_num_t_x *L, md_num_t_x fluc);
md_num_t_x md_pimd_ENk_x(md_pimd_t_x *pimd, int N2, int k, int image);
void md_pimd_fill_ENk_x(md_pimd_t_x *pimd, int image);
int md_pimd_fprint_ENk_x(FILE *out, md_pimd_t_x *pimd);
md_num_t_x md_pimd_xexp_x(md_num_t_x k, md_num_t_x E, md_num_t_x EE, md_num_t_x beta, md_num_t_x vi);
md_num_t_x md_pimd_xminE_x(md_pimd_t_x *pimd, int N2);
void md_pimd_fill_VB_x(md_pimd_t_x *pimd);
int md_pimd_fprint_VBN_x(FILE *out, md_pimd_t_x *pimd);
void md_pimd_dENk_x(md_pimd_t_x *pimd, int N2, int k, int l, int j, int image, md_num_t_x *dENk);
void md_pimd_fill_force_VB_x(md_pimd_t_x *pimd, int image);
void md_pimd_calc_VBN_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats);
void md_pimd_calc_trap_force_x(md_pimd_t_x *pimd, md_trap_force_t_x tf);
void md_pimd_calc_trap_energy_x(md_pimd_t_x *pimd, md_trap_energy_t_x te, md_stats_t_x *stats);
void md_pimd_calc_pair_force_x(md_pimd_t_x *pimd, md_pair_force_t_x pf);
void md_pimd_calc_pair_energy_x(md_pimd_t_x *pimd, md_pair_energy_t_x pe, md_stats_t_x *stats);
void md_pimd_calc_density_distribution_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmax, int image, md_num_t_x *center);
void md_pimd_fast_fill_ENk_x(md_pimd_t_x *pimd, int image);
md_num_t_x md_pimd_fast_xminE_x(md_pimd_t_x *pimd, int u, md_num_t_x *V);
md_num_t_x md_pimd_fast_xexp_x(md_num_t_x l, md_num_t_x k, md_num_t_x E, md_num_t_x EE, md_num_t_x beta, md_num_t_x vi);
void md_pimd_fast_dENk_x(md_pimd_t_x *pimd, int index, int index2, int index3, int image, md_num_t_x *dENk);
void md_pimd_fast_fill_force_VB_x(md_pimd_t_x *pimd, int image);
void md_pimd_calc_ITCF_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmin, md_num_t_x rmax, int pi);
int md_pimd_calc_pair_force_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_pair_force_t_x pf);
int md_pimd_calc_pair_energy_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_pair_energy_t_x pe, md_stats_t_x *stats);
void md_pimd_calc_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats);
void md_pimd_calc_centroid_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats, int fc);
md_num_t_x md_pimd_ENk2_x(md_pimd_t_x *pimd, int N2, int k, int image);
void md_pimd_fill_ENk2_x(md_pimd_t_x *pimd, int image);
int md_pimd_fprint_ENk2_x(FILE *out, md_pimd_t_x *pimd);
void md_pimd_calc_vi_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats);
void md_pimd_polymer_periodic_boundary_x(md_pimd_t_x *pimd);
md_num_t_x md_pimd_fast_ENk2_x(md_pimd_t_x *pimd, int index, int index2, int index3, int image);
void md_pimd_fast_fill_ENk2_x(md_pimd_t_x *pimd, int image);

#endif