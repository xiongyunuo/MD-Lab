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
  int pcount_in, pcount_ex;
  md_index_pair_t_x *pair_in;
  md_index_pair_t_x *pair_ex;
  int *Eindices;
  md_num_t_x *minE;
#ifdef MD_USE_OPENCL_X
  cl_mem ENk_mem;
  cl_mem ENk2_mem;
  cl_mem VBN_mem;
  cl_mem pair_in_mem;
  cl_mem pair_ex_mem;
  cl_mem Eindices_mem;
  cl_context context;
  cl_command_queue queue;
  cl_kernel fillE_kernel;
  cl_kernel minE_kernel;
  cl_kernel min_kernel;
  cl_mem minE_mem;
  cl_kernel fillV_kernel;
  cl_kernel addV_kernel;
  cl_kernel fillFV_kernel;
  cl_kernel filleV_kernel;
  cl_kernel addeV_kernel;
  cl_mem eVBN_mem;
  cl_kernel addeVst_kernel;
  cl_kernel ctf_kernel;
  char tfkname[64];
  cl_kernel cte_kernel;
  char tekname[64];
  cl_kernel cpf_kernel;
  cl_kernel upf_kernel;
  char pfkname[64];
  cl_mem forces;
  cl_kernel cpe_kernel;
  char pekname[64];
  cl_kernel cdd_kernel;
  cl_mem den_mem;
  int points;
  cl_kernel ffillE_kernel;
  cl_kernel ffillE2_kernel;
  cl_mem Eint_mem;
  cl_kernel fminE_kernel;
  cl_mem V_mem;
  cl_mem fminE_mem;
  cl_kernel ffillV_kernel;
  cl_kernel faddV_kernel;
  cl_mem G_mem;
  cl_kernel ffillG_kernel;
  cl_kernel ffillFV_kernel;
  cl_kernel ffillFV2_kernel;
  cl_kernel ffillFV3_kernel;
  cl_kernel fillE2_kernel;
  cl_kernel ffillE21_kernel;
  cl_kernel ffillE22_kernel;
  cl_kernel filleV2_kernel;
  cl_kernel addeVst2_kernel;
  cl_kernel cvire_kernel;
  cl_kernel cpf2_kernel;
  cl_kernel upf21_kernel;
  cl_kernel upf22_kernel;
  cl_mem forces2;
  int N2;
  char pf2kname[64];
  cl_kernel cpe2_kernel;
  char pe2kname[64];
  cl_kernel cITCF_kernel;
#endif
} md_pimd_t_x;

#ifdef MD_USE_OPENCL_X
cl_int md_pimd_to_context_x(md_pimd_t_x *pimd, cl_context context);
cl_event md_pimd_sync_queue_x(md_pimd_t_x *pimd, cl_command_queue queue, cl_int *err);
cl_int md_pimd_clear_cache_x(md_pimd_t_x *pimd);
#endif

md_pimd_t_x *md_pimd_sync_host_x(md_pimd_t_x *pimd, int read_only);

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
void md_reset_stats_x(md_stats_t_x *stats, md_num_t_x e);
void md_stats_add_to_x(md_stats_t_x *stats, md_stats_t_x *stats2);
void md_stats_copy_to_x(md_stats_t_x *stats, md_stats_t_x *stats2);
md_num_t_x md_pimd_xminE2_x(md_pimd_t_x *pimd, int N2, md_num_t_x *VBN2, md_num_t_x vi2);
void md_pimd_calc_vi_sign_x(md_pimd_t_x *pimd, md_stats_t_x *e_s, md_stats_t_x *s_s, md_num_t_x vi2);

#endif