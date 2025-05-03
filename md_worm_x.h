#ifndef MD_WORM_X_H
#define MD_WORM_X_H

#include "md_particle_x.h"
#include "md_statistics_x.h"
#include "md_simulation_x.h"
#include "md_util_x.h"
#include "md_pimd_x.h"
#include <stdio.h>

typedef struct {
  md_pimd_t_x *pimd;
  int *permu_index;
  int *comp;
  int *is_worm;
  md_num_t_x *worm_pos;
  md_num_t_x *old_bead_pos;
  int *old_permu_index;
  int *old_is_worm;
  md_num_t_x *old_worm_pos;
  int *change;
  int *bead_image_count;
  int *worm_image_count;
  int *old_bead_image_count;
  int *old_worm_image_count;
} mc_worm_t_x;

mc_worm_t_x *pimc_alloc_worm_x(md_pimd_t_x *pimd);
int pimc_backup_worm_x(mc_worm_t_x *worm);
int pimc_restore_worm_x(mc_worm_t_x *worm);
int pimc_recenter_worm_x(mc_worm_t_x *worm);
int pimc_periodic_box_x(mc_worm_t_x *worm);
int pimc_restore_periodic_box_x(mc_worm_t_x *worm);
int pimc_has_worm_x(mc_worm_t_x *worm);
int pimc_calc_virial_energy_x(mc_worm_t_x *worm, md_stats_t_x *stats);
int pimc_recalc_pair_energy_x(mc_worm_t_x *worm, md_pair_energy_t_x pe, md_stats_t_x *stats);
int pimc_translate_worm_x(mc_worm_t_x *worm, md_num_t_x max_trans);
//int pimc_min_image_worm_x(mc_worm_t_x *worm, int l);
int pimc_redraw_worm_x(mc_worm_t_x *worm, int max_len);
int pimc_open_worm_x(mc_worm_t_x *worm, md_num_t_x C, md_stats_t_x *acc, int j0);
int pimc_close_worm_x(mc_worm_t_x *worm, md_num_t_x C, md_stats_t_x *acc, int j0);
int pimc_move_head_worm_x(mc_worm_t_x *worm, int max_len);
int pimc_move_tail_worm_x(mc_worm_t_x *worm, int max_len);
int pimc_count_permu_worm_x(mc_worm_t_x *worm);
int pimc_swap_worm_x(mc_worm_t_x *worm, md_stats_t_x *acc, int jP);

#endif