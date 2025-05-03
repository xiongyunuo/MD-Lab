#include "md_worm_x.h"
#include <stdlib.h>

mc_worm_t_x *pimc_alloc_worm_x(md_pimd_t_x *pimd) {
  mc_worm_t_x *res = (mc_worm_t_x *)malloc(sizeof(mc_worm_t_x));
  if (res == NULL)
    return NULL;
  res->pimd = pimd;
  res->comp = (int *)malloc(sizeof(int)*pimd->N);
  if (res->comp == NULL) {
    free(res);
    return NULL;
  }
  int i;
  for (i = 0; i < pimd->N; ++i)
    res->comp[i] = 0;
  res->permu_index = (int *)malloc(sizeof(int)*pimd->N);
  if (res->permu_index == NULL) {
    free(res);
    return NULL;
  }
  for (i = 0; i < pimd->N; ++i)
    res->permu_index[i] = i;
  res->is_worm = (int *)malloc(sizeof(int)*pimd->N);
  if (res->is_worm == NULL) {
    free(res);
    return NULL;
  }
  for (i = 0; i < pimd->N; ++i)
    res->is_worm[i] = 0;
  res->worm_pos = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_DIMENSION_X*pimd->N);
  if (res->worm_pos == NULL) {
    free(res);
    return NULL;
  }
  res->old_bead_pos = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_DIMENSION_X*pimd->N*pimd->P);
  if (res->old_bead_pos == NULL) {
    free(res);
    return NULL;
  }
  res->old_permu_index = (int *)malloc(sizeof(int)*pimd->N);
  if (res->old_permu_index == NULL) {
    free(res);
    return NULL;
  }
  res->old_is_worm = (int *)malloc(sizeof(int)*pimd->N);
  if (res->old_is_worm == NULL) {
    free(res);
    return NULL;
  }
  res->old_worm_pos = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_DIMENSION_X*pimd->N);
  if (res->old_worm_pos == NULL) {
    free(res);
    return NULL;
  }
  res->change = (int *)malloc(sizeof(int)*pimd->N*pimd->P);
  if (res->change == NULL) {
    free(res);
    return NULL;
  }
  res->bead_image_count = (int *)malloc(sizeof(int)*pimd->N*pimd->P*MD_DIMENSION_X);
  if (res->bead_image_count == NULL) {
    free(res);
    return NULL;
  }
  res->worm_image_count = (int *)malloc(sizeof(int)*pimd->N*MD_DIMENSION_X);
  if (res->worm_image_count == NULL) {
    free(res);
    return NULL;
  }
  res->old_bead_image_count = (int *)malloc(sizeof(int)*pimd->N*pimd->P*MD_DIMENSION_X);
  if (res->old_bead_image_count == NULL) {
    free(res);
    return NULL;
  }
  res->old_worm_image_count = (int *)malloc(sizeof(int)*pimd->N*MD_DIMENSION_X);
  if (res->old_worm_image_count == NULL) {
    free(res);
    return NULL;
  }
  return res;
}

int pimc_backup_worm_x(mc_worm_t_x *worm) {
  md_pimd_t_x *pimd = worm->pimd;
  int i;
  for (i = 0; i < pimd->N*pimd->P; ++i)
    worm->change[i] = 0;
  int l, j, index;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        worm->old_bead_pos[index*MD_DIMENSION_X+i] = pimd->sim->particles[index].x[i];
    }
  for (i = 0; i < pimd->N; ++i)
    worm->old_permu_index[i] = worm->permu_index[i];
  for (i = 0; i < pimd->N; ++i)
    worm->old_is_worm[i] = worm->is_worm[i];
  for (i = 0; i < pimd->N*MD_DIMENSION_X; ++i)
    worm->old_worm_pos[i] = worm->worm_pos[i];
  return 0;
}

int pimc_restore_worm_x(mc_worm_t_x *worm) {
  md_pimd_t_x *pimd = worm->pimd;
  int i;
  for (i = 0; i < pimd->N*pimd->P; ++i)
    worm->change[i] = 0;
  int l, j, index;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] = worm->old_bead_pos[index*MD_DIMENSION_X+i];
    }
  for (i = 0; i < pimd->N; ++i)
    worm->permu_index[i] = worm->old_permu_index[i];
  for (i = 0; i < pimd->N; ++i)
    worm->is_worm[i] = worm->old_is_worm[i];
  for (i = 0; i < pimd->N*MD_DIMENSION_X; ++i)
    worm->worm_pos[i] = worm->old_worm_pos[i];
  return 0;
}

int pimc_recenter_worm_x(mc_worm_t_x *worm) {
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int l, j, i;
  for (l = 0; l < pimd->N; ++l) {
    int index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      int count = md_periodic_image_count_x(pimd->sim->particles[index].x[i], L[i]);
      if (count != 0) {
        for (j = 0; j < pimd->P; ++j) {
          int index2 = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
          pimd->sim->particles[index2].x[i] += count*L[i];
        }
        if (worm->is_worm[l])
          worm->worm_pos[l*MD_DIMENSION_X+i] += count*L[i];
      }
    }
  }
  return 0;
}

int pimc_recalc_pair_energy_x(mc_worm_t_x *worm, md_pair_energy_t_x pe, md_stats_t_x *stats) {
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int l, j, i;
  int l2, index, index2;
  for (l = 0; l < pimd->N; ++l) {
    if (!worm->is_worm[l]) {
      index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        worm->worm_pos[l*MD_DIMENSION_X+i] = pimd->sim->particles[index].x[i];
    }
    if (!worm->old_is_worm[l]) {
      index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        worm->old_worm_pos[l*MD_DIMENSION_X+i] = worm->old_bead_pos[index*MD_DIMENSION_X+i];
    }
  }
  for (l = 0; l < pimd->N; ++l)
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      if (worm->change[index]) {
        for (l2 = 0; l2 < pimd->N; ++l2) {
          if (l == l2)
            continue;
          index2 = MD_PIMD_INDEX_X(l2+1, j+1, pimd->P);
          if (worm->change[index2] && l2 < l)
            continue;
          if (j == 0) {
            stats->es[0] -= 0.5*pe(&worm->old_bead_pos[index*MD_DIMENSION_X], &worm->old_bead_pos[index2*MD_DIMENSION_X], L, pimd->sim->particles[index].m)/pimd->P;
            stats->es[0] -= 0.5*pe(&worm->old_worm_pos[l*MD_DIMENSION_X], &worm->old_worm_pos[l2*MD_DIMENSION_X], L, pimd->sim->particles[index].m)/pimd->P;
          }
          else
            stats->es[0] -= pe(&worm->old_bead_pos[index*MD_DIMENSION_X], &worm->old_bead_pos[index2*MD_DIMENSION_X], L, pimd->sim->particles[index].m)/pimd->P;
        }
      }
    }
  for (l = 0; l < pimd->N; ++l)
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      if (worm->change[index]) {
        for (l2 = 0; l2 < pimd->N; ++l2) {
          if (l == l2)
            continue;
          index2 = MD_PIMD_INDEX_X(l2+1, j+1, pimd->P);
          if (worm->change[index2] && l2 < l)
            continue;
          if (j == 0) {
            stats->es[0] += 0.5*pe(pimd->sim->particles[index].x, pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
            stats->es[0] += 0.5*pe(&worm->worm_pos[l*MD_DIMENSION_X], &worm->worm_pos[l2*MD_DIMENSION_X], L, pimd->sim->particles[index].m)/pimd->P;
          }
          else
            stats->es[0] += pe(pimd->sim->particles[index].x, pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        }
      }
    }
  return 0;
}

int pimc_periodic_box_x(mc_worm_t_x *worm) {
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int l, j, i;
  int index;
  for (l = 0; l < pimd->N; ++l)
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        worm->bead_image_count[index*MD_DIMENSION_X+i] = md_periodic_image_count_x(pimd->sim->particles[index].x[i], L[i]);
        pimd->sim->particles[index].x[i] += worm->bead_image_count[index*MD_DIMENSION_X+i]*L[i];
      }
    }
  for (l = 0; l < pimd->N; ++l) {
    if (worm->is_worm[l]) {
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        worm->worm_image_count[l*MD_DIMENSION_X+i] = md_periodic_image_count_x(worm->worm_pos[l*MD_DIMENSION_X+i], L[i]);
        worm->worm_pos[l*MD_DIMENSION_X+i] += worm->worm_image_count[l*MD_DIMENSION_X+i]*L[i];
      }
    }
  }
  for (l = 0; l < pimd->N; ++l)
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        worm->old_bead_image_count[index*MD_DIMENSION_X+i] = md_periodic_image_count_x(worm->old_bead_pos[index*MD_DIMENSION_X+i], L[i]);
        worm->old_bead_pos[index*MD_DIMENSION_X+i] += worm->old_bead_image_count[index*MD_DIMENSION_X+i]*L[i];
      }
    }
  for (l = 0; l < pimd->N; ++l) {
    if (worm->old_is_worm[l]) {
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        worm->old_worm_image_count[l*MD_DIMENSION_X+i] = md_periodic_image_count_x(worm->old_worm_pos[l*MD_DIMENSION_X+i], L[i]);
        worm->old_worm_pos[l*MD_DIMENSION_X+i] += worm->old_worm_image_count[l*MD_DIMENSION_X+i]*L[i];
      }
    }
  }
  return 0;
}

int pimc_restore_periodic_box_x(mc_worm_t_x *worm) {
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int l, j, i;
  int index;
  for (l = 0; l < pimd->N; ++l)
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        pimd->sim->particles[index].x[i] -= worm->bead_image_count[index*MD_DIMENSION_X+i]*L[i];
      }
    }
  for (l = 0; l < pimd->N; ++l) {
    if (worm->is_worm[l]) {
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        worm->worm_pos[l*MD_DIMENSION_X+i] -= worm->worm_image_count[l*MD_DIMENSION_X+i]*L[i];
      }
    }
  }
  for (l = 0; l < pimd->N; ++l)
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        worm->old_bead_pos[index*MD_DIMENSION_X+i] -= worm->old_bead_image_count[index*MD_DIMENSION_X+i]*L[i];
      }
    }
  for (l = 0; l < pimd->N; ++l) {
    if (worm->old_is_worm[l]) {
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        worm->old_worm_pos[l*MD_DIMENSION_X+i] -= worm->old_worm_image_count[l*MD_DIMENSION_X+i]*L[i];
      }
    }
  }
  return 0;
}

int pimc_has_worm_x(mc_worm_t_x *worm) {
  int l;
  for (l = 0; l < worm->pimd->N; ++l) {
    if (worm->is_worm[l])
      return 1;
  }
  return 0;
}

int pimc_calc_virial_energy_x(mc_worm_t_x *worm, md_stats_t_x *stats) {
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  stats->es[stats->N-1] += MD_DIMENSION_X*pimd->N/(2*pimd->beta);
  md_num_t_x res2 = 0;
  int l, j, i;
  int index, index2;
  md_simulation_t_x *sim = pimd->sim;
  for (j = 2; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        res2 += -(sim->particles[index].x[i]-sim->particles[index2].x[i])*sim->particles[index].f[i]*sim->particles[index].m;
      }
    }
  stats->es[stats->N-1] += res2/2.0;
  for (l = 1; l <= pimd->N; ++l) {
    index = MD_PIMD_INDEX_X(l, 1, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
    int l2 = worm->permu_index[l-1];
    int index3 = MD_PIMD_INDEX_X(l2+1, 1, pimd->P);
    int image_count[MD_DIMENSION_X];
    for (i = 0; i < MD_DIMENSION_X; ++i)
      image_count[i] = 0;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      image_count[i] = -md_periodic_image_count_x(pimd->sim->particles[index2].x[i], L[i]);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index3].x[i] += image_count[i]*L[i];
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index2].x[i] > L[i]/2) {
        pimd->sim->particles[index3].x[i] -= L[i];
        --image_count[i];
      }
      else if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index2].x[i] < -L[i]/2) {
        pimd->sim->particles[index3].x[i] += L[i];
        ++image_count[i];
      }
    }
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (index3 == index)
        stats->es[stats->N-1] += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*(pimd->sim->particles[index2].x[i]-pimd->sim->particles[index3].x[i])*(pimd->sim->particles[index3].x[i]-(pimd->sim->particles[index].x[i]-image_count[i]*L[i]));
      else
        stats->es[stats->N-1] += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*(pimd->sim->particles[index2].x[i]-pimd->sim->particles[index3].x[i])*(pimd->sim->particles[index3].x[i]-(pimd->sim->particles[index].x[i]));
    }
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index3].x[i] -= image_count[i]*L[i];
  }
  return 0;
}

int pimc_translate_worm_x(mc_worm_t_x *worm, md_num_t_x max_trans) {
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  md_num_t_x trans[MD_DIMENSION_X];
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    md_num_t_x tmp = MD_MIN_X(max_trans, L[i]/2.0);
    trans[i] = md_random_uniform_x(0, tmp);
  }
  int start = rand()%pimd->N;
  int l = start;
  int index, j;
  while (1) {
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      worm->change[index] = 1;
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] += trans[i];
    }
    if (worm->is_worm[l]) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        worm->worm_pos[l*MD_DIMENSION_X+i] += trans[i];
    }
    l = worm->permu_index[l];
    if (l == start)
      break;
  }
  pimc_recenter_worm_x(worm);
  return 0;
}

/*int pimc_min_image_worm_x(mc_worm_t_x *worm, int l) {
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int i;
  int index = MD_PIMD_INDEX_X(l+1, pimd->P, pimd->P);
  int image_count[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    worm->worm_pos[l*MD_DIMENSION_X+i] += md_periodic_image_count_x(worm->worm_pos[l*MD_DIMENSION_X+i], L[i])*L[i];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count[i] = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count[i] = -md_periodic_image_count_x(pimd->sim->particles[index].x[i], L[i]);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    worm->worm_pos[l*MD_DIMENSION_X+i] += image_count[i]*L[i];
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (worm->worm_pos[l*MD_DIMENSION_X+i]-pimd->sim->particles[index].x[i] > L[i]/2) {
      worm->worm_pos[l*MD_DIMENSION_X+i] -= L[i];
      --image_count[i];
    }
    else if (worm->worm_pos[l*MD_DIMENSION_X+i]-pimd->sim->particles[index].x[i] < -L[i]/2) {
      worm->worm_pos[l*MD_DIMENSION_X+i] += L[i];
      ++image_count[i];
    }
  }
  return 0;
}*/

static int pimc_redraw_sl = -1;
static int pimc_redraw_sj = -1;
//static int pimc_min_worm = 0;

int pimc_redraw_worm_x(mc_worm_t_x *worm, int max_len) {
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int startl = rand()%pimd->N;
  int startj = rand()%pimd->P;
  if (pimc_redraw_sl >= 0)
    startl = pimc_redraw_sl;
  if (pimc_redraw_sj >= 0)
    startj = pimc_redraw_sj;
  int length = 0;
  int curl = startl;
  int curj = startj;
  int i;
  int image_count[MD_DIMENSION_X*5];
  int count = 0;
  md_num_t_x end_pos[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count[i] = 0;
  while (1) {
    if (curj == pimd->P-1) {
      if (worm->is_worm[curl]) {
        for (i = 0; i < MD_DIMENSION_X; ++i)
          worm->worm_pos[curl*MD_DIMENSION_X+i] += image_count[count*MD_DIMENSION_X+i]*L[i];
        //if (pimc_min_worm)
          //pimc_min_image_worm_x(worm, curl);
        for (i = 0; i < MD_DIMENSION_X; ++i)
          end_pos[i] = worm->worm_pos[curl*MD_DIMENSION_X+i];
        ++length;
        break;
      }
      else {
        int tmpl = worm->permu_index[curl];
        int tmpj = 0;
        if (tmpl != startl || tmpj != startj) {
          ++count;
          int index = MD_PIMD_INDEX_X(curl+1, curj+1, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            image_count[count*MD_DIMENSION_X+i] = -md_periodic_image_count_x(pimd->sim->particles[index].x[i], L[i]);
          int index2 = MD_PIMD_INDEX_X(tmpl+1, tmpj+1, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            pimd->sim->particles[index2].x[i] += image_count[count*MD_DIMENSION_X+i]*L[i];
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (pimd->sim->particles[index2].x[i]-pimd->sim->particles[index].x[i] > L[i]/2) {
              pimd->sim->particles[index2].x[i] -= L[i];
              --image_count[count*MD_DIMENSION_X+i];
            }
            else if (pimd->sim->particles[index2].x[i]-pimd->sim->particles[index].x[i] < -L[i]/2) {
              pimd->sim->particles[index2].x[i] += L[i];
              ++image_count[count*MD_DIMENSION_X+i];
            }
          }
          /*for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image_count[count*MD_DIMENSION_X+i]) {
              printf("Another image3\n");
            }
          }*/
          ++length;
          if (length >= max_len) {
            int index = MD_PIMD_INDEX_X(tmpl+1, tmpj+1, pimd->P);
            for (i = 0; i < MD_DIMENSION_X; ++i)
              end_pos[i] = pimd->sim->particles[index].x[i];
            break;
          }
          curl = tmpl;
          curj = tmpj;
        }
        else {
          int index = MD_PIMD_INDEX_X(curl+1, curj+1, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            end_pos[i] = pimd->sim->particles[index].x[i];
          break;
        }
      }
    }
    else {
      int tmpl = curl;
      int tmpj = curj+1;
      if (tmpl != startl || tmpj != startj) {
        int index = MD_PIMD_INDEX_X(tmpl+1, tmpj+1, pimd->P);
        for (i = 0; i < MD_DIMENSION_X; ++i)
          pimd->sim->particles[index].x[i] += image_count[count*MD_DIMENSION_X+i]*L[i];
        ++length;
        if (length >= max_len) {
          int index = MD_PIMD_INDEX_X(tmpl+1, tmpj+1, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            end_pos[i] = pimd->sim->particles[index].x[i];
          break;
        }
        curl = tmpl;
        curj = tmpj;
      }
      else {
        int index = MD_PIMD_INDEX_X(curl+1, curj+1, pimd->P);
        for (i = 0; i < MD_DIMENSION_X; ++i)
          end_pos[i] = pimd->sim->particles[index].x[i];
        break;
      }
    }
  }
  md_num_t_x j1 = length;
  length = 0;
  curl = startl;
  curj = startj;
  while (1) {
    if (curj == pimd->P-1) {
      if (worm->is_worm[curl]) {
        ++length;
        break;
      }
      else {
        int tmpl = worm->permu_index[curl];
        int tmpj = 0;
        if (tmpl != startl || tmpj != startj) {
          ++length;
          if (length >= max_len) {
            break;
          }
          int index = MD_PIMD_INDEX_X(tmpl+1, tmpj+1, pimd->P);
          int index2 = MD_PIMD_INDEX_X(curl+1, curj+1, pimd->P);
          md_num_t_x aj = (j1-length)/(j1-length+1);
          md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            md_num_t_x r2 = (end_pos[i]+(j1-length)*pimd->sim->particles[index2].x[i])/(j1-length+1);
            pimd->sim->particles[index].x[i] = r2+sqrt(sig)*md_random_gaussian_x();
          }
          worm->change[index] = 1;
          curl = tmpl;
          curj = tmpj;
        }
        else {
          break;
        }
      }
    }
    else {
      int tmpl = curl;
      int tmpj = curj+1;
      if (tmpl != startl || tmpj != startj) {
        ++length;
        if (length >= max_len) {
          break;
        }
        int index = MD_PIMD_INDEX_X(tmpl+1, tmpj+1, pimd->P);
        int index2 = MD_PIMD_INDEX_X(curl+1, curj+1, pimd->P);
        md_num_t_x aj = (j1-length)/(j1-length+1);
        md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          md_num_t_x r2 = (end_pos[i]+(j1-length)*pimd->sim->particles[index2].x[i])/(j1-length+1);
          pimd->sim->particles[index].x[i] = r2+sqrt(sig)*md_random_gaussian_x();
        }
        worm->change[index] = 1;
        curl = tmpl;
        curj = tmpj;
      }
      else {
        break;
      }
    }
  }
  length = 0;
  curl = startl;
  curj = startj;
  count = 0;
  while (1) {
    if (curj == pimd->P-1) {
      if (worm->is_worm[curl]) {
        for (i = 0; i < MD_DIMENSION_X; ++i)
          worm->worm_pos[curl*MD_DIMENSION_X+i] -= image_count[count*MD_DIMENSION_X+i]*L[i];
        //if (pimc_min_worm)
          //pimc_min_image_worm_x(worm, curl);
        ++length;
        break;
      }
      else {
        int tmpl = worm->permu_index[curl];
        int tmpj = 0;
        if (tmpl != startl || tmpj != startj) {
          ++count;
          int index2 = MD_PIMD_INDEX_X(tmpl+1, tmpj+1, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            pimd->sim->particles[index2].x[i] -= image_count[count*MD_DIMENSION_X+i]*L[i];
          ++length;
          if (length >= max_len) {
            break;
          }
          curl = tmpl;
          curj = tmpj;
        }
        else {
          break;
        }
      }
    }
    else {
      int tmpl = curl;
      int tmpj = curj+1;
      if (tmpl != startl || tmpj != startj) {
        int index = MD_PIMD_INDEX_X(tmpl+1, tmpj+1, pimd->P);
        for (i = 0; i < MD_DIMENSION_X; ++i)
          pimd->sim->particles[index].x[i] -= image_count[count*MD_DIMENSION_X+i]*L[i];
        ++length;
        if (length >= max_len) {
          break;
        }
        curl = tmpl;
        curj = tmpj;
      }
      else {
        break;
      }
    }
  }
  pimc_recenter_worm_x(worm);
  return 0;
}

int pimc_open_worm_x(mc_worm_t_x *worm, md_num_t_x C, md_stats_t_x *acc, int j0) {
  int max_len = worm->pimd->P-j0;
  acc->es[0] = 0;
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int i;
  int l = rand()%pimd->N+1;
  if (worm->is_worm[l-1])
    return 0;
  int index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
  int l2 = worm->permu_index[l-1];
  int index3 = MD_PIMD_INDEX_X(l2+1, 1, pimd->P);
  int image_count[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count[i] = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count[i] = -md_periodic_image_count_x(pimd->sim->particles[index2].x[i], L[i]);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index3].x[i] += image_count[i]*L[i];
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index2].x[i] > L[i]/2) {
      pimd->sim->particles[index3].x[i] -= L[i];
      --image_count[i];
    }
    else if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index2].x[i] < -L[i]/2) {
      pimd->sim->particles[index3].x[i] += L[i];
      ++image_count[i];
    }
  }
  int index = MD_PIMD_INDEX_X(l, pimd->P-(max_len-1), pimd->P);
  md_num_t_x start[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (index == index3)
      start[i] = pimd->sim->particles[index].x[i]-image_count[i]*L[i];
    else
      start[i] = pimd->sim->particles[index].x[i];
  }
  md_num_t_x dis = md_distance_x(start, pimd->sim->particles[index3].x);
  md_num_t_x del[MD_DIMENSION_X];
  md_num_t_x aj = max_len;
  md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
  worm->is_worm[l-1] = 1;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    del[i] = MD_MIN_X(sqrt(sig), L[i]/2.0);
    worm->worm_pos[(l-1)*MD_DIMENSION_X+i] = pimd->sim->particles[index3].x[i]+md_random_uniform_x(-del[i],del[i]);
  }
  //pimc_min_image_worm_x(worm, l-1);
  md_num_t_x dis2 = md_distance_x(start, &worm->worm_pos[(l-1)*MD_DIMENSION_X]);
  md_num_t_x mult = 1.0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    mult *= 2*del[i]/L[i];
  acc->es[0] = C*pimd->N*mult*exp(-(dis2*dis2-dis*dis)/(2*sig));
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index3].x[i] -= image_count[i]*L[i];
  pimc_redraw_sl = l-1;
  pimc_redraw_sj = pimd->P-(max_len-1)-1;
  //pimc_min_worm = 0;
  pimc_redraw_worm_x(worm, max_len);
  pimc_redraw_sl = -1;
  pimc_redraw_sj = -1;
  //pimc_min_worm = 1;
  worm->change[index3] = 1;
  index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
  worm->change[index3] = 1;
  return 0;
}

int pimc_close_worm_x(mc_worm_t_x *worm, md_num_t_x C, md_stats_t_x *acc, int j0) {
  int max_len = worm->pimd->P-j0;
  acc->es[0] = 0;
  md_pimd_t_x *pimd = worm->pimd;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int i;
  int l;
  for (l = 1; l <= pimd->N; ++l) {
    if (worm->is_worm[l-1])
      break;
  }
  if (l > pimd->N)
    return 0;
  //int index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
  int l2 = worm->permu_index[l-1];
  int index3 = MD_PIMD_INDEX_X(l2+1, 1, pimd->P);
  int image_count[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count[i] = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count[i] = -md_periodic_image_count_x(worm->worm_pos[(l-1)*MD_DIMENSION_X+i], L[i]);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index3].x[i] += image_count[i]*L[i];
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (pimd->sim->particles[index3].x[i]-worm->worm_pos[(l-1)*MD_DIMENSION_X+i] > L[i]/2) {
      pimd->sim->particles[index3].x[i] -= L[i];
      --image_count[i];
    }
    else if (pimd->sim->particles[index3].x[i]-worm->worm_pos[(l-1)*MD_DIMENSION_X+i] < -L[i]/2) {
      pimd->sim->particles[index3].x[i] += L[i];
      ++image_count[i];
    }
  }
  /*for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (image_count[i]) {
      printf("Another image\n");
    }
  }*/
  int index = MD_PIMD_INDEX_X(l, pimd->P-(max_len-1), pimd->P);
  md_num_t_x start[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (index == index3)
      start[i] = pimd->sim->particles[index].x[i]-image_count[i]*L[i];
    else
      start[i] = pimd->sim->particles[index].x[i];
  }
  //pimc_min_image_worm_x(worm, l-1);
  md_num_t_x dis = md_distance_x(start, &worm->worm_pos[(l-1)*MD_DIMENSION_X]);
  md_num_t_x del[MD_DIMENSION_X];
  md_num_t_x aj = max_len;
  md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    del[i] = MD_MIN_X(sqrt(sig), L[i]/2.0);
    if (fabs(pimd->sim->particles[index3].x[i]-worm->worm_pos[(l-1)*MD_DIMENSION_X+i]) > del[i]) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index3].x[i] -= image_count[i]*L[i];
      return 0;
    }
  }
  /*for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (image_count[i]) {
      printf("Another image2\n");
    }
  }*/
  md_num_t_x dis2 = md_distance_x(start, pimd->sim->particles[index3].x);
  md_num_t_x mult = 1.0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    mult *= 2*del[i]/L[i];
  acc->es[0] = (1.0/(C*pimd->N*mult))*exp(-(dis2*dis2-dis*dis)/(2*sig));
  /*for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (image_count[i]) {
      printf("Another image2 %f\n", acc->es[0]);
    }
  }*/
  for (i = 0; i < MD_DIMENSION_X; ++i)
    worm->worm_pos[(l-1)*MD_DIMENSION_X+i] = pimd->sim->particles[index3].x[i];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index3].x[i] -= image_count[i]*L[i];
  pimc_redraw_sl = l-1;
  pimc_redraw_sj = pimd->P-(max_len-1)-1;
  pimc_redraw_worm_x(worm, max_len);
  pimc_redraw_sl = -1;
  pimc_redraw_sj = -1;
  worm->is_worm[l-1] = 0;
  worm->change[index3] = 1;
  index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
  worm->change[index3] = 1;
  return 0;
}

int pimc_move_head_worm_x(mc_worm_t_x *worm, int max_len) {
  md_pimd_t_x *pimd = worm->pimd;
  int i;
  int l;
  for (l = 1; l <= pimd->N; ++l) {
    if (worm->is_worm[l-1])
      break;
  }
  if (l > pimd->N)
    return 0;
  int index = MD_PIMD_INDEX_X(l, pimd->P-(max_len-1), pimd->P);
  md_num_t_x aj = max_len;
  md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    worm->worm_pos[(l-1)*MD_DIMENSION_X+i] = pimd->sim->particles[index].x[i]+sqrt(sig)*md_random_gaussian_x();
  //pimc_min_image_worm_x(worm, l-1);
  pimc_redraw_sl = l-1;
  pimc_redraw_sj = pimd->P-(max_len-1)-1;
  //pimc_min_worm = 0;
  pimc_redraw_worm_x(worm, max_len);
  pimc_redraw_sl = -1;
  pimc_redraw_sj = -1;
  //pimc_min_worm = 1;
  index = MD_PIMD_INDEX_X(l, 1, pimd->P);
  worm->change[index] = 1;
  return 0;
}

int pimc_move_tail_worm_x(mc_worm_t_x *worm, int max_len) {
  md_pimd_t_x *pimd = worm->pimd;
  int i;
  int l;
  for (l = 1; l <= pimd->N; ++l) {
    if (worm->is_worm[l-1])
      break;
  }
  if (l > pimd->N)
    return 0;
  l = worm->permu_index[l-1]+1;
  int index = MD_PIMD_INDEX_X(l, 1+max_len, pimd->P);
  int index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
  md_num_t_x aj = max_len;
  md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index2].x[i] = pimd->sim->particles[index].x[i]+sqrt(sig)*md_random_gaussian_x();
  pimc_recenter_worm_x(worm);
  pimc_redraw_sl = l-1;
  pimc_redraw_sj = 0;
  pimc_redraw_worm_x(worm, max_len);
  pimc_redraw_sl = -1;
  pimc_redraw_sj = -1;
  index = MD_PIMD_INDEX_X(l, 1, pimd->P);
  worm->change[index] = 1;
  return 0;
}

int pimc_count_permu_worm_x(mc_worm_t_x *worm) {
  int res = 0;
  md_pimd_t_x *pimd = worm->pimd;
  int *visit = (int *)malloc(sizeof(int)*pimd->N);
  int l;
  for (l = 0; l < pimd->N; ++l)
    visit[l] = 0;
  for (l = 0; l < pimd->N; ++l) {
    if (visit[l])
      continue;
    int start = l;
    int cur = l;
    while (1) {
      cur = worm->permu_index[cur];
      visit[cur] = 1;
      if (cur == start)
        break;
      ++res;
    }
  }
  free(visit);
  return res;
}

int pimc_swap_worm_x(mc_worm_t_x *worm, md_stats_t_x *acc, int jP) {
  acc->es[0] = 0;
  md_pimd_t_x *pimd = worm->pimd;
  if (pimd->vi == 0.0)
    return 0;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int i;
  int l;
  for (l = 1; l <= pimd->N; ++l) {
    if (worm->is_worm[l-1])
      break;
  }
  if (l > pimd->N)
    return 0;
  int lT = worm->permu_index[l-1]+1;
  int image_count[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count[i] = md_periodic_image_count_x(worm->worm_pos[(l-1)*MD_DIMENSION_X+i], L[i]);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    worm->worm_pos[(l-1)*MD_DIMENSION_X+i] += image_count[i]*L[i];
  int index, l2;
  md_num_t_x *prob = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->N);
  md_num_t_x aj = jP;
  md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
  for (l2 = 1; l2 <= pimd->N; ++l2) {
    if (worm->comp[l-1] != worm->comp[l2-1]) {
      prob[l2-1] = 0;
      continue;
    }
    index = MD_PIMD_INDEX_X(l2, jP+1, pimd->P);
    md_num_t_x dis = md_distance_x(&worm->worm_pos[(l-1)*MD_DIMENSION_X], pimd->sim->particles[index].x);
    prob[l2-1] = exp(-dis*dis/(2*sig));
  }
  md_num_t_x sigP = 0;
  for (l2 = 1; l2 <= pimd->N; ++l2)
    sigP += prob[l2-1];
  for (l2 = 1; l2 <= pimd->N; ++l2)
    prob[l2-1] /= sigP;
  md_num_t_x sum = 0;
  md_num_t_x r = md_random_uniform_x(0, 1);
  int l0;
  for (l0 = 1; l0 <= pimd->N; ++l0) {
    if (worm->comp[l-1] != worm->comp[l0-1])
      continue;
    sum += prob[l0-1];
    if (sum >= r)
      break;
  }
  if (l0 > pimd->N)
    return 0;
  if (l0 == lT) {
    for (i = 0; i < MD_DIMENSION_X; ++i)
      worm->worm_pos[(l-1)*MD_DIMENSION_X+i] -= image_count[i]*L[i];
    free(prob);
    return 0;
  }
  md_num_t_x sig0 = 0;
  int index2 = MD_PIMD_INDEX_X(l0, 1, pimd->P);
  for (l2 = 1; l2 <= pimd->N; ++l2) {
    if (worm->comp[l-1] != worm->comp[l2-1])
      continue;
    index = MD_PIMD_INDEX_X(l2, jP+1, pimd->P);
    md_num_t_x dis = md_distance_x(pimd->sim->particles[index2].x, pimd->sim->particles[index].x);
    sig0 += exp(-dis*dis/(2*sig));
  }
  int l3;
  for (l3 = 1; l3 <= pimd->N; ++l3) {
    if (worm->permu_index[l3-1] == l0-1)
      break;
  }
  int index3 = MD_PIMD_INDEX_X(l0, 1, pimd->P);
  index2 = MD_PIMD_INDEX_X(l3, pimd->P, pimd->P);
  int image_count2[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count2[i] = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    image_count2[i] = -md_periodic_image_count_x(pimd->sim->particles[index2].x[i], L[i]);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index3].x[i] += image_count2[i]*L[i];
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index2].x[i] > L[i]/2) {
      pimd->sim->particles[index3].x[i] -= L[i];
      --image_count2[i];
    }
    else if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index2].x[i] < -L[i]/2) {
      pimd->sim->particles[index3].x[i] += L[i];
      ++image_count2[i];
    }
  }
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index3].x[i] -= image_count2[i]*L[i];
  md_num_t_x start[MD_DIMENSION_X];
  index3 = MD_PIMD_INDEX_X(l0, 1, pimd->P);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    start[i] = pimd->sim->particles[index3].x[i]+image_count2[i]*L[i];
  index2 = MD_PIMD_INDEX_X(l0, 1, pimd->P);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index2].x[i] = worm->worm_pos[(l-1)*MD_DIMENSION_X+i];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    worm->worm_pos[(l-1)*MD_DIMENSION_X+i] -= image_count[i]*L[i];
  worm->is_worm[l-1] = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    worm->worm_pos[(l3-1)*MD_DIMENSION_X+i] = start[i];
  worm->is_worm[l3-1] = 1;
  worm->change[index2] = 1;
  index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
  worm->change[index2] = 1;
  int old_permu_count = pimc_count_permu_worm_x(worm);
  for (l2 = 1; l2 <= pimd->N; ++l2) {
    if (worm->permu_index[l2-1] == l0-1) {
      worm->permu_index[l2-1] = worm->permu_index[l-1];
      break;
    }
  }
  worm->permu_index[l-1] = l0-1;
  int new_permu_count = pimc_count_permu_worm_x(worm);
  acc->es[0] = pow(pimd->vi, new_permu_count-old_permu_count)*sigP/sig0;
  //printf("acc %f\n", acc->es[0]);
  pimc_redraw_sl = l0-1;
  pimc_redraw_sj = 0;
  pimc_redraw_worm_x(worm, jP);
  pimc_redraw_sl = -1;
  pimc_redraw_sj = -1;
  free(prob);
  return 0;
}