#include "md_statistics_x.h"
#include "md_simulation_x.h"
#include <stdlib.h>
#include <math.h>

md_stats_t_x *md_alloc_stats_x(int points) {
  md_stats_t_x *stats = (md_stats_t_x *)malloc(sizeof(md_stats_t_x));
  if (stats == NULL)
    return NULL;
  stats->N = 1;
  stats->count = 0;
  stats->es = (md_num_t_x *)malloc(sizeof(md_num_t_x));
  if (stats->es == NULL) {
    free(stats);
    return NULL;
  }
  stats->es[0] = 0;
  stats->Ts = (md_num_t_x *)malloc(sizeof(md_num_t_x));
  if (stats->Ts == NULL) {
    free(stats);
    return NULL;
  }
  stats->Ts[0] = 0;
  stats->points = points;
  stats->fx = (md_num_t_x *)malloc(sizeof(md_num_t_x)*points);
  if (stats->fx == NULL) {
    free(stats);
    return NULL;
  }
  stats->fxs = (md_num_t_x **)malloc(sizeof(md_num_t_x *));
  if (stats->fxs == NULL) {
    free(stats);
    return NULL;
  }
  stats->fxs[0] = (md_num_t_x *)malloc(sizeof(md_num_t_x)*points);
  if (stats->fxs[0] == NULL) {
    free(stats);
    return NULL;
  }
  int i;
  for (i = 0; i < points; ++i)
    stats->fxs[0][i] = 0;
  return stats;
}

md_stats_t_x *md_update_stats_x(md_stats_t_x *stats) {
  stats->count++;
  if (stats->count >= MD_MAX_STATS_COUNT) {
    stats->es[stats->N-1] /= MD_MAX_STATS_COUNT;
    stats->Ts[stats->N-1] /= MD_MAX_STATS_COUNT;
    int i;
    for (i = 0; i < stats->points; ++i)
      stats->fxs[stats->N-1][i] /= MD_MAX_STATS_COUNT;
    stats->N++;
    stats->count = 0;
    stats->es = (md_num_t_x *)realloc(stats->es, sizeof(md_num_t_x)*stats->N);
    if (stats->es == NULL)
      return NULL;
    stats->es[stats->N-1] = 0;
    stats->Ts = (md_num_t_x *)realloc(stats->Ts, sizeof(md_num_t_x)*stats->N);
    if (stats->Ts == NULL)
      return NULL;
    stats->Ts[stats->N-1] = 0;
    stats->fxs = (md_num_t_x **)realloc(stats->fxs, sizeof(md_num_t_x *)*stats->N);
    if (stats->fxs == NULL)
      return NULL;
    stats->fxs[stats->N-1] = (md_num_t_x *)malloc(sizeof(md_num_t_x)*stats->points);
    if (stats->fxs[stats->N-1] == NULL)
      return NULL;
    for (i = 0; i < stats->points; ++i)
      stats->fxs[stats->N-1][i] = 0;
  }
  return stats;
}

void md_finalize_stats_x(md_stats_t_x *stats) {
  int i, j;
  stats->e = 0;
  stats->T = 0;
  for (i = 0; i < stats->points; ++i)
    stats->fx[i] = 0;
  if (stats->count != 0) {
    stats->es[stats->N-1] /= stats->count;
    stats->Ts[stats->N-1] /= stats->count;
    for (i = 0; i < stats->points; ++i)
      stats->fxs[stats->N-1][i] /= stats->count;
  }
  else
    stats->N--;
  for (i = 0; i < stats->N; ++i) {
    stats->e += stats->es[i];
    stats->T += stats->Ts[i];
    for (j = 0; j < stats->points; ++j)
      stats->fx[j] += stats->fxs[i][j];
  }
  stats->e /= stats->N;
  stats->T /= stats->N;
  for (j = 0; j < stats->points; ++j)
    stats->fx[j] /= stats->N;
}

md_num_t_x md_LJ_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m)) {
  md_num_t_x r = md_minimum_image_distance_x(p1, p2, L);
  return 4*(pow(r, -12)-pow(r, -6));
}

md_num_t_x md_harmonic_trap_energy_x(md_num_t_x *p1, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m) {
  md_num_t_x d = md_distance_x(p1, md_trap_center_x);
  return 0.5*m*md_trap_frequency_x*md_trap_frequency_x*d*d;
}

md_num_t_x md_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m)) {
  md_num_t_x d = md_distance_x(p1, p2);
  md_num_t_x mult = (md_gaussian_strength_x/(M_PI*md_gaussian_range_x*md_gaussian_range_x))*exp(-d*d/(md_gaussian_range_x*md_gaussian_range_x));
  return mult;
}

md_num_t_x md_coulomb_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m)) {
  md_num_t_x d = md_distance_x(p1, p2);
  md_num_t_x mult = md_coulomb_strength_x/d;
  return mult;
}

md_num_t_x md_coulomb_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m)) {
  md_num_t_x d = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x mult = md_coulomb_strength_x/d;
  return mult;
}

md_num_t_x md_coulomb_3d_ewald_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m)) {
  if (MD_DIMENSION_X != 3)
    return 0;
  md_num_t_x V = L[0]*L[1]*L[2];
  md_num_t_x res = 0;
  md_num_t_x origin[3] = { 0, 0, 0 };
  md_num_t_x g[3], r[3];
  md_num_t_x G, R;
  int i, j, k;
  for (i = -md_coulomb_n_sum_x; i <= md_coulomb_n_sum_x; ++i)
    for (j = -md_coulomb_n_sum_x; j <= md_coulomb_n_sum_x; ++j)
      for (k = -md_coulomb_n_sum_x; k <= md_coulomb_n_sum_x; ++k) {
        if (i == 0 && j == 0 && k == 0)
          continue;
        g[0] = i/L[0];
        g[1] = j/L[1];
        g[2] = k/L[2];
        G = md_distance_x(g, origin);
        md_num_t_x dot = 2*M_PI*(g[0]*(p1[0]-p2[0])+g[1]*(p1[1]-p2[1])+g[2]*(p1[2]-p2[2]));
        res += cos(dot)*pow(G,-2)*exp(-M_PI*M_PI*G*G/(md_coulomb_truc_x*md_coulomb_truc_x))/(V*M_PI);
      }
  res -= M_PI/(md_coulomb_truc_x*md_coulomb_truc_x*V);
  for (i = -md_coulomb_n_sum_x; i <= md_coulomb_n_sum_x; ++i)
    for (j = -md_coulomb_n_sum_x; j <= md_coulomb_n_sum_x; ++j)
      for (k = -md_coulomb_n_sum_x; k <= md_coulomb_n_sum_x; ++k) {
        r[0] = p2[0]+i*L[0];
        r[1] = p2[1]+j*L[1];
        r[2] = p2[2]+k*L[2];
        R = md_distance_x(p1, r);
        res += erfc(md_coulomb_truc_x*R)/R;
      }
  return md_coulomb_strength_x*res;
}

void md_calc_pair_energy_x(md_simulation_t_x *sim, md_pair_energy_t_x pe, md_stats_t_x *stats) {
  int i, j;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < sim->N; ++j) {
      if (i != j)
        stats->es[stats->N-1] += 0.5*pe(sim->particles[i].x, sim->particles[j].x, L, sim->particles[i].m);
    }
}

void md_calc_temperature_x(md_simulation_t_x *sim, md_stats_t_x *stats) {
  int f = 0;
  int i, j;
  for (i = 0; i < sim->Nf; ++i)
    f += sim->fs[i];
  md_num_t_x sum = 0;
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j)
      sum += sim->particles[i].m*sim->particles[i].v[j]*sim->particles[i].v[j];
  stats->Ts[stats->N-1] += sum/f;
}

void md_calc_pair_correlation_x(md_simulation_t_x *sim, md_stats_t_x *stats, md_num_t_x rmax, int image) {
  int i, j;
  md_num_t_x *L = NULL;
  md_num_t_x incre = rmax/stats->points;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < sim->N; ++j) {
      if (i != j) {
        md_num_t_x r;
        if (image)
          r = md_minimum_image_distance_x(sim->particles[i].x, sim->particles[j].x, L);
        else
          r = md_distance_x(sim->particles[i].x, sim->particles[j].x);
        int index = (int)(r/incre);
        if (index >= stats->points)
          index = stats->points-1;
        stats->fxs[stats->N-1][index] += 1.0/sim->N;
      }
    }
}

void md_normalize_distribution_x(int N, md_num_t_x *dis, md_num_t_x rmax, md_num_t_x norm) {
  int i;
  md_num_t_x incre = rmax/N;
  md_num_t_x sum = 0;
  for (i = 0; i < N; ++i)
    sum += dis[i]*incre;
  for (i = 0; i < N; ++i)
    dis[i] /= sum;
  for (i = 0; i < N; ++i)
    dis[i] /= norm*pow((i+0.5)*incre, MD_DIMENSION_X-1);
}

void md_normalize_2d_distribution_x(int N, md_num_t_x *dis, md_num_t_x rmin, md_num_t_x rmax) {
  int n = (int)sqrt(N);
  md_num_t_x incre = (rmax-rmin)/n;
  int i, j;
  md_num_t_x sum = 0;
  for (i = 0; i < n-1; ++i)
    for (j = 0; j < n-1; ++j) {
      md_num_t_x tmp = 0;
      tmp += dis[i*n+j];
      tmp += dis[(i+1)*n+j];
      tmp += dis[i*n+j+1];
      tmp += dis[(i+1)*n+j+1];
      tmp /= 4;
      sum += tmp*incre*incre;
    }
  for (i = 0; i < N; ++i)
    dis[i] /= sum;
}

md_num_t_x md_2d_fourier_transform_x(int N, md_num_t_x *dis, md_num_t_x rmin, md_num_t_x rmax, md_num_t_x q) {
  int n = (int)sqrt(N);
  md_num_t_x incre = (rmax-rmin)/n;
  int i, j;
  md_num_t_x sum = 0;
  md_num_t_x x, y;
  for (i = 0; i < n-1; ++i)
    for (j = 0; j < n-1; ++j) {
      md_num_t_x tmp = 0;
      x = rmin+i*incre;
      y = rmin+j*incre;
      tmp += dis[i*n+j]*cos(q*(x-y));
      x = rmin+(i+1)*incre;
      y = rmin+j*incre;
      tmp += dis[(i+1)*n+j]*cos(q*(x-y));
      x = rmin+i*incre;
      y = rmin+(j+1)*incre;
      tmp += dis[i*n+j+1]*cos(q*(x-y));
      x = rmin+(i+1)*incre;
      y = rmin+(j+1)*incre;
      tmp += dis[(i+1)*n+j+1]*cos(q*(x-y));
      tmp /= 4;
      sum += tmp*incre*incre;
    }
  return sum;
}