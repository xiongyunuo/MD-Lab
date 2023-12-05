#include "md_pimd_x.h"
#include "md_util_x.h"
#include "md_simulation_x.h"
#include <stdlib.h>
#include <math.h>

md_pimd_t_x *md_alloc_pimd_x(md_simulation_t_x *sim, int N, int P, md_num_t_x vi, md_num_t_x T) {
  md_pimd_t_x *pimd = (md_pimd_t_x *)malloc(sizeof(md_pimd_t_x));
  if (pimd == NULL)
    return NULL;
  pimd->sim = sim;
  pimd->N = N;
  pimd->P = P;
  pimd->vi = vi;
  pimd->sim->T = T;
  pimd->beta = 1.0/(MD_kB_X*T);
  pimd->omegaP = sqrt((md_num_t_x)P)/(pimd->beta*MD_hBar_X);
  pimd->VBN = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(N+1));
  if (pimd->VBN == NULL) {
    free(pimd);
    return NULL;
  }
  pimd->ENk = (md_num_t_x **)malloc(sizeof(md_num_t_x *)*N);
  if (pimd->ENk == NULL) {
    free(pimd);
    return NULL;
  }
  int count = 0;
  int i;
  for (i = 1; i <= N; ++i)
    count += i;
  pimd->ENk_count = count;
  pimd->ENk[0] = (md_num_t_x *)malloc(sizeof(md_num_t_x)*count);
  if (pimd->ENk[0] == NULL) {
    free(pimd);
    return NULL;
  }
  count = 1;
  for (i = 2; i <= N; ++i) {
    pimd->ENk[i-1] = &pimd->ENk[0][count];
    count += i;
  }
  pimd->ENk2 = (md_num_t_x **)malloc(sizeof(md_num_t_x *)*N);
  if (pimd->ENk2 == NULL) {
    free(pimd);
    return NULL;
  }
  pimd->ENk2[0] = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->ENk_count);
  if (pimd->ENk2[0] == NULL) {
    free(pimd);
    return NULL;
  }
  count = 1;
  for (i = 2; i <= N; ++i) {
    pimd->ENk2[i-1] = &pimd->ENk2[0][count];
    count += i;
  }
  return pimd;
}

void md_pimd_change_temperature_x(md_pimd_t_x *pimd, md_num_t_x T) {
  pimd->sim->T = T;
  pimd->beta = 1.0/(MD_kB_X*T);
  pimd->omegaP = sqrt((md_num_t_x)pimd->P)/(pimd->beta*MD_hBar_X);
}

int md_pimd_next_index_x(int l, int j, int N2, int k, int P) {
  int N;
  if (j == P) {
    if (l == N2) {
      N = N2-k+1;
      return MD_PIMD_INDEX_X(N,1,P);
    }
    else {
      N = l+1;
      return MD_PIMD_INDEX_X(N,1,P);
    }
  }
  int j2 = j+1;
  return MD_PIMD_INDEX_X(l,j2,P);
}

int md_pimd_prev_index_x(int l, int j, int N2, int k, int P) {
  int N;
  if (j == 1) {
    if (l == N2-k+1)
      return MD_PIMD_INDEX_X(N2,P,P);
    else {
      N = l-1;
      return MD_PIMD_INDEX_X(N,P,P);
    }
  }
  int j2 = j-1;
  return MD_PIMD_INDEX_X(l,j2,P);
}

void md_pimd_init_particle_uniform_pos_x(md_pimd_t_x *pimd, md_num_t_x *center, md_num_t_x *L, md_num_t_x fluc) {
  int i, j, k;
  for (i = 0; i < pimd->N; ++i) {
    md_num_t_x pos[MD_DIMENSION_X];
    for (k = 0; k < MD_DIMENSION_X; ++k)
      pos[k] = md_random_uniform_x(center[k]-L[k]/2, center[k]+L[k]/2);
    for (j = 0; j < pimd->P; ++j) {
      int index = MD_PIMD_INDEX_X(i+1, j+1, pimd->P);
      for (k = 0; k < MD_DIMENSION_X; ++k)
        pimd->sim->particles[index].x[k] = pos[k]+fluc*(md_random_uniform_x(0,2)-1);
    }
  }
}

md_num_t_x md_pimd_ENk_x(md_pimd_t_x *pimd, int N2, int k, int image) {
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, j;
  int index, index2;
  md_num_t_x res = 0;
  md_num_t_x d;
  for (l = N2-k+1; l <= N2; ++l)
    for (j = 1; j <= pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = md_pimd_next_index_x(l, j, N2, k, pimd->P);
      if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      res += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }
  return res;
}

void md_pimd_fill_ENk_x(md_pimd_t_x *pimd, int image) {
  int l, j;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= l; ++j)
      pimd->ENk[l-1][j-1] = md_pimd_ENk_x(pimd, l, j, image);
}

int md_pimd_fprint_ENk_x(FILE *out, md_pimd_t_x *pimd) {
  fprintf(out, "ENk %d\n", pimd->N);
  int l, j;
  for (l = 1; l <= pimd->N; ++l) {
    for (j = 1; j <= l; ++j)
      fprintf(out, "%f ", pimd->ENk[l-1][j-1]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

md_num_t_x md_pimd_xexp_x(md_num_t_x k, md_num_t_x E, md_num_t_x EE, md_num_t_x beta, md_num_t_x vi) {
  if (vi == 0.0)
    return exp(-beta*E+EE);
  else
    return exp((k-1)*log(vi)-beta*E+EE);
}

md_num_t_x md_pimd_xminE_x(md_pimd_t_x *pimd, int N2) {
  if (pimd->vi == 0.0)
    return pimd->beta*(pimd->ENk[N2-1][0]+pimd->VBN[N2-1]);
  int k;
  md_num_t_x res = 1e10;
  for (k = 1; k <= N2; ++k) {
    md_num_t_x tmp = -(k-1)*log(pimd->vi)+pimd->beta*(pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k]);
    if (tmp < res)
      res = tmp;
  }
  return res;
}

void md_pimd_fill_VB_x(md_pimd_t_x *pimd) {
  int N2, k;
  pimd->VBN[0] = 0.0;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    md_num_t_x sum = 0;
    md_num_t_x tmp = md_pimd_xminE_x(pimd, N2);
    for (k = 1; k <= N2; ++k) {
      if (pimd->vi == 0 && k-1 != 0)
        continue;
      sum += md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
    }
    pimd->VBN[N2] = (tmp-log(sum)+log(N2))/pimd->beta;
  }
}

int md_pimd_fprint_VBN_x(FILE *out, md_pimd_t_x *pimd) {
  fprintf(out, "VBN %d\n", pimd->N);
  int i;
  for (i = 0; i <= pimd->N; ++i)
    fprintf(out, "%f ", pimd->VBN[i]);
  fprintf(out, "\n");
  return ferror(out);
}

void md_pimd_dENk_x(md_pimd_t_x *pimd, int N2, int k, int l, int j, int image, md_num_t_x *dENk) {
  if (!(l >= N2-k+1 && l <= N2)) {
    int i;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      dENk[i] = 0;
    return;
  }
  int index = MD_PIMD_INDEX_X(l, j, pimd->P);
  int index2 = md_pimd_next_index_x(l, j, N2, k, pimd->P);
  int index3 = md_pimd_prev_index_x(l, j, N2, k, pimd->P);
  int i;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    dENk[i] = 0;
    if (image) {
      dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
      dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index3].x[i], L[i]);
    }
    else {
      dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*(sim->particles[index].x[i]-sim->particles[index2].x[i]);
      dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*(sim->particles[index].x[i]-sim->particles[index3].x[i]);
    }
  }
}

void md_pimd_fill_force_VB_x(md_pimd_t_x *pimd, int image) {
  int N2, k, l, j;
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_DIMENSION_X*(pimd->N+1));
  md_num_t_x *grad = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_DIMENSION_X*pimd->N);
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->P; ++j) {
      int i;
      for (i = 0; i < MD_DIMENSION_X; ++i)
        res[i] = 0;
      for (N2 = 1; N2 <= pimd->N; ++N2) {
        md_num_t_x sum2 = 0;
        md_num_t_x tmp = md_pimd_xminE_x(pimd, N2);
        for (k = 1; k <= N2; ++k) {
          if (pimd->vi == 0 && k-1 != 0)
            continue;
          sum2 += md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
        }
        for (k = 1; k <= N2; ++k)
          md_pimd_dENk_x(pimd, N2, k, l, j, image, &grad[MD_DIMENSION_X*(k-1)]);
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          md_num_t_x sum = 0;
          for (k = 1; k <= N2; ++k) {
            if (pimd->vi == 0 && k-1 != 0)
              continue;
            sum += (grad[MD_DIMENSION_X*(k-1)+i]+res[MD_DIMENSION_X*(N2-k)+i])*md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
          }
          res[MD_DIMENSION_X*N2+i] = sum/sum2;
        }
      }
      int index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].f[i] += -res[MD_DIMENSION_X*pimd->N+i]/pimd->sim->particles[index].m;
    }
  free(res);
  free(grad);
}

void md_pimd_calc_VBN_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats) {
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  res[0] = 0;
  int N2, k;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    md_num_t_x sum2 = 0;
    md_num_t_x tmp = md_pimd_xminE_x(pimd, N2);
    for (k = 1; k <= N2; ++k) {
      if (pimd->vi == 0 && k-1 != 0)
        continue;
      sum2 += md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
    }
    md_num_t_x sum = 0;
    for (k = 1; k <= N2; ++k) {
      if (pimd->vi == 0 && k-1 != 0)
        continue;
      sum += (res[N2-k]-pimd->ENk[N2-1][k-1])*md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
    }
    res[N2] = sum/sum2;
  }
  stats->es[stats->N-1] += res[pimd->N];
  //stats->es[stats->N-1] += pimd->sim->fc/(2*pimd->beta);
  stats->es[stats->N-1] += MD_DIMENSION_X*pimd->N*pimd->P/(2*pimd->beta);
  free(res);
}

void md_pimd_calc_trap_force_x(md_pimd_t_x *pimd, md_trap_force_t_x tf) {
  int l, j;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  md_num_t_x f[MD_DIMENSION_X];
  int i;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->P; ++j) {
      int index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        f[i] = 0;
      tf(sim->particles[index].x, f, L, sim->particles[index].m);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        sim->particles[index].f[i] += f[i]/pimd->P;
    }
}

void md_pimd_calc_trap_energy_x(md_pimd_t_x *pimd, md_trap_energy_t_x te, md_stats_t_x *stats) {
  int l, j;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->P; ++j) {
      int index = MD_PIMD_INDEX_X(l, j, pimd->P);
      stats->es[stats->N-1] += te(sim->particles[index].x, L, sim->particles[index].m)/pimd->P;
    }
}

void md_pimd_calc_pair_force_x(md_pimd_t_x *pimd, md_pair_force_t_x pf) {
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  md_num_t_x f[MD_DIMENSION_X];
  int i;
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 < l; ++l2) {
          //if (l == l2)
            //continue;
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            f[i] = 0;
          pf(sim->particles[index].x, sim->particles[index2].x, f, L, sim->particles[index].m);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            sim->particles[index].f[i] += f[i]/pimd->P;
            sim->particles[index2].f[i] -= f[i]/pimd->P;
          }
        }
}

void md_pimd_calc_pair_energy_x(md_pimd_t_x *pimd, md_pair_energy_t_x pe, md_stats_t_x *stats) {
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 < l; ++l2) {
          //if (l == l2)
            //continue;
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          stats->es[stats->N-1] += pe(sim->particles[index].x, sim->particles[index2].x, L, sim->particles[index].m)/pimd->P;
        }
}

void md_pimd_calc_density_distribution_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmax, int image, md_num_t_x *center) {
  int l, j;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x incre = rmax/stats->points;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      int index = MD_PIMD_INDEX_X(l, j, pimd->P);
      md_num_t_x r;
      if (image)
        r = md_minimum_image_distance_x(sim->particles[index].x, center, L);
      else
        r = md_distance_x(sim->particles[index].x, center);
      index = (int)(r/incre);
      if (index >= stats->points)
        index = stats->points-1;
      stats->fxs[stats->N-1][index] += 1.0/pimd->P;
    }
}

void md_pimd_fast_fill_ENk_x(md_pimd_t_x *pimd, int image) {
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, j;
  md_num_t_x *Eint = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->N);
  int index, index2;
  md_num_t_x d;
  for (l = 1; l <= pimd->N; ++l) {
    md_num_t_x res = 0;
    for (j = 1; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, j+1, pimd->P);
      if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      res += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }
    Eint[l-1] = res;
    index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
    if (image)
      d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
    else
      d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
    pimd->ENk[l-1][0] = res+0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
  }
  for (l = 1; l <= pimd->N; ++l)
    for (j = 2; j <= l; ++j) {
      pimd->ENk[l-1][j-1] = pimd->ENk[l-1][j-2]+Eint[l-j];
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      pimd->ENk[l-1][j-1] -= 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
      index = MD_PIMD_INDEX_X(l-j+1, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      pimd->ENk[l-1][j-1] += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      pimd->ENk[l-1][j-1] += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }
  free(Eint);
}

md_num_t_x md_pimd_fast_xminE_x(md_pimd_t_x *pimd, int u, md_num_t_x *V) {
  if (pimd->vi == 0.0)
    return log(u)+pimd->beta*(pimd->ENk[u-1][0]+V[u]);
  int l;
  md_num_t_x res = 1e10;
  for (l = u; l <= pimd->N; ++l) {
    int k = l-u+1;
    md_num_t_x tmp;
    tmp = -(k-1)*log(pimd->vi)+log(l)+pimd->beta*(pimd->ENk[l-1][k-1]+V[l]);
    if (tmp < res)
      res = tmp;
  }
  return res;
}

md_num_t_x md_pimd_fast_xexp_x(md_num_t_x l, md_num_t_x k, md_num_t_x E, md_num_t_x EE, md_num_t_x beta, md_num_t_x vi) {
  if (vi == 0.0)
    return exp(-beta*E+EE-log(l));
  else
    return exp((k-1)*log(vi)-beta*E+EE-log(l));
}

void md_pimd_fast_dENk_x(md_pimd_t_x *pimd, int index, int index2, int index3, int image, md_num_t_x *dENk) {
  int i;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    dENk[i] = 0;
    if (image) {
      dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
      dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index3].x[i], L[i]);
    }
    else {
      dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*(sim->particles[index].x[i]-sim->particles[index2].x[i]);
      dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*(sim->particles[index].x[i]-sim->particles[index3].x[i]);
    }
  }
}

void md_pimd_fast_fill_force_VB_x(md_pimd_t_x *pimd, int image) {
  md_num_t_x *tmpV = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  tmpV[pimd->N] = 0;
  int l, j, u;
  for (u = pimd->N; u >= 1; --u) {
    md_num_t_x sum = 0;
    md_num_t_x tmp = md_pimd_fast_xminE_x(pimd, u, tmpV);
    for (l = u; l <= pimd->N; ++l) {
      int k = l-u+1;
      if (pimd->vi == 0 && k-1 != 0)
        continue;
      sum += md_pimd_fast_xexp_x(l, k, pimd->ENk[l-1][k-1]+tmpV[l], tmp, pimd->beta, pimd->vi);
    }
    tmpV[u-1] = (tmp-log(sum))/pimd->beta;
  }
  md_num_t_x *G = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->N*pimd->N);
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->N; ++j) {
      if (j > l+1)
        G[(l-1)*pimd->N+(j-1)] = 0;
      else if (j == l+1)
        G[(l-1)*pimd->N+(j-1)] = 1-exp(-pimd->beta*(pimd->VBN[l]+tmpV[l]-pimd->VBN[pimd->N]));
      else {
        if (pimd->vi == 0 && l-j != 0)
          G[(l-1)*pimd->N+(j-1)] = 0;
        else if (pimd->vi == 0 && l-j == 0)
          G[(l-1)*pimd->N+(j-1)] = exp(-pimd->beta*(pimd->VBN[j-1]+pimd->ENk[l-1][l-j]+tmpV[l]-pimd->VBN[pimd->N]))/l;
        else
          G[(l-1)*pimd->N+(j-1)] = exp(-pimd->beta*(pimd->VBN[j-1]+pimd->ENk[l-1][l-j]+tmpV[l]-pimd->VBN[pimd->N])+(l-j)*log(pimd->vi))/l;
      }
    }
  /*for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->N; ++j)
      printf("%f ", G[(l-1)*pimd->N+(j-1)]);
  printf("\n");*/
  md_num_t_x grad[MD_DIMENSION_X];
  int index, index2, index3;
  int i;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 2; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, j+1, pimd->P);
      index3 = MD_PIMD_INDEX_X(l, j-1, pimd->P);
      md_pimd_fast_dENk_x(pimd, index, index2, index3, image, grad);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].f[i] += -grad[i]/pimd->sim->particles[index].m;
    }
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->N; ++j) {
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, 2, pimd->P);
      index3 = MD_PIMD_INDEX_X(j, pimd->P, pimd->P);
      md_pimd_fast_dENk_x(pimd, index, index2, index3, image, grad);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].f[i] += -G[(j-1)*pimd->N+(l-1)]*grad[i]/pimd->sim->particles[index].m;
    }
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->N; ++j) {
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, pimd->P-1, pimd->P);
      index3 = MD_PIMD_INDEX_X(j, 1, pimd->P);
      md_pimd_fast_dENk_x(pimd, index, index2, index3, image, grad);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].f[i] += -G[(l-1)*pimd->N+(j-1)]*grad[i]/pimd->sim->particles[index].m;
    }
  free(tmpV);
  free(G);
}

void md_pimd_calc_ITCF_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmin, md_num_t_x rmax, int pi) {
  int l, j;
  int index, index2;
  int n = (int)sqrt(stats->points);
  md_num_t_x incre = (rmax-rmin)/n;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->N; ++j) {
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(j, pi, pimd->P);
      int i1 = (int)((pimd->sim->particles[index].x[0]-rmin)/incre);
      int i2 = (int)((pimd->sim->particles[index2].x[0]-rmin)/incre);
      int idx = i1*n+i2;
      if (idx < 0)
        idx = 0;
      if (idx >= stats->points)
        idx = stats->points-1;
      stats->fxs[stats->N-1][idx] += 1.0/pimd->N;
    }
}

int md_pimd_calc_pair_force_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_pair_force_t_x pf) {
  if (pimd->P != pimd2->P)
    return -1;
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  md_num_t_x f[MD_DIMENSION_X];
  int i;
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 <= pimd2->N; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            f[i] = 0;
          pf(sim->particles[index].x, sim2->particles[index2].x, f, L, sim->particles[index].m);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            sim->particles[index].f[i] += f[i]/pimd->P;
            sim2->particles[index2].f[i] -= f[i]/pimd->P;
          }
        }
  /*for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 <= pimd2->N; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            f[i] = 0;
          pf(sim2->particles[index2].x, sim->particles[index].x, f, L, sim2->particles[index2].m);
          for (i = 0; i < MD_DIMENSION_X; ++i)
            sim2->particles[index2].f[i] += f[i]/pimd->P;
        }*/
  return 0;
}

int md_pimd_calc_pair_energy_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_pair_energy_t_x pe, md_stats_t_x *stats) {
  if (pimd->P != pimd2->P)
    return -1;
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 <= pimd2->N; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          stats->es[stats->N-1] += pe(sim->particles[index].x, sim2->particles[index2].x, L, sim->particles[index].m)/pimd->P;
        }
  /*for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 <= pimd2->N; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          stats->es[stats->N-1] += 0.5*pe(sim2->particles[index2].x, sim->particles[index].x, L, sim2->particles[index2].m)/pimd->P;
        }*/
  return 0;
}

void md_pimd_calc_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats) {
  md_num_t_x res = 0;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, j, i;
  int index;
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        res += -(sim->particles[index].x[i]-L[i]/2)*sim->particles[index].f[i]*sim->particles[index].m;
    }
  stats->es[stats->N-1] += res/2.0;
}

void md_pimd_calc_centroid_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats, int fc) {
  md_num_t_x *centroid = (md_num_t_x *)malloc(MD_DIMENSION_X*sizeof(md_num_t_x)*pimd->N);
  int l, j, i;
  md_simulation_t_x *sim = pimd->sim;
  int index;
  for (l = 1; l <= pimd->N; ++l) {
    for (i = 0; i < MD_DIMENSION_X; ++i)
      centroid[MD_DIMENSION_X*(l-1)+i] = 0;
    for (j = 1; j <= pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        centroid[MD_DIMENSION_X*(l-1)+i] += sim->particles[index].x[i]/pimd->P;
    }
  }
  md_num_t_x res = 0;
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        res += -(sim->particles[index].x[i]-centroid[MD_DIMENSION_X*(l-1)+i])*sim->particles[index].f[i]*sim->particles[index].m;
    }
  stats->es[stats->N-1] += res/2.0;
  stats->es[stats->N-1] += (MD_DIMENSION_X*pimd->N-fc/((md_num_t_x)pimd->P))/(2*pimd->beta);
  free(centroid);
}

md_num_t_x md_pimd_ENk2_x(md_pimd_t_x *pimd, int N2, int k, int image) {
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, i;
  int index, index2, index3;
  md_num_t_x res = 0;
  md_num_t_x d, d2;
  for (l = N2-k+1; l <= N2; ++l) {
    index = md_pimd_next_index_x(l, pimd->P, N2, k, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
    index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (image) {
        d = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
        d2 = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index3].x[i], L[i]);
      }
      else {
        d = sim->particles[index].x[i]-sim->particles[index2].x[i];
        d2 = sim->particles[index].x[i]-sim->particles[index3].x[i];
      }
      res += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d2;
    }
  }
  return res;
}

void md_pimd_fill_ENk2_x(md_pimd_t_x *pimd, int image) {
  int l, j;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= l; ++j)
      pimd->ENk2[l-1][j-1] = md_pimd_ENk2_x(pimd, l, j, image);
}

int md_pimd_fprint_ENk2_x(FILE *out, md_pimd_t_x *pimd) {
  fprintf(out, "ENk2 %d\n", pimd->N);
  int l, j;
  for (l = 1; l <= pimd->N; ++l) {
    for (j = 1; j <= l; ++j)
      fprintf(out, "%f ", pimd->ENk2[l-1][j-1]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

void md_pimd_calc_vi_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats) {
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  res[0] = 0;
  int N2, k;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    md_num_t_x sum2 = 0;
    md_num_t_x tmp = md_pimd_xminE_x(pimd, N2);
    for (k = 1; k <= N2; ++k) {
      if (pimd->vi == 0 && k-1 != 0)
        continue;
      sum2 += md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
    }
    md_num_t_x sum = 0;
    for (k = 1; k <= N2; ++k) {
      if (pimd->vi == 0 && k-1 != 0)
        continue;
      sum += (res[N2-k]-pimd->ENk2[N2-1][k-1])*md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
    }
    res[N2] = sum/sum2;
    /*md_num_t_x sum = 0;
    for (k = 1; k <= N2; ++k) {
      if (pimd->vi == 0 && k-1 != 0)
        continue;
      else if (pimd->vi == 0 && k-1 == 0)
        sum += res[N2-k]+pimd->ENk2[N2-1][k-1];
      else
        sum += pow(pimd->vi, k-1)*(res[N2-k]+pimd->ENk2[N2-1][k-1]);
    }
    res[N2] = sum/N2;*/
  }
  stats->es[stats->N-1] += res[pimd->N];
  stats->es[stats->N-1] += MD_DIMENSION_X*pimd->N/(2*pimd->beta);
  free(res);
  md_num_t_x res2 = 0;
  int l, j, i;
  int index, index2;
  md_simulation_t_x *sim = pimd->sim;
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        res2 += -(sim->particles[index].x[i]-sim->particles[index2].x[i])*sim->particles[index].f[i]*sim->particles[index].m;
    }
  stats->es[stats->N-1] += res2/2.0;
}

void md_pimd_polymer_periodic_boundary_x(md_pimd_t_x *pimd) {
  int l, j, i;
  int index;
  int count[MD_DIMENSION_X];
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (l = 1; l <= pimd->N; ++l) {
    index = MD_PIMD_INDEX_X(l, 1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      count[i] = md_periodic_image_count_x(sim->particles[index].x[i], L[i]);
    for (j = 1; j <= pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        sim->particles[index].x[i] += count[i]*L[i];
    }
  }
}

md_num_t_x md_pimd_fast_ENk2_x(md_pimd_t_x *pimd, int index, int index2, int index3, int image) {
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int i;
  md_num_t_x res = 0;
  md_num_t_x d, d2;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (image) {
      d = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
      d2 = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index3].x[i], L[i]);
    }
    else {
      d = sim->particles[index].x[i]-sim->particles[index2].x[i];
      d2 = sim->particles[index].x[i]-sim->particles[index3].x[i];
    }
    res += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d2;
  }
  return res;
}

void md_pimd_fast_fill_ENk2_x(md_pimd_t_x *pimd, int image) {
  int l, j;
  int index, index2, index3;
  for (l = 1; l <= pimd->N; ++l) {
    index = MD_PIMD_INDEX_X(l, 1, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
    index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
    pimd->ENk2[l-1][0] = md_pimd_fast_ENk2_x(pimd, index, index2, index3, image);
  }
  for (l = 1; l <= pimd->N; ++l)
    for (j = 2; j <= l; ++j) {
      index = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      pimd->ENk2[l-1][j-1] = pimd->ENk2[l-1][j-2]-md_pimd_fast_ENk2_x(pimd, index, index2, index3, image);
      index = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+1, pimd->P, pimd->P);
      index3 = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      pimd->ENk2[l-1][j-1] += md_pimd_fast_ENk2_x(pimd, index, index2, index3, image);
      index = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      pimd->ENk2[l-1][j-1] += md_pimd_fast_ENk2_x(pimd, index, index2, index3, image);
    }
}