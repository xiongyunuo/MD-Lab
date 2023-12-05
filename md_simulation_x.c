#include "md_simulation_x.h"
#include <stdlib.h>
#include <math.h>

md_num_t_x md_minimum_image_x(md_num_t_x a, md_num_t_x L) {
  /*if (fabs(a) > L/2) {
    if (a < 0)
      return L - fabs(a);
    else
      return -(L - fabs(a));
  }
  return a;*/
  a += L/2;
  a += L*md_periodic_image_count_x(a, L);
  return a-L/2;
}

md_num_t_x md_minimum_image_distance_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L) {
  md_num_t_x d = 0;
  md_num_t_x tmp;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    tmp = md_minimum_image_x(p1[i]-p2[i], L[i]);
    d += tmp*tmp;
  }
  return sqrt(d);
}

md_num_t_x md_distance_x(md_num_t_x *p1, md_num_t_x *p2) {
  md_num_t_x d = 0;
  md_num_t_x tmp;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    tmp = p1[i]-p2[i];
    d += tmp*tmp;
  }
  return sqrt(d);
}

void md_LJ_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m) {
  md_num_t_x r = md_minimum_image_distance_x(p1, p2, L);
  int i;
  md_num_t_x mult = 48*pow(r, -14.0)-24*pow(r, -8.0);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += md_minimum_image_x(p1[i]-p2[i], L[i])*mult/m;
}

md_num_t_x md_trap_frequency_x;
md_num_t_x md_trap_center_x[MD_DIMENSION_X];

void md_harmonic_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m)) {
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += -md_trap_frequency_x*md_trap_frequency_x*(p1[i]-md_trap_center_x[i]);
}

void md_set_harmonic_trap_frequency_x(md_num_t_x f) {
  md_trap_frequency_x = f;
}

void md_set_harmonic_trap_center_x(md_num_t_x *center) {
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    md_trap_center_x[i] = center[i];
}

md_num_t_x md_gaussian_strength_x;
md_num_t_x md_gaussian_range_x;

void md_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m) {
  int i;
  md_num_t_x d = md_distance_x(p1, p2);
  md_num_t_x mult = (md_gaussian_strength_x/(M_PI*md_gaussian_range_x*md_gaussian_range_x))*exp(-d*d/(md_gaussian_range_x*md_gaussian_range_x));
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += mult*(2*(p1[i]-p2[i]))/(md_gaussian_range_x*md_gaussian_range_x)/m;
}

void md_set_gaussian_force_strength_x(md_num_t_x g) {
  md_gaussian_strength_x = g;
}

void md_set_guassian_force_range_x(md_num_t_x s) {
  md_gaussian_range_x = s;
}

md_num_t_x md_coulomb_strength_x;

void md_coulomb_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m) {
  int i;
  md_num_t_x d = md_distance_x(p1, p2);
  md_num_t_x mult = md_coulomb_strength_x/pow(d,3);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += mult*(p1[i]-p2[i])/m;
}

void md_coulomb_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m) {
  int i;
  md_num_t_x d = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x mult = md_coulomb_strength_x/pow(d,3);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += mult*md_minimum_image_x(p1[i]-p2[i], L[i])/m;
}

void md_set_coulomb_force_strength_x(md_num_t_x g) {
  md_coulomb_strength_x = g;
}

md_num_t_x md_coulomb_truc_x;
int md_coulomb_n_sum_x;

void md_set_coulomb_truc_x(md_num_t_x k) {
  md_coulomb_truc_x = k;
}

void md_set_coulomb_n_sum_x(int n) {
  md_coulomb_n_sum_x = n;
}

md_num_t_x md_calc_3d_ueg_madelung_x(md_num_t_x *L) {
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
        res += pow(G,-2)*exp(-M_PI*M_PI*G*G/(md_coulomb_truc_x*md_coulomb_truc_x))/(V*M_PI);
      }
  res -= M_PI/(md_coulomb_truc_x*md_coulomb_truc_x*V);
  for (i = -md_coulomb_n_sum_x; i <= md_coulomb_n_sum_x; ++i)
    for (j = -md_coulomb_n_sum_x; j <= md_coulomb_n_sum_x; ++j)
      for (k = -md_coulomb_n_sum_x; k <= md_coulomb_n_sum_x; ++k) {
        if (i == 0 && j == 0 && k == 0)
          continue;
        r[0] = i*L[0];
        r[1] = j*L[1];
        r[2] = k*L[2];
        R = md_distance_x(r, origin);
        res += erfc(md_coulomb_truc_x*R)/R;
      }
  res -= 2*md_coulomb_truc_x*pow(M_PI, -0.5);
  return md_coulomb_strength_x*res;
}

void md_coulomb_3d_ewald_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m) {
  if (MD_DIMENSION_X != 3)
    return;
  md_num_t_x V = L[0]*L[1]*L[2];
  md_num_t_x origin[3] = { 0, 0, 0 };
  md_num_t_x g[3], r[3];
  md_num_t_x G, R;
  int i, j, k, d;
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
        md_num_t_x mult = md_coulomb_strength_x*sin(dot)*pow(G,-2)*exp(-M_PI*M_PI*G*G/(md_coulomb_truc_x*md_coulomb_truc_x))/(V*M_PI);
        for (d = 0; d < MD_DIMENSION_X; ++d)
          f[d] += 2*M_PI*g[d]*mult/m;
      }
  for (i = -md_coulomb_n_sum_x; i <= md_coulomb_n_sum_x; ++i)
    for (j = -md_coulomb_n_sum_x; j <= md_coulomb_n_sum_x; ++j)
      for (k = -md_coulomb_n_sum_x; k <= md_coulomb_n_sum_x; ++k) {
        r[0] = p2[0]+i*L[0];
        r[1] = p2[1]+j*L[1];
        r[2] = p2[2]+k*L[2];
        R = md_distance_x(p1, r);
        md_num_t_x mult = md_coulomb_strength_x*2*md_coulomb_truc_x*exp(-md_coulomb_truc_x*md_coulomb_truc_x*R*R)/(sqrt(M_PI)*R*R);
        mult += md_coulomb_strength_x*erfc(md_coulomb_truc_x*R)/(R*R*R);
        for (d = 0; d < MD_DIMENSION_X; ++d)
          f[d] += (p1[d]-r[d])*mult/m;
      }
}

void md_reset_force_x(md_simulation_t_x *sim) {
  int i, j;
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j)
      sim->particles[i].f[j] = 0;
}

void md_calc_pair_force_x(md_simulation_t_x *sim, md_pair_force_t_x pf) {
  int i, j;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < sim->N; ++j) {
      if (i != j)
        pf(sim->particles[i].x, sim->particles[j].x, sim->particles[i].f, L, sim->particles[i].m);
    }
}

void md_calc_nhc_force_x(md_simulation_t_x *sim, int f0, int f2, int i, md_num_t_x *res) {
  int j;
  int f = sim->nhcs[i].f;
  md_num_t_x sum = 0;
  int index, pi, pj;
  for (j = 0; j < f2; ++j) {
    index = f0+j;
    pi = index/MD_DIMENSION_X;
    pj = index%MD_DIMENSION_X;
    sum += sim->particles[pi].m*sim->particles[pi].v[pj]*sim->particles[pi].v[pj];
  }
  res[0] = (sum-f*MD_kB_X*sim->T)/sim->nhcs[i].Q[0];
  for (j = 1; j < MD_NHC_LENGTH_X; ++j)
    res[j] = (sim->nhcs[i].Q[j-1]*sim->nhcs[i].vtheta[j-1]*sim->nhcs[i].vtheta[j-1]-MD_kB_X*sim->T)/sim->nhcs[i].Q[j];
}

void md_update_nhc_VV3_1_x(md_simulation_t_x *sim, md_num_t_x h) {
  int f0 = 0;
  int i, j;
  int index, pi, pj;
  int f, f2;
  md_num_t_x *nhf = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_NHC_LENGTH_X);
  for (i = 0; i < sim->Nf; ++i) {
    f = sim->nhcs[i].f;
    if (i == sim->Nf-1)
      f2 = MD_DIMENSION_X*sim->N-f0;
    else
      f2 = f;
    md_calc_nhc_force_x(sim, f0, f2, i, nhf);
    for (j = 0; j < f2; ++j) {
      index = f0+j;
      pi = index/MD_DIMENSION_X;
      pj = index%MD_DIMENSION_X;
      sim->particles[pi].v[pj] = sim->particles[pi].v[pj]*exp(-0.5*h*sim->nhcs[i].vtheta[0])+0.5*h*sim->particles[pi].f[pj]*exp(-0.25*h*sim->nhcs[i].vtheta[0]);
    }
    int M2 = MD_NHC_LENGTH_X/2;
    for (j = 1; j <= M2; ++j)
      sim->nhcs[i].theta[2*j-2] = sim->nhcs[i].theta[2*j-2]+h*sim->nhcs[i].vtheta[2*j-2]/2;
    for (j = 1; j <= M2; ++j)
      sim->nhcs[i].vtheta[2*j-1] = sim->nhcs[i].vtheta[2*j-1]*exp(-0.5*h*((j==M2)?0:sim->nhcs[i].vtheta[2*j]))+0.5*h*nhf[2*j-1]*exp(-0.25*h*((j==M2)?0:sim->nhcs[i].vtheta[2*j]));
    for (j = 0; j < f2; ++j) {
      index = f0+j;
      pi = index/MD_DIMENSION_X;
      pj = index%MD_DIMENSION_X;
      sim->particles[pi].x[pj] = sim->particles[pi].x[pj]+h*sim->particles[pi].v[pj];
    }
    for (j = 1; j <= M2; ++j)
      sim->nhcs[i].theta[2*j-1] = sim->nhcs[i].theta[2*j-1]+h*sim->nhcs[i].vtheta[2*j-1];
    md_calc_nhc_force_x(sim, f0, f2, i, nhf);
    for (j = 1; j <= M2; ++j)
      sim->nhcs[i].vtheta[2*j-2] = sim->nhcs[i].vtheta[2*j-2]*exp(-h*sim->nhcs[i].vtheta[2*j-1])+h*nhf[2*j-2]*exp(-0.5*h*sim->nhcs[i].vtheta[2*j-1]);
    f0 += f;
  }
  free(nhf);
}

void md_update_nhc_VV3_2_x(md_simulation_t_x *sim, md_num_t_x h) {
  int f0 = 0;
  int i, j;
  int index, pi, pj;
  int f, f2;
  md_num_t_x *nhf = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_NHC_LENGTH_X);
  for (i = 0; i < sim->Nf; ++i) {
    f = sim->nhcs[i].f;
    if (i == sim->Nf-1)
      f2 = MD_DIMENSION_X*sim->N-f0;
    else
      f2 = f;
    for (j = 0; j < f2; ++j) {
      index = f0+j;
      pi = index/MD_DIMENSION_X;
      pj = index%MD_DIMENSION_X;
      sim->particles[pi].v[pj] = sim->particles[pi].v[pj]*exp(-0.5*h*sim->nhcs[i].vtheta[0])+0.5*h*sim->particles[pi].f[pj]*exp(-0.25*h*sim->nhcs[i].vtheta[0]);
    }
    int M2 = MD_NHC_LENGTH_X/2;
    for (j = 1; j <= M2; ++j)
      sim->nhcs[i].theta[2*j-2] = sim->nhcs[i].theta[2*j-2]+h*sim->nhcs[i].vtheta[2*j-2]/2;
    md_calc_nhc_force_x(sim, f0, f2, i, nhf);
    for (j = 1; j <= M2; ++j)
      sim->nhcs[i].vtheta[2*j-1] = sim->nhcs[i].vtheta[2*j-1]*exp(-0.5*h*((j==M2)?0:sim->nhcs[i].vtheta[2*j]))+0.5*h*nhf[2*j-1]*exp(-0.25*h*((j==M2)?0:sim->nhcs[i].vtheta[2*j]));
    f0 += f;
  }
  free(nhf);
}

void md_periodic_boundary_x(md_simulation_t_x *sim) {
  int i, j;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = (nd_rect_t_x *)sim->box->box;
    md_num_t_x *L = rect->L;
    for (i = 0; i < sim->N; ++i)
      for (j = 0; j < MD_DIMENSION_X; ++j) {
        if (sim->particles[i].x[j] < 0)
          sim->particles[i].x[j] += L[j];
        else if (sim->particles[i].x[j] > L[j])
          sim->particles[i].x[j] -= L[j];
      }
  }
}

int md_periodic_image_count_x(md_num_t_x x, md_num_t_x L) {
  int count = (int)(fabs(x)/L);
  if (x > L)
    return -count;
  else if (x < 0)
    return count+1;
  return 0;
}

void md_periodic_boundary_count_x(md_simulation_t_x *sim, int *count) {
  int l, i;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (l = 0; l < sim->N; ++l)
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      count[MD_DIMENSION_X*l+i] = md_periodic_image_count_x(sim->particles[l].x[i], L[i]);
      sim->particles[l].x[i] += count[MD_DIMENSION_X*l+i]*L[i];
    }
}

void md_periodic_boundary_recover_x(md_simulation_t_x *sim, int *count) {
  int l, i;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (l = 0; l < sim->N; ++l)
    for (i = 0; i < MD_DIMENSION_X; ++i)
      sim->particles[l].x[i] -= count[MD_DIMENSION_X*l+i]*L[i];
}