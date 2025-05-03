#ifdef MD_DOUBLE_PREC_X
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double md_num_t_x;
#else
typedef float md_num_t_x;
#endif

#ifndef MD_DIMENSION_X
#define MD_DIMENSION_X 1
#endif

#ifndef MD_NHC_LENGTH_X
#define MD_NHC_LENGTH_X 1
#endif

#define MD_kB_X 1.0
#define MD_hBar_X 1.0

typedef struct {
  md_num_t_x m;
  md_num_t_x x[MD_DIMENSION_X];
  md_num_t_x v[MD_DIMENSION_X];
  md_num_t_x f[MD_DIMENSION_X];
} md_particle_t_x;

typedef struct {
  int f;
  md_num_t_x theta[MD_NHC_LENGTH_X];
  md_num_t_x vtheta[MD_NHC_LENGTH_X];
  md_num_t_x Q[MD_NHC_LENGTH_X];
} md_nhc_t_x;

typedef struct {
  md_num_t_x L[MD_DIMENSION_X];
} nd_rect_t_x;

void md_calc_nhc_force_x(global md_particle_t_x *particles, global md_nhc_t_x *nhcs, int f0, int f2, int i, md_num_t_x *res, md_num_t_x T);

void md_calc_nhc_force_x(global md_particle_t_x *particles, global md_nhc_t_x *nhcs, int f0, int f2, int i, md_num_t_x *res, md_num_t_x T) {
  int j;
  int f = nhcs[i].f;
  md_num_t_x sum = 0;
  int index, pi, pj;
  for (j = 0; j < f2; ++j) {
    index = f0+j;
    pi = index/MD_DIMENSION_X;
    pj = index%MD_DIMENSION_X;
    sum += particles[pi].m*particles[pi].v[pj]*particles[pi].v[pj];
  }
  res[0] = (sum-f*MD_kB_X*T)/nhcs[i].Q[0];
  for (j = 1; j < MD_NHC_LENGTH_X; ++j)
    res[j] = (nhcs[i].Q[j-1]*nhcs[i].vtheta[j-1]*nhcs[i].vtheta[j-1]-MD_kB_X*T)/nhcs[i].Q[j];
}

kernel void md_update_nhc_VV3_1_kx(global md_particle_t_x *particles, global md_nhc_t_x *nhcs, global int *f0s, md_num_t_x h, int N, int Nf, md_num_t_x T) {
  int i = get_global_id(0);
  if (i >= Nf)
    return;
  int f0 = f0s[i];
  int j;
  int index, pi, pj;
  int f, f2;
  md_num_t_x nhf[MD_NHC_LENGTH_X];
  f = nhcs[i].f;
  if (i == Nf-1)
    f2 = MD_DIMENSION_X*N-f0;
  else
    f2 = f;
  md_calc_nhc_force_x(particles, nhcs, f0, f2, i, nhf, T);
  for (j = 0; j < f2; ++j) {
    index = f0+j;
    pi = index/MD_DIMENSION_X;
    pj = index%MD_DIMENSION_X;
    particles[pi].v[pj] = particles[pi].v[pj]*exp(-0.5*h*nhcs[i].vtheta[0])+0.5*h*particles[pi].f[pj]*exp(-0.25*h*nhcs[i].vtheta[0]);
  }
  int M2 = MD_NHC_LENGTH_X/2;
  for (j = 1; j <= M2; ++j)
    nhcs[i].theta[2*j-2] = nhcs[i].theta[2*j-2]+h*nhcs[i].vtheta[2*j-2]/2;
  for (j = 1; j <= M2; ++j)
    nhcs[i].vtheta[2*j-1] = nhcs[i].vtheta[2*j-1]*exp(-0.5*h*((j==M2)?0:nhcs[i].vtheta[2*j]))+0.5*h*nhf[2*j-1]*exp(-0.25*h*((j==M2)?0:nhcs[i].vtheta[2*j]));
  for (j = 0; j < f2; ++j) {
    index = f0+j;
    pi = index/MD_DIMENSION_X;
    pj = index%MD_DIMENSION_X;
    particles[pi].x[pj] = particles[pi].x[pj]+h*particles[pi].v[pj];
  }
  for (j = 1; j <= M2; ++j)
    nhcs[i].theta[2*j-1] = nhcs[i].theta[2*j-1]+h*nhcs[i].vtheta[2*j-1];
  md_calc_nhc_force_x(particles, nhcs, f0, f2, i, nhf, T);
  for (j = 1; j <= M2; ++j)
    nhcs[i].vtheta[2*j-2] = nhcs[i].vtheta[2*j-2]*exp(-h*nhcs[i].vtheta[2*j-1])+h*nhf[2*j-2]*exp(-0.5*h*nhcs[i].vtheta[2*j-1]);
}

kernel void md_update_nhc_VV3_2_kx(global md_particle_t_x *particles, global md_nhc_t_x *nhcs, global int *f0s, md_num_t_x h, int N, int Nf, md_num_t_x T) {
  int i = get_global_id(0);
  if (i >= Nf)
    return;
  int f0 = f0s[i];
  int j;
  int index, pi, pj;
  int f, f2;
  md_num_t_x nhf[MD_NHC_LENGTH_X];
  f = nhcs[i].f;
  if (i == Nf-1)
    f2 = MD_DIMENSION_X*N-f0;
  else
    f2 = f;
  for (j = 0; j < f2; ++j) {
    index = f0+j;
    pi = index/MD_DIMENSION_X;
    pj = index%MD_DIMENSION_X;
    particles[pi].v[pj] = particles[pi].v[pj]*exp(-0.5*h*nhcs[i].vtheta[0])+0.5*h*particles[pi].f[pj]*exp(-0.25*h*nhcs[i].vtheta[0]);
  }
  int M2 = MD_NHC_LENGTH_X/2;
  for (j = 1; j <= M2; ++j)
    nhcs[i].theta[2*j-2] = nhcs[i].theta[2*j-2]+h*nhcs[i].vtheta[2*j-2]/2;
  md_calc_nhc_force_x(particles, nhcs, f0, f2, i, nhf, T);
  for (j = 1; j <= M2; ++j)
    nhcs[i].vtheta[2*j-1] = nhcs[i].vtheta[2*j-1]*exp(-0.5*h*((j==M2)?0:nhcs[i].vtheta[2*j]))+0.5*h*nhf[2*j-1]*exp(-0.25*h*((j==M2)?0:nhcs[i].vtheta[2*j]));
}

kernel void md_reset_force_kx(global md_particle_t_x *particles, int N) {
  int id = get_global_id(0);
  if (id >= MD_DIMENSION_X*N)
    return;
  int i = id/MD_DIMENSION_X;
  int j = id%MD_DIMENSION_X;
  particles[i].f[j] = 0;
}

kernel void md_rect_periodic_boundary_kx(global md_particle_t_x *particles, int N, nd_rect_t_x rect) {
  int id = get_global_id(0);
  if (id >= MD_DIMENSION_X*N)
    return;
  int i = id/MD_DIMENSION_X;
  int j = id%MD_DIMENSION_X;
  md_num_t_x *L = rect.L;
  if (particles[i].x[j] < 0)
    particles[i].x[j] += L[j];
  else if (particles[i].x[j] > L[j])
    particles[i].x[j] -= L[j];
}

#define MD_MAX_INTER_PARAMETERS_X 15

typedef struct {
  md_num_t_x params[MD_MAX_INTER_PARAMETERS_X];
} md_inter_params_t_x;

typedef struct {
  int i, j;
} md_index_pair_t_x;

int md_periodic_image_count_x(md_num_t_x x, md_num_t_x L);

int md_periodic_image_count_x(md_num_t_x x, md_num_t_x L) {
  int count = (int)(fabs(x)/L);
  if (x > L)
    return -count;
  else if (x < 0)
    return count+1;
  return 0;
}

md_num_t_x md_minimum_image_x(md_num_t_x a, md_num_t_x L);
md_num_t_x md_minimum_image_distance_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L);
md_num_t_x md_distance_x(md_num_t_x *p1, md_num_t_x *p2);

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

#ifdef __GNUC__
  #define MD_UNUSED_X(x) UNUSED_ ## x __attribute__((__unused__))
#else
  #define MD_UNUSED_X(x) UNUSED_ ## x
#endif

void md_LJ_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params);

void md_LJ_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *MD_UNUSED_X(params)) {
  md_num_t_x r = md_minimum_image_distance_x(p1, p2, L);
  int i;
  md_num_t_x mult = 48*pown(r, -14)-24*pown(r, -8);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += md_minimum_image_x(p1[i]-p2[i], L[i])*mult/m;
}

void md_harmonic_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params);

void md_harmonic_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params) {
  int i;
  md_num_t_x md_trap_frequency_x = params->params[0];
  md_num_t_x md_trap_center_x[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    md_trap_center_x[i] = params->params[1+i];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += -md_trap_frequency_x*md_trap_frequency_x*(p1[i]-md_trap_center_x[i]);
}

void md_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m, md_inter_params_t_x *params);

void md_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m, md_inter_params_t_x *params) {
  int i;
  md_num_t_x md_gaussian_strength_x = params->params[0];
  md_num_t_x md_gaussian_range_x = params->params[1];
  md_num_t_x d = md_distance_x(p1, p2);
  md_num_t_x mult = (md_gaussian_strength_x/(M_PI*md_gaussian_range_x*md_gaussian_range_x))*exp(-d*d/(md_gaussian_range_x*md_gaussian_range_x));
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += mult*(2*(p1[i]-p2[i]))/(md_gaussian_range_x*md_gaussian_range_x)/m;
}

void md_periodic_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params);

void md_periodic_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params) {
  int i;
  md_num_t_x md_gaussian_strength_x = params->params[0];
  md_num_t_x md_gaussian_range_x = params->params[1];
  md_num_t_x d = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x mult = (md_gaussian_strength_x/(M_PI*md_gaussian_range_x*md_gaussian_range_x))*exp(-d*d/(md_gaussian_range_x*md_gaussian_range_x));
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += mult*(2*md_minimum_image_x(p1[i]-p2[i], L[i]))/(md_gaussian_range_x*md_gaussian_range_x)/m;
}

void md_coulomb_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m, md_inter_params_t_x *params);

void md_coulomb_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m, md_inter_params_t_x *params) {
  int i;
  md_num_t_x md_coulomb_strength_x = params->params[0];
  md_num_t_x d = md_distance_x(p1, p2);
  md_num_t_x mult = md_coulomb_strength_x/pow(d,3);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += mult*(p1[i]-p2[i])/m;
}

void md_coulomb_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params);

void md_coulomb_periodic_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params) {
  int i;
  md_num_t_x md_coulomb_strength_x = params->params[0];
  md_num_t_x d = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x mult = md_coulomb_strength_x/pow(d,3);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += mult*md_minimum_image_x(p1[i]-p2[i], L[i])/m;
}

void md_coulomb_3d_ewald_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params);

void md_coulomb_3d_ewald_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params) {
  if (MD_DIMENSION_X != 3)
    return;
  md_num_t_x md_coulomb_strength_x = params->params[0];
  md_num_t_x md_coulomb_truc_x = params->params[1];
  int md_coulomb_n_sum_x = (int)params->params[2];
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

void md_hubbard_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m, md_inter_params_t_x *params);

void md_hubbard_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m, md_inter_params_t_x *params) {
  int i;
  md_num_t_x md_hubbard_trap_strength_x = params->params[0];
  md_num_t_x md_hubbard_trap_frequency_x = params->params[1];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += 2*md_hubbard_trap_strength_x*md_hubbard_trap_frequency_x*cos(md_hubbard_trap_frequency_x*p1[i])*sin(md_hubbard_trap_frequency_x*p1[i])/m;
}

void md_periodic_helium_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params);

void md_periodic_helium_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m, md_inter_params_t_x *params) {
  md_num_t_x md_he_eps_x = params->params[0];
  md_num_t_x md_he_A_x = params->params[1];
  md_num_t_x md_he_alpha_x = params->params[2];
  md_num_t_x md_he_C6_x = params->params[3];
  md_num_t_x md_he_C8_x = params->params[4];
  md_num_t_x md_he_C10_x = params->params[5];
  md_num_t_x md_he_D_x = params->params[6];
  md_num_t_x md_he_rm_x = params->params[7];
  md_num_t_x r = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x x = r/md_he_rm_x;
  md_num_t_x F = 0;
  if (x < md_he_D_x)
    F = exp(-pow(md_he_D_x/x-1, 2));
  else
    F = 1;
  md_num_t_x mult = md_he_alpha_x*md_he_A_x*exp(-md_he_alpha_x*x)-(6*md_he_C6_x*pow(x, -7)+8*md_he_C8_x*pow(x, -9)+10*md_he_C10_x*pow(x, -11))*F;
  if (x < md_he_D_x)
    mult += (md_he_C6_x*pow(x, -6)+md_he_C8_x*pow(x, -8)+md_he_C10_x*pow(x, -10))*F*2*md_he_D_x*(md_he_D_x/x-1)/(x*x);
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += md_he_eps_x*mult*md_minimum_image_x(p1[i]-p2[i], L[i])/(r*md_he_rm_x*m);
}

#define MD_FILL_PAIR_FORCE_KX(name, func) kernel void md_fill_pair_force_ ## name ## _kx(global md_particle_t_x *particles, global md_num_t_x *forces, global md_index_pair_t_x *indices, int count, int N, md_inter_params_t_x params, nd_rect_t_x rect) {\
  int id = get_global_id(0);\
  if (id >= count)\
    return;\
  int i = indices[id].i;\
  int j = indices[id].j;\
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X], f[MD_DIMENSION_X];\
  int d;\
  for (d = 0; d < MD_DIMENSION_X; ++d) {\
    p1[d] = particles[i].x[d];\
    p2[d] = particles[j].x[d];\
    f[d] = 0;\
  }\
  func(p1, p2, f, rect.L, particles[i].m, &params);\
  int index = i*N+j;\
  int index2 = j*N+i;\
  for (d = 0; d < MD_DIMENSION_X; ++d) {\
    forces[MD_DIMENSION_X*index+d] = f[d];\
    forces[MD_DIMENSION_X*index2+d] = -f[d];\
  }\
}

MD_FILL_PAIR_FORCE_KX(pLJ, md_LJ_periodic_force_x)
MD_FILL_PAIR_FORCE_KX(Gau, md_gaussian_force_x)
MD_FILL_PAIR_FORCE_KX(pGau, md_periodic_gaussian_force_x)
MD_FILL_PAIR_FORCE_KX(Cou, md_coulomb_force_x)
MD_FILL_PAIR_FORCE_KX(pCou, md_coulomb_periodic_force_x)
MD_FILL_PAIR_FORCE_KX(Ewald, md_coulomb_3d_ewald_force_x)
MD_FILL_PAIR_FORCE_KX(He, md_periodic_helium_force_x)

kernel void md_update_pair_force_kx(global md_particle_t_x *particles, global md_num_t_x *forces, int N) {
  int i = get_global_id(0);
  if (i >= N)
    return;
  int j, index, d;
  for (j = 0; j < N; ++j) {
    if (i == j)
      continue;
    index = i*N+j;
    for (d = 0; d < MD_DIMENSION_X; ++d)
      particles[i].f[d] += forces[MD_DIMENSION_X*index+d];
  }
}

md_num_t_x md_LJ_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *MD_UNUSED_X(params));

md_num_t_x md_LJ_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *MD_UNUSED_X(params)) {
  md_num_t_x r = md_minimum_image_distance_x(p1, p2, L);
  return 4*(pown(r, -12)-pown(r, -6));
}

md_num_t_x md_harmonic_trap_energy_x(md_num_t_x *p1, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m, md_inter_params_t_x *params);

md_num_t_x md_harmonic_trap_energy_x(md_num_t_x *p1, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m, md_inter_params_t_x *params) {
  int i;
  md_num_t_x md_trap_frequency_x = params->params[0];
  md_num_t_x md_trap_center_x[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    md_trap_center_x[i] = params->params[1+i];
  md_num_t_x d = md_distance_x(p1, md_trap_center_x);
  return 0.5*m*md_trap_frequency_x*md_trap_frequency_x*d*d;
}

md_num_t_x md_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params);

md_num_t_x md_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params) {
  md_num_t_x md_gaussian_strength_x = params->params[0];
  md_num_t_x md_gaussian_range_x = params->params[1];
  md_num_t_x d = md_distance_x(p1, p2);
  md_num_t_x mult = (md_gaussian_strength_x/(M_PI*md_gaussian_range_x*md_gaussian_range_x))*exp(-d*d/(md_gaussian_range_x*md_gaussian_range_x));
  return mult;
}

md_num_t_x md_periodic_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params);

md_num_t_x md_periodic_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params) {
  md_num_t_x md_gaussian_strength_x = params->params[0];
  md_num_t_x md_gaussian_range_x = params->params[1];
  md_num_t_x d = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x mult = (md_gaussian_strength_x/(M_PI*md_gaussian_range_x*md_gaussian_range_x))*exp(-d*d/(md_gaussian_range_x*md_gaussian_range_x));
  return mult;
}

md_num_t_x md_coulomb_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params);

md_num_t_x md_coulomb_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params) {
  md_num_t_x md_coulomb_strength_x = params->params[0];
  md_num_t_x d = md_distance_x(p1, p2);
  md_num_t_x mult = md_coulomb_strength_x/d;
  return mult;
}

md_num_t_x md_coulomb_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params);

md_num_t_x md_coulomb_periodic_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params) {
  md_num_t_x md_coulomb_strength_x = params->params[0];
  md_num_t_x d = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x mult = md_coulomb_strength_x/d;
  return mult;
}

md_num_t_x md_coulomb_3d_ewald_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params);

md_num_t_x md_coulomb_3d_ewald_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params) {
  if (MD_DIMENSION_X != 3)
    return 0;
  md_num_t_x md_coulomb_strength_x = params->params[0];
  md_num_t_x md_coulomb_truc_x = params->params[1];
  int md_coulomb_n_sum_x = (int)params->params[2];
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

md_num_t_x md_hubbard_trap_energy_x(md_num_t_x *p1, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params);

md_num_t_x md_hubbard_trap_energy_x(md_num_t_x *p1, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params) {
  md_num_t_x md_hubbard_trap_strength_x = params->params[0];
  md_num_t_x md_hubbard_trap_frequency_x = params->params[1];
  md_num_t_x res = 0;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    res += md_hubbard_trap_strength_x*pow(cos(md_hubbard_trap_frequency_x*p1[i]), 2);
  return res;
}

md_num_t_x md_periodic_helium_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params);

md_num_t_x md_periodic_helium_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m), md_inter_params_t_x *params) {
  md_num_t_x md_he_eps_x = params->params[0];
  md_num_t_x md_he_A_x = params->params[1];
  md_num_t_x md_he_alpha_x = params->params[2];
  md_num_t_x md_he_C6_x = params->params[3];
  md_num_t_x md_he_C8_x = params->params[4];
  md_num_t_x md_he_C10_x = params->params[5];
  md_num_t_x md_he_D_x = params->params[6];
  md_num_t_x md_he_rm_x = params->params[7];
  md_num_t_x r = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x x = r/md_he_rm_x;
  md_num_t_x F = 0;
  if (x < md_he_D_x)
    F = exp(-pow(md_he_D_x/x-1, 2));
  else
    F = 1;
  md_num_t_x res = md_he_A_x*exp(-md_he_alpha_x*x)-(md_he_C6_x*pow(x, -6)+md_he_C8_x*pow(x, -8)+md_he_C10_x*pow(x, -10))*F;
  return md_he_eps_x*res;
}

#define MD_CALC_PAIR_ENERGY_KX(name, func) kernel void md_calc_pair_energy_ ## name ## _kx(global md_particle_t_x *particles, global md_num_t_x *output, global md_index_pair_t_x *indices, int count, md_inter_params_t_x params, nd_rect_t_x rect, local md_num_t_x *target) {\
  const int gid = get_global_id(0);\
  const int lid = get_local_id(0);\
  if (gid < count) {\
    int i = indices[gid].i;\
    int j = indices[gid].j;\
    md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];\
    int d;\
    for (d = 0; d < MD_DIMENSION_X; ++d) {\
      p1[d] = particles[i].x[d];\
      p2[d] = particles[j].x[d];\
    }\
    target[lid] = func(p1, p2, rect.L, particles[i].m, &params);\
  }\
  else\
    target[lid] = 0;\
  barrier(CLK_LOCAL_MEM_FENCE);\
  int gsize = get_local_size(0);\
  int half_gsize = gsize/2;\
  while (half_gsize > 0) {\
    if (lid < half_gsize) {\
      target[lid] += target[lid+half_gsize];\
      if (gsize%2 != 0)\
        if (lid == 0)\
          target[0] += target[gsize-1];\
    }\
    barrier(CLK_LOCAL_MEM_FENCE);\
    gsize = half_gsize;\
    half_gsize = gsize/2;\
  }\
  if (lid == 0)\
    output[get_group_id(0)] = target[0];\
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\
}

MD_CALC_PAIR_ENERGY_KX(pLJ, md_LJ_periodic_energy_x)
MD_CALC_PAIR_ENERGY_KX(Gau, md_gaussian_energy_x)
MD_CALC_PAIR_ENERGY_KX(pGau, md_periodic_gaussian_energy_x)
MD_CALC_PAIR_ENERGY_KX(Cou, md_coulomb_energy_x)
MD_CALC_PAIR_ENERGY_KX(pCou, md_coulomb_periodic_energy_x)
MD_CALC_PAIR_ENERGY_KX(Ewald, md_coulomb_3d_ewald_energy_x)
MD_CALC_PAIR_ENERGY_KX(He, md_periodic_helium_energy_x)

kernel void md_calc_temperature_kx(global md_particle_t_x *particles, global md_num_t_x *output, int N, local md_num_t_x *target) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  if (gid < MD_DIMENSION_X*N) {
    int i = gid/MD_DIMENSION_X;
    int j = gid%MD_DIMENSION_X;
    target[lid] = particles[i].m*particles[i].v[j]*particles[i].v[j];
  }
  else
    target[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int gsize = get_local_size(0);
  int half_gsize = gsize/2;
  while (half_gsize > 0) {
    if (lid < half_gsize) {
      target[lid] += target[lid+half_gsize];
      if (gsize%2 != 0)
        if (lid == 0)
          target[0] += target[gsize-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gsize = half_gsize;
    half_gsize = gsize/2;
  }
  if (lid == 0)
    output[get_group_id(0)] = target[0];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

kernel void md_calc_pair_correlation_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int points, int image, md_num_t_x rmax, nd_rect_t_x rect) {
  int i = get_global_id(0);
  if (i >= N)
    return;
  int j;
  for (j = 0; j < points; ++j)
    den[i*points+j] = 0;
  md_num_t_x incre = rmax/points;
  md_num_t_x r;
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];
  int d;
  for (d = 0; d < MD_DIMENSION_X; ++d)
    p1[d] = particles[i].x[d];
  for (j = 0; j < N; ++j) {
    if (i == j)
      continue;
    for (d = 0; d < MD_DIMENSION_X; ++d)
      p2[d] = particles[j].x[d];
    if (image)
      r = md_minimum_image_distance_x(p1, p2, rect.L);
    else
      r = md_distance_x(p1, p2);
    int index = (int)(r/incre);
    if (index >= points)
      continue; //index = points-1;
    den[i*points+index] += 1.0/N;
  }
}

kernel void md_add_density_kx(global md_num_t_x *den, global md_num_t_x *fx, int N, int points) {
  int i = get_global_id(0);
  if (i >= points)
    return;
  int j;
  for (j = 0; j < N; ++j)
    fx[i] += den[j*points+i];
}

#define MD_PIMD_INDEX_X(l,j,P) (((l)-1)*(P)+(j)-1)

int md_pimd_next_index_x(int l, int j, int N2, int k, int P);

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

int md_pimd_prev_index_x(int l, int j, int N2, int k, int P);

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

md_num_t_x md_pimd_ENk_x(global md_particle_t_x *particles, int N2, int k, int image, md_num_t_x *L, int P, md_num_t_x omegaP);

md_num_t_x md_pimd_ENk_x(global md_particle_t_x *particles, int N2, int k, int image, md_num_t_x *L, int P, md_num_t_x omegaP) {
  int l, j;
  int index, index2;
  md_num_t_x res = 0;
  md_num_t_x d;
  int i;
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];
  for (l = N2-k+1; l <= N2; ++l)
    for (j = 1; j <= P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, P);
      index2 = md_pimd_next_index_x(l, j, N2, k, P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        p1[i] = particles[index].x[i];
        p2[i] = particles[index2].x[i];
      }
      if (image)
        d = md_minimum_image_distance_x(p1, p2, L);
      else
        d = md_distance_x(p1, p2);
      res += 0.5*particles[index].m*omegaP*omegaP*d*d;
    }
  return res;
}

kernel void md_pimd_fill_ENk_kx(global md_particle_t_x *particles, global md_num_t_x *ENk, global md_index_pair_t_x *indices, int count, int image, int P, md_num_t_x omegaP, nd_rect_t_x rect) {
  int id = get_global_id(0);
  if (id >= count)
    return;
  int l = indices[id].i+1;
  int j = indices[id].j+1;
  ENk[id] = md_pimd_ENk_x(particles, l, j, image, rect.L, P, omegaP);
}

md_num_t_x md_pimd_xexp_x(md_num_t_x k, md_num_t_x E, md_num_t_x EE, md_num_t_x beta, md_num_t_x vi);

md_num_t_x md_pimd_xexp_x(md_num_t_x k, md_num_t_x E, md_num_t_x EE, md_num_t_x beta, md_num_t_x vi) {
  if (vi == 0.0)
    return exp(-beta*E+EE);
  else
    return exp((k-1)*log(vi)-beta*E+EE);
}

kernel void md_pimd_xminE_kx(global md_num_t_x *ENk, global md_num_t_x *VBN, global int *Eindices, global md_num_t_x *output, int N, int N2, md_num_t_x vi, md_num_t_x beta, local md_num_t_x *target) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  int index, k;
  if (vi == 0.0) {
    index = Eindices[(N2-1)*N];
    if (lid == 0)
      output[get_group_id(0)] = beta*(ENk[index]+VBN[N2-1]);
    return;
  }
  if (gid < N2) {
    k = gid+1;
    index = Eindices[(N2-1)*N+k-1];
    target[lid] = -(k-1)*log(vi)+beta*(ENk[index]+VBN[N2-k]);
  }
  else
    target[lid] = 1e10;
  barrier(CLK_LOCAL_MEM_FENCE);
  int gsize = get_local_size(0);
  int half_gsize = gsize/2;
  while (half_gsize > 0) {
    if (lid < half_gsize) {
      target[lid] = min(target[lid+half_gsize], target[lid]);
      if (gsize%2 != 0)
        if (lid == 0)
          target[0] = min(target[gsize-1], target[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gsize = half_gsize;
    half_gsize = gsize/2;
  }
  if (lid == 0)
    output[get_group_id(0)] = target[0];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

kernel void md_pimd_fill_VB_kx(global md_num_t_x *ENk, global md_num_t_x *VBN, global int *Eindices, global md_num_t_x *output, int N, int N2, md_num_t_x vi, md_num_t_x beta, global md_num_t_x *minE, local md_num_t_x *target) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  md_num_t_x tmp = minE[N2];
  int index, k;
  if (vi == 0.0) {
    k = 1;
    index = Eindices[(N2-1)*N+k-1];
    if (lid == 0) {
      if (gid == 0)
        output[get_group_id(0)] = md_pimd_xexp_x(k, ENk[index]+VBN[N2-k], tmp, beta, vi);
      else
        output[get_group_id(0)] = 0.0;
    }
    return;
  }
  if (gid < N2) {
    k = gid+1;
    index = Eindices[(N2-1)*N+k-1];
    target[lid] = md_pimd_xexp_x(k, ENk[index]+VBN[N2-k], tmp, beta, vi);
  }
  else
    target[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int gsize = get_local_size(0);
  int half_gsize = gsize/2;
  while (half_gsize > 0) {
    if (lid < half_gsize) {
      target[lid] += target[lid+half_gsize];
      if (gsize%2 != 0)
        if (lid == 0)
          target[0] += target[gsize-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gsize = half_gsize;
    half_gsize = gsize/2;
  }
  if (lid == 0)
    output[get_group_id(0)] = target[0];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

void md_pimd_dENk_x(global md_particle_t_x *particles, int N2, int k, int l, int j, int image, local md_num_t_x *dENk, int P, md_num_t_x omegaP, md_num_t_x *L);

void md_pimd_dENk_x(global md_particle_t_x *particles, int N2, int k, int l, int j, int image, local md_num_t_x *dENk, int P, md_num_t_x omegaP, md_num_t_x *L) {
  if (!(l >= N2-k+1 && l <= N2)) {
    int i;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      dENk[i] = 0;
    return;
  }
  int index = MD_PIMD_INDEX_X(l, j, P);
  int index2 = md_pimd_next_index_x(l, j, N2, k, P);
  int index3 = md_pimd_prev_index_x(l, j, N2, k, P);
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    dENk[i] = 0;
    if (image) {
      dENk[i] += particles[index].m*omegaP*omegaP*md_minimum_image_x(particles[index].x[i]-particles[index2].x[i], L[i]);
      dENk[i] += particles[index].m*omegaP*omegaP*md_minimum_image_x(particles[index].x[i]-particles[index3].x[i], L[i]);
    }
    else {
      dENk[i] += particles[index].m*omegaP*omegaP*(particles[index].x[i]-particles[index2].x[i]);
      dENk[i] += particles[index].m*omegaP*omegaP*(particles[index].x[i]-particles[index3].x[i]);
    }
  }
}

kernel void md_pimd_fill_force_VB_kx(global md_particle_t_x *particles, global md_num_t_x *ENk, global md_num_t_x *VBN, global int *Eindices, global md_num_t_x *minE, int N, int P, int image, md_num_t_x vi, md_num_t_x omegaP, md_num_t_x beta, nd_rect_t_x rect, local md_num_t_x *res, local md_num_t_x *grad) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  if (gid >= N*P)
    return;
  int l = gid/P+1;
  int j = gid%P+1;
  int N2, k, index;
  int offset = lid*MD_DIMENSION_X*(N+1);
  int offset2 = lid*MD_DIMENSION_X*N;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    res[offset+i] = 0;
  for (N2 = 1; N2 <= N; ++N2) {
    md_num_t_x sum2 = 0;
    md_num_t_x tmp = minE[N2];
    /*for (k = 1; k <= N2; ++k) {
      if (vi == 0 && k-1 != 0)
        continue;
      index = Eindices[(N2-1)*N+k-1];
      sum2 += md_pimd_xexp_x(k, ENk[index]+VBN[N2-k], tmp, beta, vi);
    }*/
    sum2 = exp(-beta*VBN[N2]+log((md_num_t_x)N2)+tmp);
    for (k = 1; k <= N2; ++k)
      md_pimd_dENk_x(particles, N2, k, l, j, image, &grad[offset2+MD_DIMENSION_X*(k-1)], P, omegaP, rect.L);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      md_num_t_x sum = 0;
      for (k = 1; k <= N2; ++k) {
        if (vi == 0 && k-1 != 0)
          continue;
        index = Eindices[(N2-1)*N+k-1];
        sum += (grad[offset2+MD_DIMENSION_X*(k-1)+i]+res[offset+MD_DIMENSION_X*(N2-k)+i])*md_pimd_xexp_x(k, ENk[index]+VBN[N2-k], tmp, beta, vi);
      }
      res[offset+MD_DIMENSION_X*N2+i] = sum/sum2;
    }
  }
  index = MD_PIMD_INDEX_X(l, j, P);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    particles[index].f[i] += -res[offset+MD_DIMENSION_X*N+i]/particles[index].m;
}

kernel void md_pimd_calc_VBN_energy_kx(global md_num_t_x *ENk, global md_num_t_x *VBN, global int *Eindices, global md_num_t_x *output, global md_num_t_x *res, int N, int N2, md_num_t_x vi, md_num_t_x beta, global md_num_t_x *minE, local md_num_t_x *target) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  md_num_t_x tmp = minE[N2];
  int index, k;
  if (vi == 0.0) {
    k = 1;
    index = Eindices[(N2-1)*N+k-1];
    if (lid == 0) {
      if (gid == 0)
        output[get_group_id(0)] = (res[N2-k]-ENk[index])*md_pimd_xexp_x(k, ENk[index]+VBN[N2-k], tmp, beta, vi);
      else
        output[get_group_id(0)] = 0.0;
    }
    return;
  }
  if (gid < N2) {
    k = gid+1;
    index = Eindices[(N2-1)*N+k-1];
    target[lid] = (res[N2-k]-ENk[index])*md_pimd_xexp_x(k, ENk[index]+VBN[N2-k], tmp, beta, vi);
  }
  else
    target[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int gsize = get_local_size(0);
  int half_gsize = gsize/2;
  while (half_gsize > 0) {
    if (lid < half_gsize) {
      target[lid] += target[lid+half_gsize];
      if (gsize%2 != 0)
        if (lid == 0)
          target[0] += target[gsize-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gsize = half_gsize;
    half_gsize = gsize/2;
  }
  if (lid == 0)
    output[get_group_id(0)] = target[0];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

#define MD_PIMD_CALC_TRAP_FORCE_KX(name, func) kernel void md_pimd_calc_trap_force_ ## name ## _kx(global md_particle_t_x *particles, int N, int P, md_inter_params_t_x params, nd_rect_t_x rect) {\
  int id = get_global_id(0);\
  if (id >= N*P)\
    return;\
  md_num_t_x p1[MD_DIMENSION_X], f[MD_DIMENSION_X];\
  int d;\
  for (d = 0; d < MD_DIMENSION_X; ++d) {\
    p1[d] = particles[id].x[d];\
    f[d] = 0;\
  }\
  func(p1, f, rect.L, particles[id].m, &params);\
  for (d = 0; d < MD_DIMENSION_X; ++d)\
    particles[id].f[d] += f[d]/P;\
}\

MD_PIMD_CALC_TRAP_FORCE_KX(Htrap, md_harmonic_trap_force_x)
MD_PIMD_CALC_TRAP_FORCE_KX(Hubbard, md_hubbard_trap_force_x)

#define MD_PIMD_CALC_TRAP_ENERGY_KX(name, func) kernel void md_pimd_calc_trap_energy_ ## name ## _kx(global md_particle_t_x *particles, global md_num_t_x *output, int N, int P, md_inter_params_t_x params, nd_rect_t_x rect, local md_num_t_x *target) {\
  const int gid = get_global_id(0);\
  const int lid = get_local_id(0);\
  if (gid < N*P) {\
    md_num_t_x p1[MD_DIMENSION_X];\
    int d;\
    for (d = 0; d < MD_DIMENSION_X; ++d)\
      p1[d] = particles[gid].x[d];\
    target[lid] = func(p1, rect.L, particles[gid].m, &params)/P;\
  }\
  else\
    target[lid] = 0;\
  barrier(CLK_LOCAL_MEM_FENCE);\
  int gsize = get_local_size(0);\
  int half_gsize = gsize/2;\
  while (half_gsize > 0) {\
    if (lid < half_gsize) {\
      target[lid] += target[lid+half_gsize];\
      if (gsize%2 != 0)\
        if (lid == 0)\
          target[0] += target[gsize-1];\
    }\
    barrier(CLK_LOCAL_MEM_FENCE);\
    gsize = half_gsize;\
    half_gsize = gsize/2;\
  }\
  if (lid == 0)\
    output[get_group_id(0)] = target[0];\
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\
}\

MD_PIMD_CALC_TRAP_ENERGY_KX(Htrap, md_harmonic_trap_energy_x)
MD_PIMD_CALC_TRAP_ENERGY_KX(Hubbard, md_hubbard_trap_energy_x)

#define MD_PIMD_FILL_PAIR_FORCE_KX(name, func) kernel void md_pimd_fill_pair_force_ ## name ## _kx(global md_particle_t_x *particles, global md_num_t_x *forces, global md_index_pair_t_x *indices, int count, int N, int P, int jP, md_inter_params_t_x params, nd_rect_t_x rect) {\
  int id = get_global_id(0);\
  if (id >= count)\
    return;\
  int i = MD_PIMD_INDEX_X(indices[id].i+1, jP, P);\
  int j = MD_PIMD_INDEX_X(indices[id].j+1, jP, P);\
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X], f[MD_DIMENSION_X];\
  int d;\
  for (d = 0; d < MD_DIMENSION_X; ++d) {\
    p1[d] = particles[i].x[d];\
    p2[d] = particles[j].x[d];\
    f[d] = 0;\
  }\
  func(p1, p2, f, rect.L, particles[i].m, &params);\
  i = indices[id].i;\
  j = indices[id].j;\
  int index = i*N+j;\
  int index2 = j*N+i;\
  for (d = 0; d < MD_DIMENSION_X; ++d) {\
    forces[MD_DIMENSION_X*index+d] = f[d];\
    forces[MD_DIMENSION_X*index2+d] = -f[d];\
  }\
}

MD_PIMD_FILL_PAIR_FORCE_KX(pLJ, md_LJ_periodic_force_x)
MD_PIMD_FILL_PAIR_FORCE_KX(Gau, md_gaussian_force_x)
MD_PIMD_FILL_PAIR_FORCE_KX(pGau, md_periodic_gaussian_force_x)
MD_PIMD_FILL_PAIR_FORCE_KX(Cou, md_coulomb_force_x)
MD_PIMD_FILL_PAIR_FORCE_KX(pCou, md_coulomb_periodic_force_x)
MD_PIMD_FILL_PAIR_FORCE_KX(Ewald, md_coulomb_3d_ewald_force_x)
MD_PIMD_FILL_PAIR_FORCE_KX(He, md_periodic_helium_force_x)

kernel void md_pimd_update_pair_force_kx(global md_particle_t_x *particles, global md_num_t_x *forces, int N, int P, int jP) {
  int i = get_global_id(0);
  if (i >= N)
    return;
  int j, index, d;
  int index2 = MD_PIMD_INDEX_X(i+1, jP, P);
  for (j = 0; j < N; ++j) {
    if (i == j)
      continue;
    index = i*N+j;
    for (d = 0; d < MD_DIMENSION_X; ++d)
      particles[index2].f[d] += forces[MD_DIMENSION_X*index+d]/P;
  }
}

#define MD_PIMD_CALC_PAIR_ENERGY_KX(name, func) kernel void md_pimd_calc_pair_energy_ ## name ## _kx(global md_particle_t_x *particles, global md_num_t_x *output, global md_index_pair_t_x *indices, int count, int P, int jP, md_inter_params_t_x params, nd_rect_t_x rect, local md_num_t_x *target) {\
  const int gid = get_global_id(0);\
  const int lid = get_local_id(0);\
  if (gid < count) {\
    int i = MD_PIMD_INDEX_X(indices[gid].i+1, jP, P);\
    int j = MD_PIMD_INDEX_X(indices[gid].j+1, jP, P);\
    md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];\
    int d;\
    for (d = 0; d < MD_DIMENSION_X; ++d) {\
      p1[d] = particles[i].x[d];\
      p2[d] = particles[j].x[d];\
    }\
    target[lid] = func(p1, p2, rect.L, particles[i].m, &params)/P;\
  }\
  else\
    target[lid] = 0;\
  barrier(CLK_LOCAL_MEM_FENCE);\
  int gsize = get_local_size(0);\
  int half_gsize = gsize/2;\
  while (half_gsize > 0) {\
    if (lid < half_gsize) {\
      target[lid] += target[lid+half_gsize];\
      if (gsize%2 != 0)\
        if (lid == 0)\
          target[0] += target[gsize-1];\
    }\
    barrier(CLK_LOCAL_MEM_FENCE);\
    gsize = half_gsize;\
    half_gsize = gsize/2;\
  }\
  if (lid == 0)\
    output[get_group_id(0)] = target[0];\
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\
}

MD_PIMD_CALC_PAIR_ENERGY_KX(pLJ, md_LJ_periodic_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_KX(Gau, md_gaussian_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_KX(pGau, md_periodic_gaussian_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_KX(Cou, md_coulomb_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_KX(pCou, md_coulomb_periodic_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_KX(Ewald, md_coulomb_3d_ewald_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_KX(He, md_periodic_helium_energy_x)

kernel void md_pimd_calc_density_distribution_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int P, int points, int image, md_num_t_x rmax, nd_rect_t_x rect, nd_rect_t_x rect2) {
  int i = get_global_id(0);
  if (i >= N)
    return;
  int j;
  for (j = 0; j < points; ++j)
    den[i*points+j] = 0;
  md_num_t_x incre = rmax/points;
  md_num_t_x r;
  md_num_t_x p1[MD_DIMENSION_X];
  int index, d;
  for (j = 1; j <= P; ++j) {
    index = MD_PIMD_INDEX_X(i+1, j, P);
    for (d = 0; d < MD_DIMENSION_X; ++d)
      p1[d] = particles[index].x[d];
    if (image)
      r = md_minimum_image_distance_x(p1, rect2.L, rect.L);
    else
      r = md_distance_x(p1, rect2.L);
    index = (int)(r/incre);
    if (index >= points)
      continue; //index = points-1;
    den[i*points+index] += 1.0/P;
  }
}

kernel void md_pimd_fast_fill_ENk_1_kx(global md_particle_t_x *particles, global md_num_t_x *ENk, global int *Eindices, global md_num_t_x *Eint, int N, int image, int P, md_num_t_x omegaP, nd_rect_t_x rect) {
  int id = get_global_id(0);
  if (id >= N)
    return;
  int l = id+1;
  int j, i;
  int index, index2;
  md_num_t_x d;
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];
  md_num_t_x res = 0;
  for (j = 1; j < P; ++j) {
    index = MD_PIMD_INDEX_X(l, j, P);
    index2 = MD_PIMD_INDEX_X(l, j+1, P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      p1[i] = particles[index].x[i];
      p2[i] = particles[index2].x[i];
    }
    if (image)
      d = md_minimum_image_distance_x(p1, p2, rect.L);
    else
      d = md_distance_x(p1, p2);
    res += 0.5*particles[index].m*omegaP*omegaP*d*d;
  }
  Eint[l-1] = res;
  index = MD_PIMD_INDEX_X(l, P, P);
  index2 = MD_PIMD_INDEX_X(l, 1, P);
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    p1[i] = particles[index].x[i];
    p2[i] = particles[index2].x[i];
  }
  if (image)
    d = md_minimum_image_distance_x(p1, p2, rect.L);
  else
    d = md_distance_x(p1, p2);
  ENk[Eindices[(l-1)*N]] = res+0.5*particles[index].m*omegaP*omegaP*d*d;
}

kernel void md_pimd_fast_fill_ENk_2_kx(global md_particle_t_x *particles, global md_num_t_x *ENk, global int *Eindices, global md_num_t_x *Eint, int j, int N, int image, int P, md_num_t_x omegaP, nd_rect_t_x rect) {
  int l = get_global_id(0)+j;
  if (l > N)
    return;
  int index, index2, i;
  md_num_t_x d;
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];
  ENk[Eindices[(l-1)*N+j-1]] = ENk[Eindices[(l-1)*N+j-2]]+Eint[l-j];
  index = MD_PIMD_INDEX_X(l, P, P);
  index2 = MD_PIMD_INDEX_X(l-j+2, 1, P);
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    p1[i] = particles[index].x[i];
    p2[i] = particles[index2].x[i];
  }
  if (image)
    d = md_minimum_image_distance_x(p1, p2, rect.L);
  else
    d = md_distance_x(p1, p2);
  ENk[Eindices[(l-1)*N+j-1]] -= 0.5*particles[index].m*omegaP*omegaP*d*d;
  index = MD_PIMD_INDEX_X(l-j+1, P, P);
  index2 = MD_PIMD_INDEX_X(l-j+2, 1, P);
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    p1[i] = particles[index].x[i];
    p2[i] = particles[index2].x[i];
  }
  if (image)
    d = md_minimum_image_distance_x(p1, p2, rect.L);
  else
    d = md_distance_x(p1, p2);
  ENk[Eindices[(l-1)*N+j-1]] += 0.5*particles[index].m*omegaP*omegaP*d*d;
  index = MD_PIMD_INDEX_X(l, P, P);
  index2 = MD_PIMD_INDEX_X(l-j+1, 1, P);
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    p1[i] = particles[index].x[i];
    p2[i] = particles[index2].x[i];
  }
  if (image)
    d = md_minimum_image_distance_x(p1, p2, rect.L);
  else
    d = md_distance_x(p1, p2);
  ENk[Eindices[(l-1)*N+j-1]] += 0.5*particles[index].m*omegaP*omegaP*d*d;
}

kernel void md_pimd_fast_xminE_kx(global md_num_t_x *ENk, global md_num_t_x *VBN, global int *Eindices, global md_num_t_x *output, int N, int u, md_num_t_x vi, md_num_t_x beta, local md_num_t_x *target) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  int index, k;
  if (vi == 0.0) {
    index = Eindices[(u-1)*N];
    if (lid == 0)
      output[get_group_id(0)] = log((md_num_t_x)u)+beta*(ENk[index]+VBN[u]);
    return;
  }
  int l = u+gid;
  if (l <= N) {
    k = l-u+1;
    index = Eindices[(l-1)*N+k-1];
    target[lid] = -(k-1)*log(vi)+log((md_num_t_x)l)+beta*(ENk[index]+VBN[l]);
  }
  else
    target[lid] = 1e10;
  barrier(CLK_LOCAL_MEM_FENCE);
  int gsize = get_local_size(0);
  int half_gsize = gsize/2;
  while (half_gsize > 0) {
    if (lid < half_gsize) {
      target[lid] = min(target[lid+half_gsize], target[lid]);
      if (gsize%2 != 0)
        if (lid == 0)
          target[0] = min(target[gsize-1], target[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gsize = half_gsize;
    half_gsize = gsize/2;
  }
  if (lid == 0)
    output[get_group_id(0)] = target[0];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

md_num_t_x md_pimd_fast_xexp_x(md_num_t_x l, md_num_t_x k, md_num_t_x E, md_num_t_x EE, md_num_t_x beta, md_num_t_x vi);

md_num_t_x md_pimd_fast_xexp_x(md_num_t_x l, md_num_t_x k, md_num_t_x E, md_num_t_x EE, md_num_t_x beta, md_num_t_x vi) {
  if (vi == 0.0)
    return exp(-beta*E+EE-log(l));
  else
    return exp((k-1)*log(vi)-beta*E+EE-log(l));
}

void md_pimd_fast_dENk_x(global md_particle_t_x *particles, int index, int index2, int index3, int image, md_num_t_x *dENk, md_num_t_x omegaP, md_num_t_x *L);

void md_pimd_fast_dENk_x(global md_particle_t_x *particles, int index, int index2, int index3, int image, md_num_t_x *dENk, md_num_t_x omegaP, md_num_t_x *L) {
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    dENk[i] = 0;
    if (image) {
      dENk[i] += particles[index].m*omegaP*omegaP*md_minimum_image_x(particles[index].x[i]-particles[index2].x[i], L[i]);
      dENk[i] += particles[index].m*omegaP*omegaP*md_minimum_image_x(particles[index].x[i]-particles[index3].x[i], L[i]);
    }
    else {
      dENk[i] += particles[index].m*omegaP*omegaP*(particles[index].x[i]-particles[index2].x[i]);
      dENk[i] += particles[index].m*omegaP*omegaP*(particles[index].x[i]-particles[index3].x[i]);
    }
  }
}

kernel void md_pimd_fast_fill_VB_kx(global md_num_t_x *ENk, global md_num_t_x *VBN, global int *Eindices, global md_num_t_x *output, int N, int u, md_num_t_x vi, md_num_t_x beta, global md_num_t_x *minE, local md_num_t_x *target) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  int index, k, l;
  md_num_t_x tmp = minE[u];
  if (vi == 0.0) {
    k = 1;
    l = u;
    index = Eindices[(l-1)*N+k-1];
    if (lid == 0) {
      if (gid == 0)
        output[get_group_id(0)] = md_pimd_fast_xexp_x(l, k, ENk[index]+VBN[l], tmp, beta, vi);
      else
        output[get_group_id(0)] = 0.0;
    }
    return;
  }
  l = u+gid;
  if (l <= N) {
    k = l-u+1;
    index = Eindices[(l-1)*N+k-1];
    target[lid] = md_pimd_fast_xexp_x(l, k, ENk[index]+VBN[l], tmp, beta, vi);
  }
  else
    target[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int gsize = get_local_size(0);
  int half_gsize = gsize/2;
  while (half_gsize > 0) {
    if (lid < half_gsize) {
      target[lid] += target[lid+half_gsize];
      if (gsize%2 != 0)
        if (lid == 0)
          target[0] += target[gsize-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gsize = half_gsize;
    half_gsize = gsize/2;
  }
  if (lid == 0)
    output[get_group_id(0)] = target[0];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

kernel void md_pimd_fast_fill_G_kx(global md_num_t_x *ENk, global md_num_t_x *VBN, global md_num_t_x *tmpV, global md_num_t_x *G, global int *Eindices, int N, md_num_t_x vi, md_num_t_x beta) {
  const int gid = get_global_id(0);
  if (gid >= N*N)
    return;
  int l = gid/N+1;
  int j = gid%N+1;
  if (j > l+1)
    G[(l-1)*N+(j-1)] = 0;
  else if (j == l+1)
    G[(l-1)*N+(j-1)] = 1-exp(-beta*(VBN[l]+tmpV[l]-VBN[N]));
  else {
    int index = Eindices[(l-1)*N+l-j];
    if (vi == 0 && l-j != 0)
      G[(l-1)*N+(j-1)] = 0;
    else if (vi == 0 && l-j == 0)
      G[(l-1)*N+(j-1)] = exp(-beta*(VBN[j-1]+ENk[index]+tmpV[l]-VBN[N]))/l;
    else
      G[(l-1)*N+(j-1)] = exp(-beta*(VBN[j-1]+ENk[index]+tmpV[l]-VBN[N])+(l-j)*log(vi))/l;
  }
}

kernel void md_pimd_fast_fill_force_VB_1_kx(global md_particle_t_x *particles, int N, int P, int image, md_num_t_x omegaP, nd_rect_t_x rect) {
  const int gid = get_global_id(0);
  if (gid >= N*(P-2))
    return;
  int l = gid/(P-2)+1;
  int j = gid%(P-2)+2;
  md_num_t_x grad[MD_DIMENSION_X];
  int index, index2, index3;
  int i;
  index = MD_PIMD_INDEX_X(l, j, P);
  index2 = MD_PIMD_INDEX_X(l, j+1, P);
  index3 = MD_PIMD_INDEX_X(l, j-1, P);
  md_pimd_fast_dENk_x(particles, index, index2, index3, image, grad, omegaP, rect.L);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    particles[index].f[i] += -grad[i]/particles[index].m;
}

kernel void md_pimd_fast_fill_force_VB_2_kx(global md_particle_t_x *particles, global md_num_t_x *G, int N, int P, int image, md_num_t_x omegaP, nd_rect_t_x rect) {
  const int gid = get_global_id(0);
  if (gid >= N)
    return;
  int l = gid+1;
  int j;
  md_num_t_x grad[MD_DIMENSION_X];
  int index, index2, index3;
  int i;
  index = MD_PIMD_INDEX_X(l, 1, P);
  index2 = MD_PIMD_INDEX_X(l, 2, P);
  for (j = 1; j <= N; ++j) {
    index3 = MD_PIMD_INDEX_X(j, P, P);
    md_pimd_fast_dENk_x(particles, index, index2, index3, image, grad, omegaP, rect.L);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      particles[index].f[i] += -G[(j-1)*N+(l-1)]*grad[i]/particles[index].m;
  }
}

kernel void md_pimd_fast_fill_force_VB_3_kx(global md_particle_t_x *particles, global md_num_t_x *G, int N, int P, int image, md_num_t_x omegaP, nd_rect_t_x rect) {
  const int gid = get_global_id(0);
  if (gid >= N)
    return;
  int l = gid+1;
  int j;
  md_num_t_x grad[MD_DIMENSION_X];
  int index, index2, index3;
  int i;
  index = MD_PIMD_INDEX_X(l, P, P);
  index2 = MD_PIMD_INDEX_X(l, P-1, P);
  for (j = 1; j <= N; ++j) {
    index3 = MD_PIMD_INDEX_X(j, 1, P);
    md_pimd_fast_dENk_x(particles, index, index2, index3, image, grad, omegaP, rect.L);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      particles[index].f[i] += -G[(l-1)*N+(j-1)]*grad[i]/particles[index].m;
  }
}

kernel void md_add_stats_kx(global md_num_t_x *res, global md_num_t_x *out, int N, md_num_t_x mult) {
  const int gid = get_global_id(0);
  if (gid >= 1)
    return;
  int i;
  for (i = 0; i < N; ++i)
    res[0] += out[i]*mult;
}

kernel void md_min_kx(global md_num_t_x *minE, global md_num_t_x *out, int N, int N2) {
  const int gid = get_global_id(0);
  if (gid >= 1)
    return;
  md_num_t_x res = 1e10;
  int i;
  for (i = 0; i < N; ++i)
    if (out[i] < res)
      res = out[i];
  minE[N2] = res;
}

kernel void md_pimd_add_VBN_kx(global md_num_t_x *VBN, global md_num_t_x *out, global md_num_t_x *minE, int N, int N2, md_num_t_x beta) {
  const int gid = get_global_id(0);
  if (gid >= 1)
    return;
  md_num_t_x res = 0;
  int i;
  for (i = 0; i < N; ++i)
    res += out[i];
  md_num_t_x tmp = minE[N2];
  VBN[N2] = (tmp-log(res)+log((md_num_t_x)N2))/beta;
}

kernel void md_pimd_add_eVBN_kx(global md_num_t_x *eVBN, global md_num_t_x *VBN, global md_num_t_x *out, global md_num_t_x *minE, int N, int N2, md_num_t_x beta) {
  const int gid = get_global_id(0);
  if (gid >= 1)
    return;
  md_num_t_x res = 0;
  int i;
  for (i = 0; i < N; ++i)
    res += out[i];
  md_num_t_x tmp = minE[N2];
  eVBN[N2] = res/exp(-beta*VBN[N2]+log((md_num_t_x)N2)+tmp);
}

kernel void md_pimd_add_eVBN_stats_kx(global md_num_t_x *res, global md_num_t_x *eVBN, int N, int P, md_num_t_x beta) {
  const int gid = get_global_id(0);
  if (gid >= 1)
    return;
  res[0] += eVBN[N];
  res[0] += MD_DIMENSION_X*N*P/(2*beta);
}

kernel void md_pimd_fast_add_VBN_kx(global md_num_t_x *VBN, global md_num_t_x *out, global md_num_t_x *minE, int N, int u, md_num_t_x beta) {
  const int gid = get_global_id(0);
  if (gid >= 1)
    return;
  md_num_t_x res = 0;
  int i;
  for (i = 0; i < N; ++i)
    res += out[i];
  md_num_t_x tmp = minE[u];
  VBN[u-1] = (tmp-log(res))/beta;
}

md_num_t_x md_pimd_ENk2_x(global md_particle_t_x *particles, int N2, int k, int image, md_num_t_x *L, int P, md_num_t_x omegaP);

md_num_t_x md_pimd_ENk2_x(global md_particle_t_x *particles, int N2, int k, int image, md_num_t_x *L, int P, md_num_t_x omegaP) {
  int l;
  int index, index2, index3;
  md_num_t_x res = 0;
  md_num_t_x d, d2;
  int i;
  for (l = N2-k+1; l <= N2; ++l) {
    index = md_pimd_next_index_x(l, P, N2, k, P);
    index2 = MD_PIMD_INDEX_X(l, P, P);
    index3 = MD_PIMD_INDEX_X(l, 1, P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (image) {
        d = md_minimum_image_x(particles[index].x[i]-particles[index2].x[i], L[i]);
        d2 = md_minimum_image_x(particles[index].x[i]-particles[index3].x[i], L[i]);
      }
      else {
        d = particles[index].x[i]-particles[index2].x[i];
        d2 = particles[index].x[i]-particles[index3].x[i];
      }
      res += 0.5*particles[index].m*omegaP*omegaP*d*d2;
    }
  }
  return res;
}

kernel void md_pimd_fill_ENk2_kx(global md_particle_t_x *particles, global md_num_t_x *ENk, global md_index_pair_t_x *indices, int count, int image, int P, md_num_t_x omegaP, nd_rect_t_x rect) {
  int id = get_global_id(0);
  if (id >= count)
    return;
  int l = indices[id].i+1;
  int j = indices[id].j+1;
  ENk[id] = md_pimd_ENk2_x(particles, l, j, image, rect.L, P, omegaP);
}

md_num_t_x md_pimd_fast_ENk2_x(global md_particle_t_x *particles, int index, int index2, int index3, int image, md_num_t_x *L, md_num_t_x omegaP);

md_num_t_x md_pimd_fast_ENk2_x(global md_particle_t_x *particles, int index, int index2, int index3, int image, md_num_t_x *L, md_num_t_x omegaP) {
  md_num_t_x res = 0;
  md_num_t_x d, d2;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    if (image) {
      d = md_minimum_image_x(particles[index].x[i]-particles[index2].x[i], L[i]);
      d2 = md_minimum_image_x(particles[index].x[i]-particles[index3].x[i], L[i]);
    }
    else {
      d = particles[index].x[i]-particles[index2].x[i];
      d2 = particles[index].x[i]-particles[index3].x[i];
    }
    res += 0.5*particles[index].m*omegaP*omegaP*d*d2;
  }
  return res;
}

kernel void md_pimd_fast_fill_ENk2_1_kx(global md_particle_t_x *particles, global md_num_t_x *ENk, global int *Eindices, int N, int image, int P, md_num_t_x omegaP, nd_rect_t_x rect) {
  int id = get_global_id(0);
  if (id >= N)
    return;
  int l = id+1;
  int index, index2, index3;
  index = MD_PIMD_INDEX_X(l, 1, P);
  index2 = MD_PIMD_INDEX_X(l, P, P);
  index3 = MD_PIMD_INDEX_X(l, 1, P);
  ENk[Eindices[(l-1)*N]] = md_pimd_fast_ENk2_x(particles, index, index2, index3, image, rect.L, omegaP);
}

kernel void md_pimd_fast_fill_ENk2_2_kx(global md_particle_t_x *particles, global md_num_t_x *ENk, global int *Eindices, int j, int N, int image, int P, md_num_t_x omegaP, nd_rect_t_x rect) {
  int l = get_global_id(0)+j;
  if (l > N)
    return;
  int index, index2, index3;
  index = MD_PIMD_INDEX_X(l-j+2, 1, P);
  index2 = MD_PIMD_INDEX_X(l, P, P);
  index3 = MD_PIMD_INDEX_X(l, 1, P);
  ENk[Eindices[(l-1)*N+j-1]] = ENk[Eindices[(l-1)*N+j-2]]-md_pimd_fast_ENk2_x(particles, index, index2, index3, image, rect.L, omegaP);
  index = MD_PIMD_INDEX_X(l-j+2, 1, P);
  index2 = MD_PIMD_INDEX_X(l-j+1, P, P);
  index3 = MD_PIMD_INDEX_X(l-j+1, 1, P);
  ENk[Eindices[(l-1)*N+j-1]] += md_pimd_fast_ENk2_x(particles, index, index2, index3, image, rect.L, omegaP);
  index = MD_PIMD_INDEX_X(l-j+1, 1, P);
  index2 = MD_PIMD_INDEX_X(l, P, P);
  index3 = MD_PIMD_INDEX_X(l, 1, P);
  ENk[Eindices[(l-1)*N+j-1]] += md_pimd_fast_ENk2_x(particles, index, index2, index3, image, rect.L, omegaP);
}

kernel void md_pimd_calc_VBN2_energy_kx(global md_num_t_x *ENk, global md_num_t_x *ENk2, global md_num_t_x *VBN, global int *Eindices, global md_num_t_x *output, global md_num_t_x *res, int N, int N2, md_num_t_x vi, md_num_t_x beta, global md_num_t_x *minE, local md_num_t_x *target) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  md_num_t_x tmp = minE[N2];
  int index, k;
  if (vi == 0.0) {
    k = 1;
    index = Eindices[(N2-1)*N+k-1];
    if (lid == 0) {
      if (gid == 0)
        output[get_group_id(0)] = (res[N2-k]-ENk2[index])*md_pimd_xexp_x(k, ENk[index]+VBN[N2-k], tmp, beta, vi);
      else
        output[get_group_id(0)] = 0.0;
    }
    return;
  }
  if (gid < N2) {
    k = gid+1;
    index = Eindices[(N2-1)*N+k-1];
    target[lid] = (res[N2-k]-ENk2[index])*md_pimd_xexp_x(k, ENk[index]+VBN[N2-k], tmp, beta, vi);
  }
  else
    target[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int gsize = get_local_size(0);
  int half_gsize = gsize/2;
  while (half_gsize > 0) {
    if (lid < half_gsize) {
      target[lid] += target[lid+half_gsize];
      if (gsize%2 != 0)
        if (lid == 0)
          target[0] += target[gsize-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gsize = half_gsize;
    half_gsize = gsize/2;
  }
  if (lid == 0)
    output[get_group_id(0)] = target[0];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

kernel void md_pimd_add_eVBN2_stats_kx(global md_num_t_x *res, global md_num_t_x *eVBN, int N, md_num_t_x beta) {
  const int gid = get_global_id(0);
  if (gid >= 1)
    return;
  res[0] += eVBN[N];
  res[0] += MD_DIMENSION_X*N/(2*beta);
}

kernel void md_pimd_calc_virial_energy_kx(global md_particle_t_x *particles, global md_num_t_x *output, int N, int P, local md_num_t_x *target, int image, nd_rect_t_x rect) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  if (gid < N*P) {
    int l = gid/P+1;
    int j = gid%P+1;
    md_num_t_x res2 = 0;
    int i;
    int index, index2;
    index = MD_PIMD_INDEX_X(l, j, P);
    index2 = MD_PIMD_INDEX_X(l, 1, P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (image)
        res2 += -md_minimum_image_x(particles[index].x[i]-particles[index2].x[i], rect.L[i])*particles[index].f[i]*particles[index].m;
      else
        res2 += -(particles[index].x[i]-particles[index2].x[i])*particles[index].f[i]*particles[index].m;
    }
    target[lid] = res2;
  }
  else
    target[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int gsize = get_local_size(0);
  int half_gsize = gsize/2;
  while (half_gsize > 0) {
    if (lid < half_gsize) {
      target[lid] += target[lid+half_gsize];
      if (gsize%2 != 0)
        if (lid == 0)
          target[0] += target[gsize-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gsize = half_gsize;
    half_gsize = gsize/2;
  }
  if (lid == 0)
    output[get_group_id(0)] = target[0];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

#define MD_PIMD_FILL_PAIR_FORCE_2_KX(name, func) kernel void md_pimd_fill_pair_force_2_ ## name ## _kx(global md_particle_t_x *particles, global md_particle_t_x *particles2, global md_num_t_x *forces, int N2, int N, int P, int jP, md_inter_params_t_x params, nd_rect_t_x rect) {\
  int id = get_global_id(0);\
  if (id >= N*N2)\
    return;\
  int i = MD_PIMD_INDEX_X(id/N2+1, jP, P);\
  int j = MD_PIMD_INDEX_X(id%N2+1, jP, P);\
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X], f[MD_DIMENSION_X];\
  int d;\
  for (d = 0; d < MD_DIMENSION_X; ++d) {\
    p1[d] = particles[i].x[d];\
    p2[d] = particles2[j].x[d];\
    f[d] = 0;\
  }\
  func(p1, p2, f, rect.L, particles[i].m, &params);\
  for (d = 0; d < MD_DIMENSION_X; ++d) {\
    forces[MD_DIMENSION_X*id+d] = f[d];\
  }\
}

MD_PIMD_FILL_PAIR_FORCE_2_KX(pLJ, md_LJ_periodic_force_x)
MD_PIMD_FILL_PAIR_FORCE_2_KX(Gau, md_gaussian_force_x)
MD_PIMD_FILL_PAIR_FORCE_2_KX(pGau, md_periodic_gaussian_force_x)
MD_PIMD_FILL_PAIR_FORCE_2_KX(Cou, md_coulomb_force_x)
MD_PIMD_FILL_PAIR_FORCE_2_KX(pCou, md_coulomb_periodic_force_x)
MD_PIMD_FILL_PAIR_FORCE_2_KX(Ewald, md_coulomb_3d_ewald_force_x)
MD_PIMD_FILL_PAIR_FORCE_2_KX(He, md_periodic_helium_force_x)

kernel void md_pimd_update_pair_force_2_1_kx(global md_particle_t_x *particles, global md_num_t_x *forces, int N2, int N, int P, int jP) {
  int i = get_global_id(0);
  if (i >= N)
    return;
  int j, index, d;
  int index2 = MD_PIMD_INDEX_X(i+1, jP, P);
  for (j = 0; j < N2; ++j) {
    index = i*N2+j;
    for (d = 0; d < MD_DIMENSION_X; ++d) {
      particles[index2].f[d] += forces[MD_DIMENSION_X*index+d]/P;
      forces[MD_DIMENSION_X*index+d] *= particles[index2].m;
    }
  }
}

kernel void md_pimd_update_pair_force_2_2_kx(global md_particle_t_x *particles, global md_num_t_x *forces, int N2, int N, int P, int jP) {
  int i = get_global_id(0);
  if (i >= N2)
    return;
  int j, index, d;
  int index2 = MD_PIMD_INDEX_X(i+1, jP, P);
  for (j = 0; j < N; ++j) {
    index = j*N2+i;
    for (d = 0; d < MD_DIMENSION_X; ++d)
      particles[index2].f[d] -= forces[MD_DIMENSION_X*index+d]/P/particles[index2].m;
  }
}

#define MD_PIMD_CALC_PAIR_ENERGY_2_KX(name, func) kernel void md_pimd_calc_pair_energy_2_ ## name ## _kx(global md_particle_t_x *particles, global md_particle_t_x *particles2, global md_num_t_x *output, int N2, int N, int P, int jP, md_inter_params_t_x params, nd_rect_t_x rect, local md_num_t_x *target) {\
  const int gid = get_global_id(0);\
  const int lid = get_local_id(0);\
  if (gid < N*N2) {\
    int i = MD_PIMD_INDEX_X(gid/N2+1, jP, P);\
    int j = MD_PIMD_INDEX_X(gid%N2+1, jP, P);\
    md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];\
    int d;\
    for (d = 0; d < MD_DIMENSION_X; ++d) {\
      p1[d] = particles[i].x[d];\
      p2[d] = particles2[j].x[d];\
    }\
    target[lid] = func(p1, p2, rect.L, particles[i].m, &params)/P;\
  }\
  else\
    target[lid] = 0;\
  barrier(CLK_LOCAL_MEM_FENCE);\
  int gsize = get_local_size(0);\
  int half_gsize = gsize/2;\
  while (half_gsize > 0) {\
    if (lid < half_gsize) {\
      target[lid] += target[lid+half_gsize];\
      if (gsize%2 != 0)\
        if (lid == 0)\
          target[0] += target[gsize-1];\
    }\
    barrier(CLK_LOCAL_MEM_FENCE);\
    gsize = half_gsize;\
    half_gsize = gsize/2;\
  }\
  if (lid == 0)\
    output[get_group_id(0)] = target[0];\
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\
}

MD_PIMD_CALC_PAIR_ENERGY_2_KX(pLJ, md_LJ_periodic_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_2_KX(Gau, md_gaussian_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_2_KX(pGau, md_periodic_gaussian_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_2_KX(Cou, md_coulomb_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_2_KX(pCou, md_coulomb_periodic_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_2_KX(Ewald, md_coulomb_3d_ewald_energy_x)
MD_PIMD_CALC_PAIR_ENERGY_2_KX(He, md_periodic_helium_energy_x)

kernel void md_pimd_calc_ITCF_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int P, int points, int pi, md_num_t_x rmin, md_num_t_x rmax) {
  int l = get_global_id(0);
  if (l >= N)
    return;
  int j;
  for (j = 0; j < points; ++j)
    den[l*points+j] = 0;
  int n = (int)sqrt((md_num_t_x)points);
  md_num_t_x incre = (rmax-rmin)/n;
  int index, index2;
  for (j = 1; j <= N; ++j) {
    index = MD_PIMD_INDEX_X(l, 1, P);
    index2 = MD_PIMD_INDEX_X(j, pi, P);
    int i1 = (int)((particles[index].x[0]-rmin)/incre);
    int i2 = (int)((particles[index2].x[0]-rmin)/incre);
    int idx = i1*n+i2;
    if (idx < 0)
      idx = 0;
    if (idx >= points)
      idx = points-1;
    den[l*points+idx] += 1.0/P;
  }
}

kernel void md_pimd_calc_ITCF_2_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int P, int points, int pi, md_num_t_x rmin, md_num_t_x rmax, global md_particle_t_x *particles2, int N2) {
  int l = get_global_id(0);
  if (l >= N)
    return;
  int j;
  for (j = 0; j < points; ++j)
    den[l*points+j] = 0;
  int n = (int)sqrt((md_num_t_x)points);
  md_num_t_x incre = (rmax-rmin)/n;
  int index, index2;
  for (j = 1; j <= N2; ++j) {
    index = MD_PIMD_INDEX_X(l, 1, P);
    index2 = MD_PIMD_INDEX_X(j, pi, P);
    int i1 = (int)((particles[index].x[0]-rmin)/incre);
    int i2 = (int)((particles2[index2].x[0]-rmin)/incre);
    int idx = i1*n+i2;
    if (idx < 0)
      idx = 0;
    if (idx >= points)
      idx = points-1;
    den[l*points+idx] += 1.0/P;
    idx = i2*n+i1;
    if (idx < 0)
      idx = 0;
    if (idx >= points)
      idx = points-1;
    den[l*points+idx] += 1.0/P;
  }
}

kernel void md_pimd_calc_pair_correlation_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int P, int points, int pi, md_num_t_x rmax, md_num_t_x norm, int image, nd_rect_t_x rect) {
  int l = get_global_id(0);
  if (l >= N)
    return;
  int j;
  for (j = 0; j < points; ++j)
    den[l*points+j] = 0;
  md_num_t_x incre = rmax/points;
  md_num_t_x r;
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];
  int d;
  int index, index2;
  index = MD_PIMD_INDEX_X(l, 1, P);
  for (d = 0; d < MD_DIMENSION_X; ++d)
    p1[d] = particles[index].x[d];
  for (j = 0; j < N; ++j) {
    if (l == j)
      continue;
    index2 = MD_PIMD_INDEX_X(j, pi, P);
    for (d = 0; d < MD_DIMENSION_X; ++d)
      p2[d] = particles[index2].x[d];
    if (image)
      r = md_minimum_image_distance_x(p1, p2, rect.L);
    else
      r = md_distance_x(p1, p2);
    int index3 = (int)(r/incre);
    if (index3 >= points)
      continue; //index3 = points-1;
    den[l*points+index3] += 1.0/norm;
  }
}

kernel void md_pimd_calc_pair_correlation_2_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int P, int points, int pi, md_num_t_x rmax, md_num_t_x norm, int image, nd_rect_t_x rect, global md_particle_t_x *particles2, int N2) {
  int l = get_global_id(0);
  if (l >= N)
    return;
  int j;
  for (j = 0; j < points; ++j)
    den[l*points+j] = 0;
  md_num_t_x incre = rmax/points;
  md_num_t_x r;
  md_num_t_x p1[MD_DIMENSION_X], p2[MD_DIMENSION_X];
  int d;
  int index, index2;
  index = MD_PIMD_INDEX_X(l, 1, P);
  for (d = 0; d < MD_DIMENSION_X; ++d)
    p1[d] = particles[index].x[d];
  for (j = 0; j < N2; ++j) {
    index2 = MD_PIMD_INDEX_X(j, pi, P);
    for (d = 0; d < MD_DIMENSION_X; ++d)
      p2[d] = particles2[index2].x[d];
    if (image)
      r = md_minimum_image_distance_x(p1, p2, rect.L);
    else
      r = md_distance_x(p1, p2);
    int index3 = (int)(r/incre);
    if (index3 >= points)
      continue; //index3 = points-1;
    den[l*points+index3] += 1.0/norm;
  }
}

kernel void md_calc_Sk_structure_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int points, md_num_t_x q0, md_num_t_x qincre) {
  int j = get_global_id(0);
  if (j >= points)
    return;
  int l, i;
  md_num_t_x q = q0+j*qincre;
  md_num_t_x res = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    md_num_t_x sum = 0;
    for (l = 0; l < N; ++l)
      sum += cos(q*particles[l].x[i]);
    res += sum*sum/N;
    sum = 0;
    for (l = 0; l < N; ++l)
      sum += sin(q*particles[l].x[i]);
    res += sum*sum/N;
  }
  res /= MD_DIMENSION_X;
  den[j] += res;
}

kernel void md_pimd_calc_Sk_structure_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int P, int points, md_num_t_x q0, md_num_t_x qincre, int pi) {
  int j = get_global_id(0);
  if (j >= points)
    return;
  int l, i;
  md_num_t_x q = q0+j*qincre;
  md_num_t_x res = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    md_num_t_x sum = 0;
    md_num_t_x sum2 = 0;
    int index;
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, 1, P);
      sum += cos(q*particles[index].x[i]);
    }
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, pi, P);
      sum2 += cos(q*particles[index].x[i]);
    }
    res += sum*sum2/N;
    sum = 0;
    sum2 = 0;
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, 1, P);
      sum += sin(q*particles[index].x[i]);
    }
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, pi, P);
      sum2 += sin(q*particles[index].x[i]);
    }
    res += sum*sum2/N;
  }
  res /= MD_DIMENSION_X;
  den[j] += res;
}

kernel void md_pimd_calc_Sk_structure_2_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int P, int points, md_num_t_x q0, md_num_t_x qincre, int pi, global md_particle_t_x *particles2, int N2, int P2) {
  int j = get_global_id(0);
  if (j >= points)
    return;
  int l, i;
  md_num_t_x q = q0+j*qincre;
  md_num_t_x res = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    md_num_t_x sum = 0;
    md_num_t_x sum2 = 0;
    int index;
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, 1, P);
      sum += cos(q*particles[index].x[i]);
    }
    for (l = 0; l < N2; ++l) {
      index = MD_PIMD_INDEX_X(l+1, pi, P2);
      sum2 += cos(q*particles2[index].x[i]);
    }
    res += sum*sum2/N;
    sum = 0;
    sum2 = 0;
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, 1, P);
      sum += sin(q*particles[index].x[i]);
    }
    for (l = 0; l < N2; ++l) {
      index = MD_PIMD_INDEX_X(l+1, pi, P2);
      sum2 += sin(q*particles2[index].x[i]);
    }
    res += sum*sum2/N;
  }
  res /= MD_DIMENSION_X;
  den[j] += res;
}

kernel void md_pimd_calc_Sk_structure_full_2_kx(global md_particle_t_x *particles, global md_num_t_x *den, int N, int P, int points, md_num_t_x q0, md_num_t_x qincre, int pi, global md_particle_t_x *particles2, int N2, int P2) {
  int j = get_global_id(0);
  if (j >= points)
    return;
  int l, i;
  md_num_t_x q = q0+j*qincre;
  md_num_t_x res = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    md_num_t_x sum = 0;
    md_num_t_x sum2 = 0;
    int index;
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, 1, P);
      sum += cos(q*particles[index].x[i]);
    }
    for (l = 0; l < N2; ++l) {
      index = MD_PIMD_INDEX_X(l+1, 1, P2);
      sum += cos(q*particles2[index].x[i]);
    }
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, pi, P);
      sum2 += cos(q*particles[index].x[i]);
    }
    for (l = 0; l < N2; ++l) {
      index = MD_PIMD_INDEX_X(l+1, pi, P2);
      sum2 += cos(q*particles2[index].x[i]);
    }
    res += sum*sum2/(N+N2);
    sum = 0;
    sum2 = 0;
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, 1, P);
      sum += sin(q*particles[index].x[i]);
    }
    for (l = 0; l < N2; ++l) {
      index = MD_PIMD_INDEX_X(l+1, 1, P2);
      sum += sin(q*particles2[index].x[i]);
    }
    for (l = 0; l < N; ++l) {
      index = MD_PIMD_INDEX_X(l+1, pi, P);
      sum2 += sin(q*particles[index].x[i]);
    }
    for (l = 0; l < N2; ++l) {
      index = MD_PIMD_INDEX_X(l+1, pi, P2);
      sum2 += sin(q*particles2[index].x[i]);
    }
    res += sum*sum2/(N+N2);
  }
  res /= MD_DIMENSION_X;
  den[j] += res;
}