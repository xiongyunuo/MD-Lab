#include "md_simulation_x.h"
#include "md_util_x.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef MD_USE_OPENCL_X
void md_get_pair_force_info_x(md_pair_force_t_x pf, const char *prefix, const char *postfix, char *dest, md_inter_params_t_x *params) {
  dest[0] = '\0';
  strcat(dest, prefix);
  if (pf == md_LJ_periodic_force_x)
    strcat(dest, "pLJ");
  else if (pf == md_gaussian_force_x) {
    strcat(dest, "Gau");
    params->params[0] = md_gaussian_strength_x;
    params->params[1] = md_gaussian_range_x;
  }
  else if (pf == md_periodic_gaussian_force_x) {
    strcat(dest, "pGau");
    params->params[0] = md_gaussian_strength_x;
    params->params[1] = md_gaussian_range_x;
  }
  else if (pf == md_coulomb_force_x) {
    strcat(dest, "Cou");
    params->params[0] = md_coulomb_strength_x;
  }
  else if (pf == md_coulomb_periodic_force_x) {
    strcat(dest, "pCou");
    params->params[0] = md_coulomb_strength_x;
  }
  else if (pf == md_coulomb_3d_ewald_force_x) {
    strcat(dest, "Ewald");
    params->params[0] = md_coulomb_strength_x;
    params->params[1] = md_coulomb_truc_x;
    params->params[2] = md_coulomb_n_sum_x;
  }
  else if (pf == md_periodic_helium_force_x) {
    strcat(dest, "He");
    params->params[0] = md_he_eps_x;
    params->params[1] = md_he_A_x;
    params->params[2] = md_he_alpha_x;
    params->params[3] = md_he_C6_x;
    params->params[4] = md_he_C8_x;
    params->params[5] = md_he_C10_x;
    params->params[6] = md_he_D_x;
    params->params[7] = md_he_rm_x;
  }
  strcat(dest, postfix);
}

void md_get_trap_force_info_x(md_trap_force_t_x tf, const char *prefix, const char *postfix, char *dest, md_inter_params_t_x *params) {
  dest[0] = '\0';
  strcat(dest, prefix);
  if (tf == md_harmonic_trap_force_x) {
    strcat(dest, "Htrap");
    params->params[0] = md_trap_frequency_x;
    int i;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      params->params[1+i] = md_trap_center_x[i];
  }
  else if (tf == md_hubbard_trap_force_x) {
    strcat(dest, "Hubbard");
    params->params[0] = md_hubbard_trap_strength_x;
    params->params[1] = md_hubbard_trap_frequency_x;
  }
  strcat(dest, postfix);
}
#endif

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

void md_periodic_gaussian_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m) {
  int i;
  md_num_t_x d = md_minimum_image_distance_x(p1, p2, L);
  md_num_t_x mult = (md_gaussian_strength_x/(M_PI*md_gaussian_range_x*md_gaussian_range_x))*exp(-d*d/(md_gaussian_range_x*md_gaussian_range_x));
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += mult*(2*md_minimum_image_x(p1[i]-p2[i], L[i]))/(md_gaussian_range_x*md_gaussian_range_x)/m;
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

md_num_t_x md_hubbard_trap_strength_x;
md_num_t_x md_hubbard_trap_frequency_x;

void md_set_hubbard_trap_strength_x(md_num_t_x h) {
  md_hubbard_trap_strength_x = h;
}

void md_set_hubbard_trap_frequency_x(md_num_t_x k) {
  md_hubbard_trap_frequency_x = k;
}

void md_hubbard_trap_force_x(md_num_t_x *p1, md_num_t_x *f, md_num_t_x *MD_UNUSED_X(L), md_num_t_x m) {
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    f[i] += 2*md_hubbard_trap_strength_x*md_hubbard_trap_frequency_x*cos(md_hubbard_trap_frequency_x*p1[i])*sin(md_hubbard_trap_frequency_x*p1[i])/m;
}

md_num_t_x md_he_eps_x;
md_num_t_x md_he_A_x;
md_num_t_x md_he_alpha_x;
md_num_t_x md_he_C6_x;
md_num_t_x md_he_C8_x;
md_num_t_x md_he_C10_x;
md_num_t_x md_he_D_x;
md_num_t_x md_he_rm_x;

void md_set_helium_parameters_x(md_num_t_x eps, md_num_t_x A, md_num_t_x alpha, md_num_t_x C6, md_num_t_x C8, md_num_t_x C10, md_num_t_x D, md_num_t_x rm) {
  md_he_eps_x = eps;
  md_he_A_x = A;
  md_he_alpha_x = alpha;
  md_he_C6_x = C6;
  md_he_C8_x = C8;
  md_he_C10_x = C10;
  md_he_D_x = D;
  md_he_rm_x = rm;
}

void md_periodic_helium_force_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *f, md_num_t_x *L, md_num_t_x m) {
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

void md_reset_force_x(md_simulation_t_x *sim) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(sim, context);
  cl_kernel kernel = NULL;
  if (sim->rf_kernel != NULL)
    kernel = sim->rf_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_reset_force_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(int), &sim->N);
    sim->rf_kernel = kernel;
  }
  int size[1];
  size[0] = MD_DIMENSION_X*sim->N;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  sim->queue = queue;
  //cl_event events[1];
  //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseEvent(events[0]);
#else
  int i, j;
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j)
      sim->particles[i].f[j] = 0;
#endif
}

void md_calc_pair_force_x(md_simulation_t_x *sim, md_pair_force_t_x pf) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(sim, context);
  md_inter_params_t_x params;
  nd_rect_t_x rect2;
  int i;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rect2.L[i] = L[i];
  }
  char kname[64];
  md_get_pair_force_info_x(pf, "md_fill_pair_force_", "_kx", kname, &params);
  if (sim->cpf_kernel != NULL && strcmp(sim->kname, kname)) {
    clReleaseKernel(sim->cpf_kernel);
    sim->cpf_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (sim->cpf_kernel != NULL)
    kernel = sim->cpf_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sim->pair_ex_mem);
    clSetKernelArg(kernel, 3, sizeof(int), &sim->pcount_ex);
    clSetKernelArg(kernel, 4, sizeof(int), &sim->N);
    sim->cpf_kernel = kernel;
    strcpy(sim->kname, kname);
  }
  clSetKernelArg(kernel, 5, sizeof(md_inter_params_t_x), &params);
  clSetKernelArg(kernel, 6, sizeof(nd_rect_t_x), &rect2);
  cl_mem forces = NULL;
  if (sim->forces != NULL)
    forces = sim->forces;
  else {
    forces = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*sim->N*sim->N*MD_DIMENSION_X, NULL, NULL);
    sim->forces = forces;
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &forces);
  int size[1];
  size[0] = sim->pcount_ex;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel2 = NULL;
  if (sim->cpf_kernel2 != NULL)
    kernel2 = sim->cpf_kernel2;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_update_pair_force_kx", &status);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel2, 2, sizeof(int), &sim->N);
    sim->cpf_kernel2 = kernel2;
  }
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &forces);
  size[0] = sim->N;
  md_get_work_size_x(kernel2, device, 1, size, global, local);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  sim->queue = queue;
  //cl_event events[1];
  //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseKernel(kernel2);
  //clReleaseMemObject(forces);
  //clReleaseEvent(events[0]);
#else
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
#endif
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
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(sim, context);
  cl_kernel kernel = NULL;
  if (sim->uVV1_kernel != NULL)
    kernel = sim->uVV1_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_update_nhc_VV3_1_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sim->nhcs_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sim->f0s_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &sim->N);
    clSetKernelArg(kernel, 5, sizeof(int), &sim->Nf);
    sim->uVV1_kernel = kernel;
  }
  clSetKernelArg(kernel, 3, sizeof(md_num_t_x), &h);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &sim->T);
  int size[1];
  size[0] = sim->Nf;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  sim->queue = queue;
  //cl_event events[1];
  //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseEvent(events[0]);
#else
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
#endif
}

void md_update_nhc_VV3_2_x(md_simulation_t_x *sim, md_num_t_x h) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(sim, context);
  cl_kernel kernel = NULL;
  if (sim->uVV2_kernel != NULL)
    kernel = sim->uVV2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_update_nhc_VV3_2_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sim->nhcs_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sim->f0s_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &sim->N);
    clSetKernelArg(kernel, 5, sizeof(int), &sim->Nf);
    sim->uVV2_kernel = kernel;
  }
  clSetKernelArg(kernel, 3, sizeof(md_num_t_x), &h);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &sim->T);
  int size[1];
  size[0] = sim->Nf;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  sim->queue = queue;
  //cl_event events[1];
  //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseEvent(events[0]);
#else
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
#endif
}

void md_periodic_boundary_x(md_simulation_t_x *sim) {
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = (nd_rect_t_x *)sim->box->box;
#ifdef MD_USE_OPENCL_X
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    int plat;
    cl_int status;
    md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
    md_simulation_to_context_x(sim, context);
    cl_kernel kernel = NULL;
    if (sim->pb_kernel != NULL)
      kernel = sim->pb_kernel;
    else {
      kernel = clCreateKernel(md_programs_x[plat], "md_rect_periodic_boundary_kx", &status);
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
      clSetKernelArg(kernel, 1, sizeof(int), &sim->N);
      sim->pb_kernel = kernel;
    }
    clSetKernelArg(kernel, 2, sizeof(nd_rect_t_x), rect);
    int size[1];
    size[0] = MD_DIMENSION_X*sim->N;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    sim->queue = queue;
    //cl_event events[1];
    //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
    //clWaitForEvents(1, events);
    //clReleaseKernel(kernel);
    //clReleaseEvent(events[0]);
#else
    int i, j;
    md_num_t_x *L = rect->L;
    for (i = 0; i < sim->N; ++i)
      for (j = 0; j < MD_DIMENSION_X; ++j) {
        if (sim->particles[i].x[j] < 0)
          sim->particles[i].x[j] += L[j];
        else if (sim->particles[i].x[j] > L[j])
          sim->particles[i].x[j] -= L[j];
      }
#endif
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
  md_simulation_sync_host_x(sim, 0);
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
  md_simulation_sync_host_x(sim, 0);
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