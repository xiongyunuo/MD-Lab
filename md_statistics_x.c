#include "md_statistics_x.h"
#include "md_simulation_x.h"
#include "md_util_x.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef MD_USE_OPENCL_X
void md_get_pair_energy_info_x(md_pair_energy_t_x pe, const char *prefix, const char *postfix, char *dest, md_inter_params_t_x *params) {
  dest[0] = '\0';
  strcat(dest, prefix);
  if (pe == md_LJ_periodic_energy_x)
    strcat(dest, "pLJ");
  else if (pe == md_gaussian_energy_x) {
    strcat(dest, "Gau");
    params->params[0] = md_gaussian_strength_x;
    params->params[1] = md_gaussian_range_x;
  }
  else if (pe == md_periodic_gaussian_energy_x) {
    strcat(dest, "pGau");
    params->params[0] = md_gaussian_strength_x;
    params->params[1] = md_gaussian_range_x;
  }
  else if (pe == md_coulomb_energy_x) {
    strcat(dest, "Cou");
    params->params[0] = md_coulomb_strength_x;
  }
  else if (pe == md_coulomb_periodic_energy_x) {
    strcat(dest, "pCou");
    params->params[0] = md_coulomb_strength_x;
  }
  else if (pe == md_coulomb_3d_ewald_energy_x) {
    strcat(dest, "Ewald");
    params->params[0] = md_coulomb_strength_x;
    params->params[1] = md_coulomb_truc_x;
    params->params[2] = md_coulomb_n_sum_x;
  }
  else if (pe == md_periodic_helium_energy_x) {
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

void md_get_trap_energy_info_x(md_trap_energy_t_x te, const char *prefix, const char *postfix, char *dest, md_inter_params_t_x *params) {
  dest[0] = '\0';
  strcat(dest, prefix);
  if (te == md_harmonic_trap_energy_x) {
    strcat(dest, "Htrap");
    params->params[0] = md_trap_frequency_x;
    int i;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      params->params[1+i] = md_trap_center_x[i];
  }
  else if (te == md_hubbard_trap_energy_x) {
    strcat(dest, "Hubbard");
    params->params[0] = md_hubbard_trap_strength_x;
    params->params[1] = md_hubbard_trap_frequency_x;
  }
  strcat(dest, postfix);
}
#endif

#ifdef MD_USE_OPENCL_X
cl_int md_stats_to_context_x(md_stats_t_x *stats, cl_context context) {
  cl_int status;
  if (context == stats->context && stats->e_mem != NULL)
    return 0;
  else if (context != stats->context && stats->e_mem != NULL) {
    status = clReleaseMemObject(stats->e_mem);
    status |= clReleaseMemObject(stats->T_mem);
    status |= clReleaseMemObject(stats->fx_mem);
    stats->e_mem = NULL;
    stats->T_mem = NULL;
    stats->fx_mem = NULL;
    stats->context = NULL;
    stats->queue = NULL;
    if (status != CL_SUCCESS)
      return status;
  }
  cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  stats->e_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x), &stats->es[stats->N-1], &status);
  if (status != CL_SUCCESS)
    return status;
  stats->T_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x), &stats->Ts[stats->N-1], &status);
  if (status != CL_SUCCESS)
    return status;
  stats->fx_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*stats->points, stats->fxs[stats->N-1], &status);
  if (status != CL_SUCCESS)
    return status;
  stats->context = context;
  stats->queue = NULL;
  return 0;
}
#endif

md_stats_t_x *md_stats_sync_host_x(md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  if (stats->e_mem == NULL)
    return stats;
  cl_command_queue queue = stats->queue;
  if (queue == NULL) {
    int i;
    for (i = 0; i < md_num_platforms_x; ++i)
      if (md_contexts_x[i] == stats->context) {
        queue = md_command_queues_x[i][0];
        break;
      }
  }
  cl_int status;
  cl_event events[3];
  status = clEnqueueReadBuffer(queue, stats->e_mem, CL_FALSE, 0, sizeof(md_num_t_x), &stats->es[stats->N-1], 0, NULL, &events[0]);
  status |= clEnqueueReadBuffer(queue, stats->T_mem, CL_FALSE, 0, sizeof(md_num_t_x), &stats->Ts[stats->N-1], 0, NULL, &events[1]);
  status |= clEnqueueReadBuffer(queue, stats->fx_mem, CL_FALSE, 0, sizeof(md_num_t_x)*stats->points, stats->fxs[stats->N-1], 0, NULL, &events[2]);
  if (status != CL_SUCCESS)
    return NULL;
  status = clWaitForEvents(3, events);
  status |= clReleaseEvent(events[0]);
  status |= clReleaseEvent(events[1]);
  status |= clReleaseEvent(events[2]);
  if (status != CL_SUCCESS)
    return NULL;
  status = clReleaseMemObject(stats->e_mem);
  status |= clReleaseMemObject(stats->T_mem);
  status |= clReleaseMemObject(stats->fx_mem);
  stats->e_mem = NULL;
  stats->T_mem = NULL;
  stats->fx_mem = NULL;
  stats->context = NULL;
  stats->queue = NULL;
  if (status != CL_SUCCESS)
    return NULL;
#endif
  return stats;
}

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
#ifdef MD_USE_OPENCL_X
  stats->e_mem = NULL;
  stats->T_mem = NULL;
  stats->fx_mem = NULL;
  stats->context = NULL;
  stats->queue = NULL;
#endif
  return stats;
}

md_stats_t_x *md_update_stats_x(md_stats_t_x *stats) {
  stats->count++;
  if (stats->count >= MD_MAX_STATS_COUNT) {
#ifdef MD_USE_OPENCL_X
    md_stats_sync_host_x(stats);
#endif
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
#ifdef MD_USE_OPENCL_X
  md_stats_sync_host_x(stats);
#endif
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

md_num_t_x md_periodic_gaussian_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m)) {
  md_num_t_x d = md_minimum_image_distance_x(p1, p2, L);
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

md_num_t_x md_hubbard_trap_energy_x(md_num_t_x *p1, md_num_t_x *MD_UNUSED_X(L), md_num_t_x MD_UNUSED_X(m)) {
  md_num_t_x res = 0;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    res += md_hubbard_trap_strength_x*pow(cos(md_hubbard_trap_frequency_x*p1[i]), 2);
  return res;
}

md_num_t_x md_periodic_helium_energy_x(md_num_t_x *p1, md_num_t_x *p2, md_num_t_x *L, md_num_t_x MD_UNUSED_X(m)) {
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

void md_calc_pair_energy_x(md_simulation_t_x *sim, md_pair_energy_t_x pe, md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(sim, context);
  md_stats_to_context_x(stats, context);
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
  md_get_pair_energy_info_x(pe, "md_calc_pair_energy_", "_kx", kname, &params);
  if (sim->cpe_kernel != NULL && strcmp(sim->kname2, kname)) {
    clReleaseKernel(sim->cpe_kernel);
    sim->cpe_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (sim->cpe_kernel != NULL)
    kernel = sim->cpe_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sim->pair_ex_mem);
    clSetKernelArg(kernel, 3, sizeof(int), &sim->pcount_ex);
    sim->cpe_kernel = kernel;
    strcpy(sim->kname2, kname);
  }
  int size[1];
  size[0] = sim->pcount_ex;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  int group = global[0]/local[0];
  //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel, 4, sizeof(md_inter_params_t_x), &params);
  clSetKernelArg(kernel, 5, sizeof(nd_rect_t_x), &rect2);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x)*local[0], NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    sim->add_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  md_num_t_x mult = 1.0;
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->e_mem);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel2, 2, sizeof(int), &group);
  clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  sim->queue = queue;
  stats->queue = queue;
  //cl_event events[1];
  //clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, sizeof(md_num_t_x)*group, output, 0, NULL, &events[0]);
  //clWaitForEvents(1, events);
  //md_num_t_x res = 0;
  //for (i = 0; i < group; ++i)
    //res += output[i];
  //stats->es[stats->N-1] += res;
  //clReleaseKernel(kernel);
  clReleaseMemObject(out_mem);
  //clReleaseEvent(events[0]);
  //free(output);
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
        stats->es[stats->N-1] += 0.5*pe(sim->particles[i].x, sim->particles[j].x, L, sim->particles[i].m);
    }
#endif
}

void md_calc_temperature_x(md_simulation_t_x *sim, md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(sim, context);
  md_stats_to_context_x(stats, context);
  cl_kernel kernel = NULL;
  if (sim->ct_kernel != NULL)
    kernel = sim->ct_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_calc_temperature_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(int), &sim->N);
    sim->ct_kernel = kernel;
  }
  int size[1];
  size[0] = sim->N*MD_DIMENSION_X;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  int group = global[0]/local[0];
  //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel, 3, sizeof(md_num_t_x)*local[0], NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    sim->add_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  md_num_t_x mult = 1.0/sim->fc;
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->T_mem);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel2, 2, sizeof(int), &group);
  clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  sim->queue = queue;
  stats->queue = queue;
  //cl_event events[1];
  //clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, sizeof(md_num_t_x)*group, output, 0, NULL, &events[0]);
  //clWaitForEvents(1, events);
  //md_num_t_x res = 0;
  //int i;
  //for (i = 0; i < group; ++i)
    //res += output[i];
  //stats->Ts[stats->N-1] += res/sim->fc;
  //clReleaseKernel(kernel);
  clReleaseMemObject(out_mem);
  //clReleaseEvent(events[0]);
  //free(output);
#else
  int f = 0;
  int i, j;
  for (i = 0; i < sim->Nf; ++i)
    f += sim->fs[i];
  md_num_t_x sum = 0;
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j)
      sum += sim->particles[i].m*sim->particles[i].v[j]*sim->particles[i].v[j];
  stats->Ts[stats->N-1] += sum/f;
#endif
}

void md_calc_pair_correlation_x(md_simulation_t_x *sim, md_stats_t_x *stats, md_num_t_x rmax, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(sim, context);
  md_stats_to_context_x(stats, context);
  nd_rect_t_x rect2;
  int i;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rect2.L[i] = L[i];
  }
  cl_kernel kernel = NULL;
  if (sim->cpc_kernel != NULL)
    kernel = sim->cpc_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_calc_pair_correlation_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(int), &sim->N);
    sim->cpc_kernel = kernel;
  }
  int size[1];
  size[0] = sim->N;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  if (sim->pc_mem != NULL && sim->points != stats->points) {
    clReleaseMemObject(sim->pc_mem);
    sim->pc_mem = NULL;
  }
  cl_mem den_mem = NULL;
  if (sim->pc_mem != NULL)
    den_mem = sim->pc_mem;
  else {
    den_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*stats->points*sim->N, NULL, &status);
    sim->pc_mem = den_mem;
    sim->points = stats->points;
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &den_mem);
  clSetKernelArg(kernel, 3, sizeof(int), &stats->points);
  clSetKernelArg(kernel, 4, sizeof(int), &image);
  clSetKernelArg(kernel, 5, sizeof(md_num_t_x), &rmax);
  clSetKernelArg(kernel, 6, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel2 = NULL;
  if (sim->aden_kernel != NULL)
    kernel2 = sim->aden_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_density_kx", &status);
    sim->aden_kernel = kernel2;
  }
  size[0] = stats->points;
  md_get_work_size_x(kernel2, device, 1, size, global, local);
  //cl_mem fx_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(md_num_t_x)*stats->points, stats->fxs[stats->N-1], &status);
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &den_mem);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &stats->fx_mem);
  clSetKernelArg(kernel2, 2, sizeof(int), &sim->N);
  clSetKernelArg(kernel2, 3, sizeof(int), &stats->points);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  sim->queue = queue;
  stats->queue = queue;
  //cl_event events[1];
  //clEnqueueReadBuffer(queue, fx_mem, CL_FALSE, 0, sizeof(md_num_t_x)*stats->points, stats->fxs[stats->N-1], 0, NULL, &events[0]);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseKernel(kernel2);
  //clReleaseMemObject(den_mem);
  //clReleaseMemObject(fx_mem);
  //clReleaseEvent(events[0]);
#else
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
#endif
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