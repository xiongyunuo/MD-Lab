#include "md_pimd_x.h"
#include "md_util_x.h"
#include "md_simulation_x.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef MD_USE_OPENCL_X
cl_int md_pimd_to_context_x(md_pimd_t_x *pimd, cl_context context) {
  cl_int status;
  if (context == pimd->context && pimd->ENk_mem != NULL)
    return 0;
  else if (context != pimd->context && pimd->ENk_mem != NULL) {
    status = clReleaseMemObject(pimd->ENk_mem);
    status |= clReleaseMemObject(pimd->ENk2_mem);
    status |= clReleaseMemObject(pimd->VBN_mem);
    status |= clReleaseMemObject(pimd->pair_in_mem);
    status |= clReleaseMemObject(pimd->pair_ex_mem);
    status |= clReleaseMemObject(pimd->Eindices_mem);
    status |= md_pimd_clear_cache_x(pimd);
    pimd->ENk_mem = NULL;
    pimd->ENk2_mem = NULL;
    pimd->VBN_mem = NULL;
    pimd->pair_in_mem = NULL;
    pimd->pair_ex_mem = NULL;
    pimd->Eindices_mem = NULL;
    pimd->context = NULL;
    pimd->queue = NULL;
    if (status != CL_SUCCESS)
      return status;
  }
  cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  pimd->ENk_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*pimd->ENk_count, pimd->ENk[0], &status);
  if (status != CL_SUCCESS)
    return status;
  pimd->ENk2_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*pimd->ENk_count, pimd->ENk2[0], &status);
  if (status != CL_SUCCESS)
    return status;
  pimd->VBN_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), pimd->VBN, &status);
  if (status != CL_SUCCESS)
    return status;
  pimd->pair_in_mem = clCreateBuffer(context, flags, sizeof(md_index_pair_t_x)*pimd->pcount_in, pimd->pair_in, &status);
  if (status != CL_SUCCESS)
    return status;
  pimd->pair_ex_mem = clCreateBuffer(context, flags, sizeof(md_index_pair_t_x)*pimd->pcount_ex, pimd->pair_ex, &status);
  if (status != CL_SUCCESS)
    return status;
  pimd->Eindices_mem = clCreateBuffer(context, flags, sizeof(int)*pimd->N*pimd->N, pimd->Eindices, &status);
  if (status != CL_SUCCESS)
    return status;
  pimd->context = context;
  pimd->queue = NULL;
  return 0;
}

cl_event md_pimd_sync_queue_x(md_pimd_t_x *pimd, cl_command_queue queue, cl_int *err) {
  pimd->queue = queue;
  cl_event evt = NULL;
  *err = clEnqueueMarker(queue, &evt);
  if (*err != CL_SUCCESS)
    return NULL;
  return evt;
}

cl_int md_pimd_clear_cache_x(md_pimd_t_x *pimd) {
  cl_int status = 0;
  if (pimd->fillE_kernel != NULL) {
    status |= clReleaseKernel(pimd->fillE_kernel);
    pimd->fillE_kernel = NULL;
  }
  if (pimd->minE_kernel != NULL) {
    status |= clReleaseKernel(pimd->minE_kernel);
    pimd->minE_kernel = NULL;
  }
  if (pimd->min_kernel != NULL) {
    status |= clReleaseKernel(pimd->min_kernel);
    pimd->min_kernel = NULL;
  }
  if (pimd->minE_mem != NULL) {
    status |= clReleaseMemObject(pimd->minE_mem);
    pimd->minE_mem = NULL;
  }
  if (pimd->fillV_kernel != NULL) {
    status |= clReleaseKernel(pimd->fillV_kernel);
    pimd->fillV_kernel = NULL;
  }
  if (pimd->addV_kernel != NULL) {
    status |= clReleaseKernel(pimd->addV_kernel);
    pimd->addV_kernel = NULL;
  }
  if (pimd->fillFV_kernel != NULL) {
    status |= clReleaseKernel(pimd->fillFV_kernel);
    pimd->fillFV_kernel = NULL;
  }
  if (pimd->filleV_kernel != NULL) {
    status |= clReleaseKernel(pimd->filleV_kernel);
    pimd->filleV_kernel = NULL;
  }
  if (pimd->addeV_kernel != NULL) {
    status |= clReleaseKernel(pimd->addeV_kernel);
    pimd->addeV_kernel = NULL;
  }
  if (pimd->eVBN_mem != NULL) {
    status |= clReleaseMemObject(pimd->eVBN_mem);
    pimd->eVBN_mem = NULL;
  }
  if (pimd->addeVst_kernel != NULL) {
    status |= clReleaseKernel(pimd->addeVst_kernel);
    pimd->addeVst_kernel = NULL;
  }
  if (pimd->ctf_kernel != NULL) {
    status |= clReleaseKernel(pimd->ctf_kernel);
    pimd->ctf_kernel = NULL;
  }
  if (pimd->cte_kernel != NULL) {
    status |= clReleaseKernel(pimd->cte_kernel);
    pimd->cte_kernel = NULL;
  }
  if (pimd->cpf_kernel != NULL) {
    status |= clReleaseKernel(pimd->cpf_kernel);
    pimd->cpf_kernel = NULL;
  }
  if (pimd->upf_kernel != NULL) {
    status |= clReleaseKernel(pimd->upf_kernel);
    pimd->upf_kernel = NULL;
  }
  if (pimd->forces != NULL) {
    status |= clReleaseMemObject(pimd->forces);
    pimd->forces = NULL;
  }
  if (pimd->cpe_kernel != NULL) {
    status |= clReleaseKernel(pimd->cpe_kernel);
    pimd->cpe_kernel = NULL;
  }
  if (pimd->cdd_kernel != NULL) {
    status |= clReleaseKernel(pimd->cdd_kernel);
    pimd->cdd_kernel = NULL;
  }
  if (pimd->den_mem != NULL) {
    status |= clReleaseMemObject(pimd->den_mem);
    pimd->den_mem = NULL;
  }
  if (pimd->ffillE_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillE_kernel);
    pimd->ffillE_kernel = NULL;
  }
  if (pimd->ffillE2_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillE2_kernel);
    pimd->ffillE2_kernel = NULL;
  }
  if (pimd->Eint_mem != NULL) {
    status |= clReleaseMemObject(pimd->Eint_mem);
    pimd->Eint_mem = NULL;
  }
  if (pimd->fminE_kernel != NULL) {
    status |= clReleaseKernel(pimd->fminE_kernel);
    pimd->fminE_kernel = NULL;
  }
  if (pimd->V_mem != NULL) {
    status |= clReleaseMemObject(pimd->V_mem);
    pimd->V_mem = NULL;
  }
  if (pimd->fminE_mem != NULL) {
    status |= clReleaseMemObject(pimd->fminE_mem);
    pimd->fminE_mem = NULL;
  }
  if (pimd->ffillV_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillV_kernel);
    pimd->ffillV_kernel = NULL;
  }
  if (pimd->faddV_kernel != NULL) {
    status |= clReleaseKernel(pimd->faddV_kernel);
    pimd->faddV_kernel = NULL;
  }
  if (pimd->G_mem != NULL) {
    status |= clReleaseMemObject(pimd->G_mem);
    pimd->G_mem = NULL;
  }
  if (pimd->ffillG_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillG_kernel);
    pimd->ffillG_kernel = NULL;
  }
  if (pimd->ffillFV_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillFV_kernel);
    pimd->ffillFV_kernel = NULL;
  }
  if (pimd->ffillFV2_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillFV2_kernel);
    pimd->ffillFV2_kernel = NULL;
  }
  if (pimd->ffillFV3_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillFV3_kernel);
    pimd->ffillFV3_kernel = NULL;
  }
  if (pimd->fillE2_kernel != NULL) {
    status |= clReleaseKernel(pimd->fillE2_kernel);
    pimd->fillE2_kernel = NULL;
  }
  if (pimd->ffillE21_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillE21_kernel);
    pimd->ffillE21_kernel = NULL;
  }
  if (pimd->ffillE22_kernel != NULL) {
    status |= clReleaseKernel(pimd->ffillE22_kernel);
    pimd->ffillE22_kernel = NULL;
  }
  if (pimd->filleV2_kernel != NULL) {
    status |= clReleaseKernel(pimd->filleV2_kernel);
    pimd->filleV2_kernel = NULL;
  }
  if (pimd->addeVst2_kernel != NULL) {
    status |= clReleaseKernel(pimd->addeVst2_kernel);
    pimd->addeVst2_kernel = NULL;
  }
  if (pimd->cvire_kernel != NULL) {
    status |= clReleaseKernel(pimd->cvire_kernel);
    pimd->cvire_kernel = NULL;
  }
  if (pimd->cpf2_kernel != NULL) {
    status |= clReleaseKernel(pimd->cpf2_kernel);
    pimd->cpf2_kernel = NULL;
  }
  if (pimd->upf21_kernel != NULL) {
    status |= clReleaseKernel(pimd->upf21_kernel);
    pimd->upf21_kernel = NULL;
  }
  if (pimd->upf22_kernel != NULL) {
    status |= clReleaseKernel(pimd->upf22_kernel);
    pimd->upf22_kernel = NULL;
  }
  if (pimd->forces2 != NULL) {
    status |= clReleaseMemObject(pimd->forces2);
    pimd->forces2 = NULL;
  }
  if (pimd->cpe2_kernel != NULL) {
    status |= clReleaseKernel(pimd->cpe2_kernel);
    pimd->cpe2_kernel = NULL;
  }
  if (pimd->cITCF_kernel != NULL) {
    status |= clReleaseKernel(pimd->cITCF_kernel);
    pimd->cITCF_kernel = NULL;
  }
  return status;
}
#endif

#ifdef MD_USE_OPENCL_X
md_pimd_t_x *md_pimd_sync_host_x(md_pimd_t_x *pimd, int read_only) {
#else
md_pimd_t_x *md_pimd_sync_host_x(md_pimd_t_x *pimd, int MD_UNUSED_X(read_only)) {
#endif
#ifdef MD_USE_OPENCL_X
  if (pimd->ENk_mem == NULL)
    return pimd;
  cl_command_queue queue = pimd->queue;
  if (queue == NULL) {
    int i;
    for (i = 0; i < md_num_platforms_x; ++i)
      if (md_contexts_x[i] == pimd->context) {
        queue = md_command_queues_x[i][0];
        break;
      }
  }
  cl_int status;
  cl_event events[6];
  status = clEnqueueReadBuffer(queue, pimd->ENk_mem, CL_FALSE, 0, sizeof(md_num_t_x)*pimd->ENk_count, pimd->ENk[0], 0, NULL, &events[0]);
  status |= clEnqueueReadBuffer(queue, pimd->ENk2_mem, CL_FALSE, 0, sizeof(md_num_t_x)*pimd->ENk_count, pimd->ENk2[0], 0, NULL, &events[1]);
  status |= clEnqueueReadBuffer(queue, pimd->VBN_mem, CL_FALSE, 0, sizeof(md_num_t_x)*(pimd->N+1), pimd->VBN, 0, NULL, &events[2]);
  status |= clEnqueueReadBuffer(queue, pimd->pair_in_mem, CL_FALSE, 0, sizeof(md_index_pair_t_x)*pimd->pcount_in, pimd->pair_in, 0, NULL, &events[3]);
  status |= clEnqueueReadBuffer(queue, pimd->pair_ex_mem, CL_FALSE, 0, sizeof(md_index_pair_t_x)*pimd->pcount_ex, pimd->pair_ex, 0, NULL, &events[4]);
  status |= clEnqueueReadBuffer(queue, pimd->Eindices_mem, CL_FALSE, 0, sizeof(int)*pimd->N*pimd->N, pimd->Eindices, 0, NULL, &events[5]);
  if (status != CL_SUCCESS)
    return NULL;
  status = clWaitForEvents(6, events);
  status |= clReleaseEvent(events[0]);
  status |= clReleaseEvent(events[1]);
  status |= clReleaseEvent(events[2]);
  status |= clReleaseEvent(events[3]);
  status |= clReleaseEvent(events[4]);
  status |= clReleaseEvent(events[5]);
  if (status != CL_SUCCESS)
    return NULL;
  if (!read_only) {
    status = clReleaseMemObject(pimd->ENk_mem);
    status |= clReleaseMemObject(pimd->ENk2_mem);
    status |= clReleaseMemObject(pimd->VBN_mem);
    status |= clReleaseMemObject(pimd->pair_in_mem);
    status |= clReleaseMemObject(pimd->pair_ex_mem);
    status |= clReleaseMemObject(pimd->Eindices_mem);
    status |= md_pimd_clear_cache_x(pimd);
    pimd->ENk_mem = NULL;
    pimd->ENk2_mem = NULL;
    pimd->VBN_mem = NULL;
    pimd->pair_in_mem = NULL;
    pimd->pair_ex_mem = NULL;
    pimd->Eindices_mem = NULL;
    pimd->context = NULL;
    pimd->queue = NULL;
    if (status != CL_SUCCESS)
      return NULL;
  }
#endif
  return pimd;
}

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
  pimd->minE = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(N+1));
  if (pimd->VBN == NULL || pimd->minE == NULL) {
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
  int j;
  pimd->pcount_in = pimd->N*(pimd->N+1)/2;
  pimd->pcount_ex = pimd->N*(pimd->N-1)/2;
  pimd->pair_in = (md_index_pair_t_x *)malloc(sizeof(md_index_pair_t_x)*pimd->pcount_in);
  pimd->pair_ex = (md_index_pair_t_x *)malloc(sizeof(md_index_pair_t_x)*pimd->pcount_ex);
  if (pimd->pair_in == NULL || pimd->pair_ex == NULL) {
    free(pimd);
    return NULL;
  }
  pimd->Eindices = (int *)malloc(sizeof(int)*pimd->N*pimd->N);
  if (pimd->Eindices == NULL) {
    free(pimd);
    return NULL;
  }
  count = 0;
  for (i = 0; i < pimd->N; ++i)
    for (j = 0; j <= i; ++j) {
      pimd->pair_in[count].i = i;
      pimd->pair_in[count].j = j;
      pimd->Eindices[i*pimd->N+j] = count;
      ++count;
    }
  count = 0;
  for (i = 0; i < pimd->N; ++i)
    for (j = 0; j < i; ++j) {
      pimd->pair_ex[count].i = i;
      pimd->pair_ex[count].j = j;
      ++count;
    }
#ifdef MD_USE_OPENCL_X
  pimd->ENk_mem = NULL;
  pimd->ENk2_mem = NULL;
  pimd->VBN_mem = NULL;
  pimd->pair_in_mem = NULL;
  pimd->pair_ex_mem = NULL;
  pimd->Eindices_mem = NULL;
  pimd->context = NULL;
  pimd->queue = NULL;
  pimd->fillE_kernel = NULL;
  pimd->minE_kernel = NULL;
  pimd->min_kernel = NULL;
  pimd->minE_mem = NULL;
  pimd->fillV_kernel = NULL;
  pimd->addV_kernel = NULL;
  pimd->fillFV_kernel = NULL;
  pimd->filleV_kernel = NULL;
  pimd->addeV_kernel = NULL;
  pimd->eVBN_mem = NULL;
  pimd->addeVst_kernel = NULL;
  pimd->ctf_kernel = NULL;
  pimd->tfkname[0] = '\0';
  pimd->cte_kernel = NULL;
  pimd->tekname[0] = '\0';
  pimd->cpf_kernel = NULL;
  pimd->upf_kernel = NULL;
  pimd->pfkname[0] = '\0';
  pimd->forces = NULL;
  pimd->cpe_kernel = NULL;
  pimd->pekname[0] = '\0';
  pimd->cdd_kernel = NULL;
  pimd->den_mem = NULL;
  pimd->points = 0;
  pimd->ffillE_kernel = NULL;
  pimd->ffillE2_kernel = NULL;
  pimd->Eint_mem = NULL;
  pimd->fminE_kernel = NULL;
  pimd->V_mem = NULL;
  pimd->fminE_mem = NULL;
  pimd->ffillV_kernel = NULL;
  pimd->faddV_kernel = NULL;
  pimd->G_mem = NULL;
  pimd->ffillG_kernel = NULL;
  pimd->ffillFV_kernel = NULL;
  pimd->ffillFV2_kernel = NULL;
  pimd->ffillFV3_kernel = NULL;
  pimd->fillE2_kernel = NULL;
  pimd->ffillE21_kernel = NULL;
  pimd->ffillE22_kernel = NULL;
  pimd->filleV2_kernel = NULL;
  pimd->addeVst2_kernel = NULL;
  pimd->cvire_kernel = NULL;
  pimd->cpf2_kernel = NULL;
  pimd->upf21_kernel = NULL;
  pimd->upf22_kernel = NULL;
  pimd->forces2 = NULL;
  pimd->N2 = 0;
  pimd->pf2kname[0] = '\0';
  pimd->cpe2_kernel = NULL;
  pimd->pe2kname[0] = '\0';
  pimd->cITCF_kernel = NULL;
#endif
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
  md_simulation_sync_host_x(pimd->sim, 0);
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
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_pimd_to_context_x(pimd, context);
  md_simulation_t_x *sim = pimd->sim;
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
  if (pimd->fillE_kernel != NULL)
    kernel = pimd->fillE_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fill_ENk_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->pair_in_mem);
    clSetKernelArg(kernel, 3, sizeof(int), &pimd->pcount_in);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    pimd->fillE_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->pcount_in;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clSetKernelArg(kernel, 4, sizeof(int), &image);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
  //cl_event events[1];
  //events[0] = md_pimd_sync_queue_x(pimd, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseEvent(events[0]);
#else
  int l, j;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= l; ++j)
      pimd->ENk[l-1][j-1] = md_pimd_ENk_x(pimd, l, j, image);
#endif
}

int md_pimd_fprint_ENk_x(FILE *out, md_pimd_t_x *pimd) {
  md_pimd_sync_host_x(pimd, 1);
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
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_pimd_to_context_x(pimd, context);
  /*cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  clReleaseMemObject(pimd->VBN_mem);
  pimd->VBN_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), pimd->VBN, &status);*/
  cl_kernel kernel = NULL;
  if (pimd->minE_kernel != NULL)
    kernel = pimd->minE_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_xminE_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->VBN_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    pimd->minE_kernel = kernel;
  }
  int size[1];
  size[0] = N2;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  int group = global[0]/local[0];
  //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel, 5, sizeof(int), &N2);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  clSetKernelArg(kernel, 8, sizeof(md_num_t_x)*local[0], NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_mem minE_mem = NULL;
  if (pimd->minE_mem != NULL)
    minE_mem = pimd->minE_mem;
  else {
    minE_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    pimd->minE_mem = minE_mem;
  }
  cl_kernel kernel2 = NULL;
  if (pimd->min_kernel != NULL)
    kernel2 = pimd->min_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_min_kx", &status);
    pimd->min_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &minE_mem);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel2, 2, sizeof(int), &group);
  clSetKernelArg(kernel2, 3, sizeof(int), &N2);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
  //cl_event events[1];
  //clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, sizeof(md_num_t_x)*group, output, 0, NULL, &events[0]);
  //clWaitForEvents(1, events);
  //md_num_t_x res = 1e10;
  //int i;
  /*size_t max_gsize;
  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_gsize, NULL);
  printf("size %zu\n", max_gsize);*/
  //for (i = 0; i < group; ++i)
    //if (output[i] < res)
      //res = output[i];
  //clReleaseKernel(kernel);
  clReleaseMemObject(out_mem);
  //clReleaseEvent(events[0]);
  //free(output);
  //return res;
  return 0;
#else
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
#endif
}

#ifdef MD_USE_OPENCL_X
static md_num_t_x md_VBN_0_x = 0.0;
#endif

void md_pimd_fill_VB_x(md_pimd_t_x *pimd) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_pimd_to_context_x(pimd, context);
  //cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  int N2;
  //pimd->VBN[0] = 0.0;
  //clReleaseMemObject(pimd->VBN_mem);
  //pimd->VBN_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), pimd->VBN, &status);
  clEnqueueWriteBuffer(queue, pimd->VBN_mem, CL_FALSE, 0, sizeof(md_num_t_x), &md_VBN_0_x, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel = NULL;
  if (pimd->fillV_kernel != NULL)
    kernel = pimd->fillV_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fill_VB_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->VBN_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    pimd->fillV_kernel = kernel;
  }
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  cl_kernel kernel2 = NULL;
  if (pimd->addV_kernel != NULL)
    kernel2 = pimd->addV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_add_VBN_kx", &status);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), &pimd->VBN_mem);
    pimd->addV_kernel = kernel2;
  }
  clSetKernelArg(kernel2, 5, sizeof(md_num_t_x), &pimd->beta);
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    //md_num_t_x tmp = md_pimd_xminE_x(pimd, N2);
    //pimd->minE[N2] = tmp;
    md_pimd_xminE_x(pimd, N2);
    int size[1];
    size[0] = N2;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    int group = global[0]/local[0];
    //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel, 5, sizeof(int), &N2);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &pimd->minE_mem);
    clSetKernelArg(kernel, 9, sizeof(md_num_t_x)*local[0], NULL);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel2, 2, sizeof(cl_mem), &pimd->minE_mem);
    clSetKernelArg(kernel2, 3, sizeof(int), &group);
    clSetKernelArg(kernel2, 4, sizeof(int), &N2);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    //cl_event events[1];
    //clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, sizeof(md_num_t_x)*group, output, 0, NULL, &events[0]);
    //clWaitForEvents(1, events);
    //md_num_t_x res = 0;
    //int i;
    //for (i = 0; i < group; ++i)
      //res += output[i];
    //pimd->VBN[N2] = (tmp-log(res)+log(N2))/pimd->beta;
    //clReleaseMemObject(pimd->VBN_mem);
    //pimd->VBN_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), pimd->VBN, &status);
    //clReleaseKernel(kernel);
    clReleaseMemObject(out_mem);
    //clReleaseEvent(events[0]);
    //free(output);
  }
  pimd->queue = queue;
#else
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
#endif
}

int md_pimd_fprint_VBN_x(FILE *out, md_pimd_t_x *pimd) {
  md_pimd_sync_host_x(pimd, 1);
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
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_pimd_to_context_x(pimd, context);
  md_simulation_t_x *sim = pimd->sim;
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
  if (pimd->fillFV_kernel != NULL)
    kernel = pimd->fillFV_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fill_force_VB_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->VBN_mem);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 6, sizeof(int), &pimd->P);
    pimd->fillFV_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N*pimd->P;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  //cl_mem minE_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(md_num_t_x)*(pimd->N+1), pimd->minE, &status);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &pimd->minE_mem);
  clSetKernelArg(kernel, 7, sizeof(int), &image);
  clSetKernelArg(kernel, 8, sizeof(md_num_t_x), &pimd->vi);
  clSetKernelArg(kernel, 9, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel, 10, sizeof(md_num_t_x), &pimd->beta);
  clSetKernelArg(kernel, 11, sizeof(nd_rect_t_x), &rect2);
  clSetKernelArg(kernel, 12, sizeof(md_num_t_x)*MD_DIMENSION_X*(pimd->N+1)*local[0], NULL);
  clSetKernelArg(kernel, 13, sizeof(md_num_t_x)*MD_DIMENSION_X*pimd->N*local[0], NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
  //cl_event events[1];
  //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseMemObject(minE_mem);
  //clReleaseEvent(events[0]);
#else
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
#endif
}

void md_pimd_calc_VBN_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_pimd_to_context_x(pimd, context);
  //cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  cl_mem res_mem = NULL;
  if (pimd->eVBN_mem != NULL)
    res_mem = pimd->eVBN_mem;
  else {
    res_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    pimd->eVBN_mem = res_mem;
  }
  clEnqueueWriteBuffer(queue, pimd->eVBN_mem, CL_FALSE, 0, sizeof(md_num_t_x), &md_VBN_0_x, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel = NULL;
  if (pimd->filleV_kernel != NULL)
    kernel = pimd->filleV_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_VBN_energy_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->VBN_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->N);
    pimd->filleV_kernel = kernel;
  }
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &res_mem);
  clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->vi);
  clSetKernelArg(kernel, 8, sizeof(md_num_t_x), &pimd->beta);
  clSetKernelArg(kernel, 9, sizeof(cl_mem), &pimd->minE_mem);
  cl_kernel kernel2 = NULL;
  if (pimd->addeV_kernel != NULL)
    kernel2 = pimd->addeV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_add_eVBN_kx", &status);
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &pimd->VBN_mem);
    pimd->addeV_kernel = kernel2;
  }
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &pimd->eVBN_mem);
  clSetKernelArg(kernel2, 3, sizeof(cl_mem), &pimd->minE_mem);
  clSetKernelArg(kernel2, 6, sizeof(md_num_t_x), &pimd->beta);
  int N2;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    //md_num_t_x tmp = pimd->minE[N2];
    int size[1];
    size[0] = N2;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    int group = global[0]/local[0];
    //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel, 6, sizeof(int), &N2);
    clSetKernelArg(kernel, 10, sizeof(md_num_t_x)*local[0], NULL);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    clSetKernelArg(kernel2, 2, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel2, 4, sizeof(int), &group);
    clSetKernelArg(kernel2, 5, sizeof(int), &N2);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    //cl_event events[1];
    //clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, sizeof(md_num_t_x)*group, output, 0, NULL, &events[0]);
    //clWaitForEvents(1, events);
    //md_num_t_x sum = 0;
    //int i;
    //for (i = 0; i < group; ++i)
      //sum += output[i];
    //res[N2] = sum/exp(-pimd->beta*pimd->VBN[N2]+log((md_num_t_x)N2)+tmp);
    //clReleaseMemObject(res_mem);
    //res_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), res, &status);
    //clReleaseKernel(kernel);
    clReleaseMemObject(out_mem);
    //clReleaseEvent(events[0]);
    //free(output);
  }
  //clReleaseMemObject(res_mem);
  md_stats_to_context_x(stats, context);
  cl_kernel kernel3 = NULL;
  if (pimd->addeVst_kernel != NULL)
    kernel3 = pimd->addeVst_kernel;
  else {
    kernel3 = clCreateKernel(md_programs_x[plat], "md_pimd_add_eVBN_stats_kx", &status);
    clSetKernelArg(kernel3, 2, sizeof(int), &pimd->N);
    clSetKernelArg(kernel3, 3, sizeof(int), &pimd->P);
    pimd->addeVst_kernel = kernel3;
  }
  clSetKernelArg(kernel3, 0, sizeof(cl_mem), &stats->e_mem);
  clSetKernelArg(kernel3, 1, sizeof(cl_mem), &pimd->eVBN_mem);
  clSetKernelArg(kernel3, 4, sizeof(md_num_t_x), &pimd->beta);
  size_t local[1], global[1];
  local[0] = 1;
  global[0] = 1;
  clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
  stats->queue = queue;
#else
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
#endif
}

void md_pimd_calc_trap_force_x(md_pimd_t_x *pimd, md_trap_force_t_x tf) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_simulation_t_x *sim = pimd->sim;
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
  md_get_trap_force_info_x(tf, "md_pimd_calc_trap_force_", "_kx", kname, &params);
  if (pimd->ctf_kernel != NULL && strcmp(pimd->tfkname, kname)) {
    clReleaseKernel(pimd->ctf_kernel);
    pimd->ctf_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (pimd->ctf_kernel != NULL)
    kernel = pimd->ctf_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 2, sizeof(int), &pimd->P);
    pimd->ctf_kernel = kernel;
    strcpy(pimd->tfkname, kname);
  }
  int size[1];
  size[0] = pimd->N*pimd->P;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clSetKernelArg(kernel, 3, sizeof(md_inter_params_t_x), &params);
  clSetKernelArg(kernel, 4, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
  sim->queue = queue;
  //cl_event events[1];
  //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseEvent(events[0]);
#else
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
#endif
}

void md_pimd_calc_trap_energy_x(md_pimd_t_x *pimd, md_trap_energy_t_x te, md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_stats_to_context_x(stats, context);
  md_simulation_t_x *sim = pimd->sim;
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
  md_get_trap_energy_info_x(te, "md_pimd_calc_trap_energy_", "_kx", kname, &params);
  if (pimd->cte_kernel != NULL && strcmp(pimd->tekname, kname)) {
    clReleaseKernel(pimd->cte_kernel);
    pimd->cte_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (pimd->cte_kernel != NULL)
    kernel = pimd->cte_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    pimd->cte_kernel = kernel;
    strcpy(pimd->tekname, kname);
  }
  int size[1];
  size[0] = pimd->N*pimd->P;
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
  pimd->queue = queue;
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
#endif
}

void md_pimd_calc_pair_force_x(md_pimd_t_x *pimd, md_pair_force_t_x pf) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_t_x *sim = pimd->sim;
  md_pimd_to_context_x(pimd, context);
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
  md_get_pair_force_info_x(pf, "md_pimd_fill_pair_force_", "_kx", kname, &params);
  if (pimd->cpf_kernel != NULL && strcmp(pimd->pfkname, kname)) {
    clReleaseKernel(pimd->cpf_kernel);
    pimd->cpf_kernel = NULL;
  }
  int j;
  cl_mem forces = NULL;
  if (pimd->forces != NULL)
    forces = pimd->forces;
  else {
    forces = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*pimd->N*pimd->N*MD_DIMENSION_X, NULL, NULL);
    pimd->forces = forces;
  }
  cl_kernel kernel = NULL;
  if (pimd->cpf_kernel != NULL)
    kernel = pimd->cpf_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->pair_ex_mem);
    clSetKernelArg(kernel, 3, sizeof(int), &pimd->pcount_ex);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    pimd->cpf_kernel = kernel;
    strcpy(pimd->pfkname, kname);
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &forces);
  clSetKernelArg(kernel, 7, sizeof(md_inter_params_t_x), &params);
  clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect2);
  cl_kernel kernel2 = NULL;
  if (pimd->upf_kernel != NULL)
    kernel2 = pimd->upf_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_update_pair_force_kx", &status);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
    clSetKernelArg(kernel2, 3, sizeof(int), &pimd->P);
    pimd->upf_kernel = kernel2;
  }
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &forces);
  for (j = 1; j <= pimd->P; ++j) {
    int size[1];
    size[0] = pimd->pcount_ex;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    clSetKernelArg(kernel, 6, sizeof(int), &j);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = pimd->N;
    md_get_work_size_x(kernel2, device, 1, size, global, local);
    clSetKernelArg(kernel2, 4, sizeof(int), &j);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
  }
  pimd->queue = queue;
  sim->queue = queue;
  //cl_event events[1];
  //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseKernel(kernel2);
  //clReleaseMemObject(forces);
  //clReleaseEvent(events[0]);
#else
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
#endif
}

void md_pimd_calc_pair_energy_x(md_pimd_t_x *pimd, md_pair_energy_t_x pe, md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_t_x *sim = pimd->sim;
  md_pimd_to_context_x(pimd, context);
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
  md_get_pair_energy_info_x(pe, "md_pimd_calc_pair_energy_", "_kx", kname, &params);
  if (pimd->cpe_kernel != NULL && strcmp(pimd->pekname, kname)) {
    clReleaseKernel(pimd->cpe_kernel);
    pimd->cpe_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (pimd->cpe_kernel != NULL)
    kernel = pimd->cpe_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->pair_ex_mem);
    clSetKernelArg(kernel, 3, sizeof(int), &pimd->pcount_ex);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->P);
    pimd->cpe_kernel = kernel;
    strcpy(pimd->pekname, kname);
  }
  clSetKernelArg(kernel, 6, sizeof(md_inter_params_t_x), &params);
  clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  cl_kernel kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    sim->add_kernel = kernel2;
  }
  md_num_t_x mult = 1.0;
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->e_mem);
  clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  int j;
  for (j = 1; j <= pimd->P; ++j) {
    int size[1];
    size[0] = pimd->pcount_ex;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    int group = global[0]/local[0];
    //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel, 5, sizeof(int), &j);
    clSetKernelArg(kernel, 8, sizeof(md_num_t_x)*local[0], NULL);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel2, 2, sizeof(int), &group);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
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
  }
  pimd->queue = queue;
  sim->queue = queue;
  stats->queue = queue;
#else
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
#endif
}

void md_pimd_calc_density_distribution_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmax, int image, md_num_t_x *center) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_to_context_x(sim, context);
  md_stats_to_context_x(stats, context);
  nd_rect_t_x rect2, rect3;
  int i;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      rect2.L[i] = L[i];
      rect3.L[i] = center[i];
    }
  }
  cl_kernel kernel = NULL;
  if (pimd->cdd_kernel != NULL)
    kernel = pimd->cdd_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_density_distribution_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    pimd->cdd_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  if (pimd->den_mem != NULL && pimd->points != stats->points) {
    clReleaseMemObject(pimd->den_mem);
    pimd->den_mem = NULL;
  }
  cl_mem den_mem = NULL;
  if (pimd->den_mem != NULL)
    den_mem = pimd->den_mem;
  else {
    den_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*stats->points*pimd->N, NULL, &status);
    pimd->den_mem = den_mem;
    pimd->points = stats->points;
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &den_mem);
  clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  clSetKernelArg(kernel, 5, sizeof(int), &image);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &rmax);
  clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect3);
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
  clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
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
#endif
}

void md_pimd_fast_fill_ENk_x(md_pimd_t_x *pimd, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_pimd_to_context_x(pimd, context);
  md_simulation_t_x *sim = pimd->sim;
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
  if (pimd->ffillE_kernel != NULL)
    kernel = pimd->ffillE_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_ENk_1_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 6, sizeof(int), &pimd->P);
    pimd->ffillE_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  cl_mem Eint_mem = NULL;
  if (pimd->Eint_mem != NULL)
    Eint_mem = pimd->Eint_mem;
  else {
    Eint_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*pimd->N, NULL, &status);
    pimd->Eint_mem = Eint_mem;
  }
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &Eint_mem);
  clSetKernelArg(kernel, 5, sizeof(int), &image);
  clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  int j;
  cl_kernel kernel2 = NULL;
  if (pimd->ffillE2_kernel != NULL)
    kernel2 = pimd->ffillE2_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_ENk_2_kx", &status);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel2, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel2, 5, sizeof(int), &pimd->N);
    clSetKernelArg(kernel2, 7, sizeof(int), &pimd->P);
    pimd->ffillE2_kernel = kernel2;
  }
  clSetKernelArg(kernel2, 3, sizeof(cl_mem), &Eint_mem);
  clSetKernelArg(kernel2, 6, sizeof(int), &image);
  clSetKernelArg(kernel2, 8, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel2, 9, sizeof(nd_rect_t_x), &rect2);
  for (j = 2; j <= pimd->N; ++j) {
    size[0] = pimd->N-j+1;
    md_get_work_size_x(kernel2, device, 1, size, global, local);
    clSetKernelArg(kernel2, 4, sizeof(int), &j);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
  }
  pimd->queue = queue;
  sim->queue = queue;
  //cl_event events[1];
  //events[0] = md_pimd_sync_queue_x(pimd, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseKernel(kernel2);
  //clReleaseMemObject(Eint_mem);
  //clReleaseEvent(events[0]);
#else
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
#endif
}

#ifdef MD_USE_OPENCL_X
md_num_t_x md_pimd_fast_xminE_x(md_pimd_t_x *pimd, int u, md_num_t_x *MD_UNUSED_X(V)) {
#else
md_num_t_x md_pimd_fast_xminE_x(md_pimd_t_x *pimd, int u, md_num_t_x *V) {
#endif
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_pimd_to_context_x(pimd, context);
  //cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  //cl_mem V_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), V, &status);
  cl_kernel kernel = NULL;
  if (pimd->fminE_kernel != NULL)
    kernel = pimd->fminE_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fast_xminE_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    pimd->fminE_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N-u+1;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  int group = global[0]/local[0];
  //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->V_mem);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel, 5, sizeof(int), &u);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  clSetKernelArg(kernel, 8, sizeof(md_num_t_x)*local[0], NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_mem minE_mem = NULL;
  if (pimd->fminE_mem != NULL)
    minE_mem = pimd->fminE_mem;
  else {
    minE_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    pimd->fminE_mem = minE_mem;
  }
  cl_kernel kernel2 = NULL;
  if (pimd->min_kernel != NULL)
    kernel2 = pimd->min_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_min_kx", &status);
    pimd->min_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &minE_mem);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel2, 2, sizeof(int), &group);
  clSetKernelArg(kernel2, 3, sizeof(int), &u);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
  //cl_event events[1];
  //clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, sizeof(md_num_t_x)*group, output, 0, NULL, &events[0]);
  //clWaitForEvents(1, events);
  //md_num_t_x res = 1e10;
  //int i;
  //for (i = 0; i < group; ++i)
    //if (output[i] < res)
      //res = output[i];
  //clReleaseKernel(kernel);
  clReleaseMemObject(out_mem);
  //clReleaseMemObject(V_mem);
  //clReleaseEvent(events[0]);
  //free(output);
  //return res;
  return 0;
#else
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
#endif
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
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_pimd_to_context_x(pimd, context);
  md_simulation_t_x *sim = pimd->sim;
  nd_rect_t_x rect2;
  int i;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rect2.L[i] = L[i];
  }
  //md_num_t_x *tmpV = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  //tmpV[pimd->N] = 0;
  //cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  //cl_mem tmpV_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), tmpV, &status);
  cl_mem tmpV_mem = NULL;
  if (pimd->V_mem != NULL)
    tmpV_mem = pimd->V_mem;
  else {
    tmpV_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    pimd->V_mem = tmpV_mem;
  }
  clEnqueueWriteBuffer(queue, pimd->V_mem, CL_FALSE, pimd->N*sizeof(md_num_t_x), sizeof(md_num_t_x), &md_VBN_0_x, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel = NULL;
  if (pimd->ffillV_kernel != NULL)
    kernel = pimd->ffillV_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_VB_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    pimd->ffillV_kernel = kernel;
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmpV_mem);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  cl_kernel kernel2 = NULL;
  if (pimd->faddV_kernel != NULL)
    kernel2 = pimd->faddV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_add_VBN_kx", &status);
    pimd->faddV_kernel = kernel2;
  }
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &tmpV_mem);
  clSetKernelArg(kernel2, 5, sizeof(md_num_t_x), &pimd->beta);
  int u;
  for (u = pimd->N; u >= 1; --u) {
    //md_num_t_x tmp = md_pimd_fast_xminE_x(pimd, u, tmpV);
    md_pimd_fast_xminE_x(pimd, u, NULL);
    int size[1];
    size[0] = pimd->N-u+1;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    int group = global[0]/local[0];
    //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel, 5, sizeof(int), &u);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &pimd->fminE_mem);
    clSetKernelArg(kernel, 9, sizeof(md_num_t_x)*local[0], NULL);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel2, 2, sizeof(cl_mem), &pimd->fminE_mem);
    clSetKernelArg(kernel2, 3, sizeof(int), &group);
    clSetKernelArg(kernel2, 4, sizeof(int), &u);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    //cl_event events[1];
    //clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, sizeof(md_num_t_x)*group, output, 0, NULL, &events[0]);
    //clWaitForEvents(1, events);
    //md_num_t_x sum = 0;
    //int i;
    //for (i = 0; i < group; ++i)
      //sum += output[i];
    //tmpV[u-1] = (tmp-log(sum))/pimd->beta;
    //clReleaseMemObject(tmpV_mem);
    //tmpV_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), tmpV, &status);
    //clReleaseKernel(kernel);
    clReleaseMemObject(out_mem);
    //clReleaseEvent(events[0]);
    //free(output);
  }
  cl_mem G_mem = NULL;
  if (pimd->G_mem != NULL)
    G_mem = pimd->G_mem;
  else {
    G_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*pimd->N*pimd->N, NULL, &status);
    pimd->G_mem = G_mem;
  }
  if (pimd->ffillG_kernel != NULL)
    kernel = pimd->ffillG_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_G_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->VBN_mem);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->N);
    pimd->ffillG_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N*pimd->N;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &tmpV_mem);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &G_mem);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  if (pimd->ffillFV_kernel != NULL)
    kernel2 = pimd->ffillFV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_force_VB_1_kx", &status);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel2, 1, sizeof(int), &pimd->N);
    clSetKernelArg(kernel2, 2, sizeof(int), &pimd->P);
    pimd->ffillFV_kernel = kernel2;
  }
  size[0] = pimd->N*(pimd->P-2);
  md_get_work_size_x(kernel2, device, 1, size, global, local);
  clSetKernelArg(kernel2, 3, sizeof(int), &image);
  clSetKernelArg(kernel2, 4, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel2, 5, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel3 = NULL;
  if (pimd->ffillFV2_kernel != NULL)
    kernel3 = pimd->ffillFV2_kernel;
  else {
    kernel3 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_force_VB_2_kx", &status);
    clSetKernelArg(kernel3, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel3, 2, sizeof(int), &pimd->N);
    clSetKernelArg(kernel3, 3, sizeof(int), &pimd->P);
    pimd->ffillFV2_kernel = kernel3;
  }
  size[0] = pimd->N;
  md_get_work_size_x(kernel3, device, 1, size, global, local);
  clSetKernelArg(kernel3, 1, sizeof(cl_mem), &G_mem);
  clSetKernelArg(kernel3, 4, sizeof(int), &image);
  clSetKernelArg(kernel3, 5, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel3, 6, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel4 = NULL;
  if (pimd->ffillFV3_kernel != NULL)
    kernel4 = pimd->ffillFV3_kernel;
  else {
    kernel4 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_force_VB_3_kx", &status);
    clSetKernelArg(kernel4, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel4, 2, sizeof(int), &pimd->N);
    clSetKernelArg(kernel4, 3, sizeof(int), &pimd->P);
    pimd->ffillFV3_kernel = kernel4;
  }
  size[0] = pimd->N;
  md_get_work_size_x(kernel4, device, 1, size, global, local);
  clSetKernelArg(kernel4, 1, sizeof(cl_mem), &G_mem);
  clSetKernelArg(kernel4, 4, sizeof(int), &image);
  clSetKernelArg(kernel4, 5, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel4, 6, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel4, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
  sim->queue = queue;
  //cl_event events[1];
  //events[0] = md_simulation_sync_queue_x(sim, queue, &status);
  //clWaitForEvents(1, events);
  //clReleaseKernel(kernel);
  //clReleaseKernel(kernel2);
  //clReleaseKernel(kernel3);
  //clReleaseKernel(kernel4);
  //clReleaseMemObject(tmpV_mem);
  //clReleaseMemObject(G_mem);
  //clReleaseEvent(events[0]);
#else
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
#endif
}

void md_pimd_calc_ITCF_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmin, md_num_t_x rmax, int pi) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_to_context_x(sim, context);
  md_stats_to_context_x(stats, context);
  cl_kernel kernel = NULL;
  if (pimd->cITCF_kernel != NULL)
    kernel = pimd->cITCF_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_ITCF_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    pimd->cITCF_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  if (pimd->den_mem != NULL && pimd->points != stats->points) {
    clReleaseMemObject(pimd->den_mem);
    pimd->den_mem = NULL;
  }
  cl_mem den_mem = NULL;
  if (pimd->den_mem != NULL)
    den_mem = pimd->den_mem;
  else {
    den_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*stats->points*pimd->N, NULL, &status);
    pimd->den_mem = den_mem;
    pimd->points = stats->points;
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &den_mem);
  clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  clSetKernelArg(kernel, 5, sizeof(int), &pi);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &rmin);
  clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &rmax);
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
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &den_mem);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &stats->fx_mem);
  clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
  clSetKernelArg(kernel2, 3, sizeof(int), &stats->points);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  sim->queue = queue;
  stats->queue = queue;
#else
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
#endif
}

int md_pimd_calc_pair_force_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_pair_force_t_x pf) {
  if (pimd->P != pimd2->P)
    return -1;
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_pimd_to_context_x(pimd, context);
  md_simulation_to_context_x(sim, context);
  md_simulation_to_context_x(sim2, context);
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
  md_get_pair_force_info_x(pf, "md_pimd_fill_pair_force_2_", "_kx", kname, &params);
  if (pimd->cpf2_kernel != NULL && strcmp(pimd->pf2kname, kname)) {
    clReleaseKernel(pimd->cpf2_kernel);
    pimd->cpf2_kernel = NULL;
  }
  int j;
  if (pimd->forces2 != NULL && pimd->N2 != pimd2->N) {
    clReleaseMemObject(pimd->forces2);
    pimd->forces2 = NULL;
  }
  cl_mem forces = NULL;
  if (pimd->forces2 != NULL)
    forces = pimd->forces2;
  else {
    forces = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*pimd->N*pimd2->N*MD_DIMENSION_X, NULL, NULL);
    pimd->forces2 = forces;
    pimd->N2 = pimd2->N;
  }
  cl_kernel kernel = NULL;
  if (pimd->cpf2_kernel != NULL)
    kernel = pimd->cpf2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    pimd->cpf2_kernel = kernel;
    strcpy(pimd->pf2kname, kname);
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &sim2->particles_mem);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &forces);
  clSetKernelArg(kernel, 3, sizeof(int), &pimd2->N);
  clSetKernelArg(kernel, 7, sizeof(md_inter_params_t_x), &params);
  clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect2);
  cl_kernel kernel2 = NULL;
  if (pimd->upf21_kernel != NULL)
    kernel2 = pimd->upf21_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_update_pair_force_2_1_kx", &status);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel2, 3, sizeof(int), &pimd->N);
    clSetKernelArg(kernel2, 4, sizeof(int), &pimd->P);
    pimd->upf21_kernel = kernel2;
  }
  clSetKernelArg(kernel2, 2, sizeof(int), &pimd2->N);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &forces);
  cl_kernel kernel3 = NULL;
  if (pimd->upf22_kernel != NULL)
    kernel3 = pimd->upf22_kernel;
  else {
    kernel3 = clCreateKernel(md_programs_x[plat], "md_pimd_update_pair_force_2_2_kx", &status);
    clSetKernelArg(kernel3, 3, sizeof(int), &pimd->N);
    clSetKernelArg(kernel3, 4, sizeof(int), &pimd->P);
    pimd->upf22_kernel = kernel3;
  }
  clSetKernelArg(kernel3, 0, sizeof(cl_mem), &sim2->particles_mem);
  clSetKernelArg(kernel3, 2, sizeof(int), &pimd2->N);
  clSetKernelArg(kernel3, 1, sizeof(cl_mem), &forces);
  for (j = 1; j <= pimd->P; ++j) {
    int size[1];
    size[0] = pimd->N*pimd2->N;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    clSetKernelArg(kernel, 6, sizeof(int), &j);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = pimd->N;
    md_get_work_size_x(kernel2, device, 1, size, global, local);
    clSetKernelArg(kernel2, 5, sizeof(int), &j);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = pimd2->N;
    md_get_work_size_x(kernel3, device, 1, size, global, local);
    clSetKernelArg(kernel3, 5, sizeof(int), &j);
    clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
  }
  pimd->queue = queue;
  sim->queue = queue;
  sim2->queue = queue;
#else
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
#endif
  return 0;
}

int md_pimd_calc_pair_energy_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_pair_energy_t_x pe, md_stats_t_x *stats) {
  if (pimd->P != pimd2->P)
    return -1;
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_pimd_to_context_x(pimd, context);
  md_simulation_to_context_x(sim, context);
  md_simulation_to_context_x(sim2, context);
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
  md_get_pair_energy_info_x(pe, "md_pimd_calc_pair_energy_2_", "_kx", kname, &params);
  if (pimd->cpe2_kernel != NULL && strcmp(pimd->pe2kname, kname)) {
    clReleaseKernel(pimd->cpe2_kernel);
    pimd->cpe2_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (pimd->cpe2_kernel != NULL)
    kernel = pimd->cpe2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    pimd->cpe2_kernel = kernel;
    strcpy(pimd->pe2kname, kname);
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &sim2->particles_mem);
  clSetKernelArg(kernel, 3, sizeof(int), &pimd2->N);
  clSetKernelArg(kernel, 7, sizeof(md_inter_params_t_x), &params);
  clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect2);
  cl_kernel kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    sim->add_kernel = kernel2;
  }
  md_num_t_x mult = 1.0;
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->e_mem);
  clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  int j;
  for (j = 1; j <= pimd->P; ++j) {
    int size[1];
    size[0] = pimd->N*pimd2->N;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    int group = global[0]/local[0];
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel, 6, sizeof(int), &j);
    clSetKernelArg(kernel, 9, sizeof(md_num_t_x)*local[0], NULL);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel2, 2, sizeof(int), &group);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    clReleaseMemObject(out_mem);
  }
  pimd->queue = queue;
  sim->queue = queue;
  sim2->queue = queue;
  stats->queue = queue;
#else
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
#endif
  return 0;
}

void md_pimd_calc_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats) {
  md_simulation_sync_host_x(pimd->sim, 1);
  md_stats_sync_host_x(stats);
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
  md_simulation_sync_host_x(pimd->sim, 1);
  md_stats_sync_host_x(stats);
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
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_pimd_to_context_x(pimd, context);
  md_simulation_t_x *sim = pimd->sim;
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
  if (pimd->fillE2_kernel != NULL)
    kernel = pimd->fillE2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fill_ENk2_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk2_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->pair_in_mem);
    clSetKernelArg(kernel, 3, sizeof(int), &pimd->pcount_in);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    pimd->fillE2_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->pcount_in;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clSetKernelArg(kernel, 4, sizeof(int), &image);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
#else
  int l, j;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= l; ++j)
      pimd->ENk2[l-1][j-1] = md_pimd_ENk2_x(pimd, l, j, image);
#endif
}

int md_pimd_fprint_ENk2_x(FILE *out, md_pimd_t_x *pimd) {
  md_pimd_sync_host_x(pimd, 1);
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
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_pimd_to_context_x(pimd, context);
  md_simulation_t_x *sim = pimd->sim;
  cl_mem res_mem = NULL;
  if (pimd->eVBN_mem != NULL)
    res_mem = pimd->eVBN_mem;
  else {
    res_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    pimd->eVBN_mem = res_mem;
  }
  clEnqueueWriteBuffer(queue, pimd->eVBN_mem, CL_FALSE, 0, sizeof(md_num_t_x), &md_VBN_0_x, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel = NULL;
  if (pimd->filleV2_kernel != NULL)
    kernel = pimd->filleV2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_VBN2_energy_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk2_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->VBN_mem);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 6, sizeof(int), &pimd->N);
    pimd->filleV2_kernel = kernel;
  }
  clSetKernelArg(kernel, 5, sizeof(cl_mem), &res_mem);
  clSetKernelArg(kernel, 8, sizeof(md_num_t_x), &pimd->vi);
  clSetKernelArg(kernel, 9, sizeof(md_num_t_x), &pimd->beta);
  clSetKernelArg(kernel, 10, sizeof(cl_mem), &pimd->minE_mem);
  cl_kernel kernel2 = NULL;
  if (pimd->addeV_kernel != NULL)
    kernel2 = pimd->addeV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_add_eVBN_kx", &status);
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &pimd->VBN_mem);
    pimd->addeV_kernel = kernel2;
  }
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &pimd->eVBN_mem);
  clSetKernelArg(kernel2, 3, sizeof(cl_mem), &pimd->minE_mem);
  clSetKernelArg(kernel2, 6, sizeof(md_num_t_x), &pimd->beta);
  int N2;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    int size[1];
    size[0] = N2;
    size_t local[1], global[1];
    md_get_work_size_x(kernel, device, 1, size, global, local);
    int group = global[0]/local[0];
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel, 7, sizeof(int), &N2);
    clSetKernelArg(kernel, 11, sizeof(md_num_t_x)*local[0], NULL);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    clSetKernelArg(kernel2, 2, sizeof(cl_mem), &out_mem);
    clSetKernelArg(kernel2, 4, sizeof(int), &group);
    clSetKernelArg(kernel2, 5, sizeof(int), &N2);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
    clReleaseMemObject(out_mem);
  }
  md_stats_to_context_x(stats, context);
  cl_kernel kernel3 = NULL;
  if (pimd->addeVst2_kernel != NULL)
    kernel3 = pimd->addeVst2_kernel;
  else {
    kernel3 = clCreateKernel(md_programs_x[plat], "md_pimd_add_eVBN2_stats_kx", &status);
    clSetKernelArg(kernel3, 2, sizeof(int), &pimd->N);
    pimd->addeVst2_kernel = kernel3;
  }
  clSetKernelArg(kernel3, 0, sizeof(cl_mem), &stats->e_mem);
  clSetKernelArg(kernel3, 1, sizeof(cl_mem), &pimd->eVBN_mem);
  clSetKernelArg(kernel3, 3, sizeof(md_num_t_x), &pimd->beta);
  size_t local[1], global[1];
  local[0] = 1;
  global[0] = 1;
  clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  cl_kernel kernel4 = NULL;
  if (pimd->cvire_kernel != NULL)
    kernel4 = pimd->cvire_kernel;
  else {
    kernel4 = clCreateKernel(md_programs_x[plat], "md_pimd_calc_virial_energy_kx", &status);
    clSetKernelArg(kernel4, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel4, 2, sizeof(int), &pimd->N);
    clSetKernelArg(kernel4, 3, sizeof(int), &pimd->P);
    pimd->cvire_kernel = kernel4;
  }
  int size[1];
  size[0] = pimd->N*pimd->P;
  md_get_work_size_x(kernel4, device, 1, size, global, local);
  int group = global[0]/local[0];
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  clSetKernelArg(kernel4, 1, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel4, 4, sizeof(md_num_t_x)*local[0], NULL);
  clEnqueueNDRangeKernel(queue, kernel4, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    sim->add_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  md_num_t_x mult = 1.0/2.0;
  clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->e_mem);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  clSetKernelArg(kernel2, 2, sizeof(int), &group);
  clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  pimd->queue = queue;
  sim->queue = queue;
  stats->queue = queue;
  clReleaseMemObject(out_mem);
#else
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
#endif
}

void md_pimd_polymer_periodic_boundary_x(md_pimd_t_x *pimd) {
  md_simulation_sync_host_x(pimd->sim, 0);
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
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  md_simulation_to_context_x(pimd->sim, context);
  md_pimd_to_context_x(pimd, context);
  md_simulation_t_x *sim = pimd->sim;
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
  if (pimd->ffillE21_kernel != NULL)
    kernel = pimd->ffillE21_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_ENk2_1_kx", &status);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk2_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel, 3, sizeof(int), &pimd->N);
    clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    pimd->ffillE21_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  md_get_work_size_x(kernel, device, 1, size, global, local);
  clSetKernelArg(kernel, 4, sizeof(int), &image);
  clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  md_update_queue_x(queue);
  int j;
  cl_kernel kernel2 = NULL;
  if (pimd->ffillE22_kernel != NULL)
    kernel2 = pimd->ffillE22_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_ENk2_2_kx", &status);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), &pimd->ENk2_mem);
    clSetKernelArg(kernel2, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    clSetKernelArg(kernel2, 4, sizeof(int), &pimd->N);
    clSetKernelArg(kernel2, 6, sizeof(int), &pimd->P);
    pimd->ffillE22_kernel = kernel2;
  }
  clSetKernelArg(kernel2, 5, sizeof(int), &image);
  clSetKernelArg(kernel2, 7, sizeof(md_num_t_x), &pimd->omegaP);
  clSetKernelArg(kernel2, 8, sizeof(nd_rect_t_x), &rect2);
  for (j = 2; j <= pimd->N; ++j) {
    size[0] = pimd->N-j+1;
    md_get_work_size_x(kernel2, device, 1, size, global, local);
    clSetKernelArg(kernel2, 3, sizeof(int), &j);
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    md_update_queue_x(queue);
  }
  pimd->queue = queue;
  sim->queue = queue;
#else
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
#endif
}

void md_reset_stats_x(md_stats_t_x *stats, md_num_t_x e) {
  md_stats_sync_host_x(stats);
  stats->es[0] = e;
}

void md_stats_add_to_x(md_stats_t_x *stats, md_stats_t_x *stats2) {
  md_stats_sync_host_x(stats);
  md_stats_sync_host_x(stats2);
  stats->es[stats->N-1] += stats2->es[0];
}

void md_stats_copy_to_x(md_stats_t_x *stats, md_stats_t_x *stats2) {
  md_stats_sync_host_x(stats);
  md_stats_sync_host_x(stats2);
  stats->es[0] = stats2->es[0];
}

md_num_t_x md_pimd_xminE2_x(md_pimd_t_x *pimd, int N2, md_num_t_x *VBN2, md_num_t_x vi2) {
  if (vi2 == 0.0)
    return pimd->beta*(pimd->ENk[N2-1][0]+VBN2[N2-1]);
  int k;
  md_num_t_x res = 1e10;
  for (k = 1; k <= N2; ++k) {
    md_num_t_x tmp = -(k-1)*log(vi2)+pimd->beta*(pimd->ENk[N2-1][k-1]+VBN2[N2-k]);
    if (tmp < res)
      res = tmp;
  }
  return res;
}

void md_pimd_calc_vi_sign_x(md_pimd_t_x *pimd, md_stats_t_x *e_s, md_stats_t_x *s_s, md_num_t_x vi2) {
  md_pimd_sync_host_x(pimd, 1);
  md_stats_sync_host_x(e_s);
  md_stats_sync_host_x(s_s);
  md_num_t_x *VBN2 = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  md_num_t_x *sign = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  VBN2[0] = 0;
  sign[0] = 1;
  int N2, k;
  md_num_t_x sv = 1;
  if (vi2 < 0)
    sv = -1;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    md_num_t_x sum = 0;
    md_num_t_x tmp = md_pimd_xminE2_x(pimd, N2, VBN2, fabs(vi2));
    for (k = 1; k <= N2; ++k) {
      if (vi2 == 0 && k-1 != 0)
        continue;
      sum += sign[N2-k]*pow(sv, k-1)*md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+VBN2[N2-k], tmp, pimd->beta, fabs(vi2));
    }
    if (sum >= 0)
      sign[N2] = 1;
    else
      sign[N2] = -1;
    VBN2[N2] = (tmp-log(fabs(sum))+log(N2))/pimd->beta;
  }
  md_num_t_x s = sign[pimd->N]*exp(-pimd->beta*VBN2[pimd->N]+pimd->beta*pimd->VBN[pimd->N]);
  e_s->es[0] *= s;
  s_s->es[0] *= s;
  free(VBN2);
  free(sign);
}