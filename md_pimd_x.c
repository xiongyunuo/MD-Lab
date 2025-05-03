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
  if (pimd->cITCF2_kernel != NULL) {
    status |= clReleaseKernel(pimd->cITCF2_kernel);
    pimd->cITCF2_kernel = NULL;
  }
  if (pimd->cpc_kernel != NULL) {
    status |= clReleaseKernel(pimd->cpc_kernel);
    pimd->cpc_kernel = NULL;
  }
  if (pimd->cpc2_kernel != NULL) {
    status |= clReleaseKernel(pimd->cpc2_kernel);
    pimd->cpc2_kernel = NULL;
  }
  if (pimd->cSk_kernel != NULL) {
    status |= clReleaseKernel(pimd->cSk_kernel);
    pimd->cSk_kernel = NULL;
  }
  if (pimd->cSk2_kernel != NULL) {
    status |= clReleaseKernel(pimd->cSk2_kernel);
    pimd->cSk2_kernel = NULL;
  }
  if (pimd->cSkf2_kernel != NULL) {
    status |= clReleaseKernel(pimd->cSkf2_kernel);
    pimd->cSkf2_kernel = NULL;
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
  pimd->Eint = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->N);
  pimd->Evir = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  if (pimd->VBN == NULL || pimd->minE == NULL || pimd->Eint == NULL || pimd->Evir == NULL) {
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
  pimd->ENk3 = (md_num_t_x **)malloc(sizeof(md_num_t_x *)*N);
  if (pimd->ENk3 == NULL) {
    free(pimd);
    return NULL;
  }
  pimd->ENk3[0] = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->ENk_count);
  if (pimd->ENk3[0] == NULL) {
    free(pimd);
    return NULL;
  }
  count = 1;
  for (i = 2; i <= N; ++i) {
    pimd->ENk3[i-1] = &pimd->ENk3[0][count];
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
  pimd->cut_Pj = -1;
  pimd->mc_par = -1;
  pimd->pbc = 0;
  pimd->pbW = (int *)malloc(sizeof(int)*pimd->N*pimd->P*MD_DIMENSION_X);
  if (pimd->pbW == NULL) {
    free(pimd);
    return NULL;
  }
  for (i = 0; i < pimd->N*pimd->P*MD_DIMENSION_X; ++i)
    pimd->pbW[i] = 0;
  pimd->worm_index = -1;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->worm_pos[i] = 0;
  pimd->old_worm_index = -1;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->old_worm_pos[i] = 0;
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
  pimd->cITCF2_kernel = NULL;
  pimd->cpc_kernel = NULL;
  pimd->cpc2_kernel = NULL;
  pimd->cSk_kernel = NULL;
  pimd->cSk2_kernel = NULL;
  pimd->cSkf2_kernel = NULL;
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

int md_pimd_init_particle_uniform_pos_x(md_pimd_t_x *pimd, md_num_t_x *center, md_num_t_x *L, md_num_t_x fluc) {
  pimd->sim = md_simulation_sync_host_x(pimd->sim, 0);
  if (pimd->sim == NULL)
    return -1;
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
  return 0;
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

int md_pimd_fill_ENk_x(md_pimd_t_x *pimd, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
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
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->pair_in_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->pcount_in);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->fillE_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->pcount_in;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel, 4, sizeof(int), &image);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
  return 0;
}

int md_pimd_fprint_ENk_x(FILE *out, md_pimd_t_x *pimd) {
  pimd = md_pimd_sync_host_x(pimd, 1);
  if (pimd == NULL)
    return -1;
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
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  /*cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  clReleaseMemObject(pimd->VBN_mem);
  pimd->VBN_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), pimd->VBN, &status);*/
  cl_kernel kernel = NULL;
  if (pimd->minE_kernel != NULL)
    kernel = pimd->minE_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_xminE_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->VBN_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    if (status != 0)
      return status;
    pimd->minE_kernel = kernel;
  }
  int size[1];
  size[0] = N2;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  int group = global[0]/local[0];
  //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &N2);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  status |= clSetKernelArg(kernel, 8, sizeof(md_num_t_x)*local[0], NULL);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_mem minE_mem = NULL;
  if (pimd->minE_mem != NULL)
    minE_mem = pimd->minE_mem;
  else {
    minE_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    if (status != 0)
      return status;
    pimd->minE_mem = minE_mem;
  }
  cl_kernel kernel2 = NULL;
  if (pimd->min_kernel != NULL)
    kernel2 = pimd->min_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_min_kx", &status);
    if (status != 0)
      return status;
    pimd->min_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &minE_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &group);
  status |= clSetKernelArg(kernel2, 3, sizeof(int), &N2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
  status = clReleaseMemObject(out_mem);
  if (status != 0)
    return status;
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

int md_pimd_fill_VB_x(md_pimd_t_x *pimd) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  //cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  int N2;
  //pimd->VBN[0] = 0.0;
  //clReleaseMemObject(pimd->VBN_mem);
  //pimd->VBN_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), pimd->VBN, &status);
  status = clEnqueueWriteBuffer(queue, pimd->VBN_mem, CL_FALSE, 0, sizeof(md_num_t_x), &md_VBN_0_x, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->fillV_kernel != NULL)
    kernel = pimd->fillV_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fill_VB_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->VBN_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    if (status != 0)
      return status;
    pimd->fillV_kernel = kernel;
  }
  status = clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (pimd->addV_kernel != NULL)
    kernel2 = pimd->addV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_add_VBN_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &pimd->VBN_mem);
    if (status != 0)
      return status;
    pimd->addV_kernel = kernel2;
  }
  status = clSetKernelArg(kernel2, 5, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    //md_num_t_x tmp = md_pimd_xminE_x(pimd, N2);
    //pimd->minE[N2] = tmp;
    status = (cl_int)md_pimd_xminE_x(pimd, N2);
    if (status != 0)
      return status;
    int size[1];
    size[0] = N2;
    size_t local[1], global[1];
    status = md_get_work_size_x(kernel, device, 1, size, global, local);
    if (status != 0)
      return status;
    int group = global[0]/local[0];
    //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &N2);
    status |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &pimd->minE_mem);
    status |= clSetKernelArg(kernel, 9, sizeof(md_num_t_x)*local[0], NULL);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &pimd->minE_mem);
    status |= clSetKernelArg(kernel2, 3, sizeof(int), &group);
    status |= clSetKernelArg(kernel2, 4, sizeof(int), &N2);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
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
    status = clReleaseMemObject(out_mem);
    if (status != 0)
      return status;
    //clReleaseEvent(events[0]);
    //free(output);
  }
  pimd->queue = queue;
#else
  int N2, k;
  pimd->VBN[0] = 0.0;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    if (pimd->worm_index > 0 && N2 == pimd->N) {
      pimd->VBN[N2] = pimd->VBN[N2-1];
      continue;
    }
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
  return 0;
}

int md_pimd_fprint_VBN_x(FILE *out, md_pimd_t_x *pimd) {
  pimd = md_pimd_sync_host_x(pimd, 1);
  if (pimd == NULL)
    return -1;
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

int md_pimd_fill_force_VB_x(md_pimd_t_x *pimd, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
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
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->VBN_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 6, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->fillFV_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N*pimd->P;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  //cl_mem minE_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(md_num_t_x)*(pimd->N+1), pimd->minE, &status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &pimd->minE_mem);
  status |= clSetKernelArg(kernel, 7, sizeof(int), &image);
  status |= clSetKernelArg(kernel, 8, sizeof(md_num_t_x), &pimd->vi);
  status |= clSetKernelArg(kernel, 9, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel, 10, sizeof(md_num_t_x), &pimd->beta);
  status |= clSetKernelArg(kernel, 11, sizeof(nd_rect_t_x), &rect2);
  status |= clSetKernelArg(kernel, 12, sizeof(md_num_t_x)*MD_DIMENSION_X*(pimd->N+1)*local[0], NULL);
  status |= clSetKernelArg(kernel, 13, sizeof(md_num_t_x)*MD_DIMENSION_X*pimd->N*local[0], NULL);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
  if (res == NULL || grad == NULL)
    return -1;
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
  return 0;
}

int md_pimd_calc_VBN_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  //cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  cl_mem res_mem = NULL;
  if (pimd->eVBN_mem != NULL)
    res_mem = pimd->eVBN_mem;
  else {
    res_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    if (status != 0)
      return status;
    pimd->eVBN_mem = res_mem;
  }
  status = clEnqueueWriteBuffer(queue, pimd->eVBN_mem, CL_FALSE, 0, sizeof(md_num_t_x), &md_VBN_0_x, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->filleV_kernel != NULL)
    kernel = pimd->filleV_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_VBN_energy_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->VBN_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->N);
    if (status != 0)
      return status;
    pimd->filleV_kernel = kernel;
  }
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &res_mem);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->vi);
  status |= clSetKernelArg(kernel, 8, sizeof(md_num_t_x), &pimd->beta);
  status |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &pimd->minE_mem);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (pimd->addeV_kernel != NULL)
    kernel2 = pimd->addeV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_add_eVBN_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &pimd->VBN_mem);
    if (status != 0)
      return status;
    pimd->addeV_kernel = kernel2;
  }
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &pimd->eVBN_mem);
  status |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), &pimd->minE_mem);
  status |= clSetKernelArg(kernel2, 6, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  int N2;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    //md_num_t_x tmp = pimd->minE[N2];
    int size[1];
    size[0] = N2;
    size_t local[1], global[1];
    status = md_get_work_size_x(kernel, device, 1, size, global, local);
    if (status != 0)
      return status;
    int group = global[0]/local[0];
    //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel, 6, sizeof(int), &N2);
    status |= clSetKernelArg(kernel, 10, sizeof(md_num_t_x)*local[0], NULL);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    status = clSetKernelArg(kernel2, 2, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel2, 4, sizeof(int), &group);
    status |= clSetKernelArg(kernel2, 5, sizeof(int), &N2);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
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
    status = clReleaseMemObject(out_mem);
    if (status != 0)
      return status;
    //clReleaseEvent(events[0]);
    //free(output);
  }
  //clReleaseMemObject(res_mem);
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel3 = NULL;
  if (pimd->addeVst_kernel != NULL)
    kernel3 = pimd->addeVst_kernel;
  else {
    kernel3 = clCreateKernel(md_programs_x[plat], "md_pimd_add_eVBN_stats_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel3, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel3, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->addeVst_kernel = kernel3;
  }
  status = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &stats->e_mem);
  status |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &pimd->eVBN_mem);
  status |= clSetKernelArg(kernel3, 4, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  size_t local[1], global[1];
  local[0] = 1;
  global[0] = 1;
  status = clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  pimd->queue = queue;
  stats->queue = queue;
#else
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  if (res == NULL)
    return -1;
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
  return 0;
}

int md_pimd_calc_trap_force_x(md_pimd_t_x *pimd, md_trap_force_t_x tf) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
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
    status = clReleaseKernel(pimd->ctf_kernel);
    if (status != 0)
      return status;
    pimd->ctf_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (pimd->ctf_kernel != NULL)
    kernel = pimd->ctf_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->ctf_kernel = kernel;
    strcpy(pimd->tfkname, kname);
  }
  int size[1];
  size[0] = pimd->N*pimd->P;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel, 3, sizeof(md_inter_params_t_x), &params);
  status |= clSetKernelArg(kernel, 4, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
  return 0;
}

int md_pimd_calc_trap_energy_x(md_pimd_t_x *pimd, md_trap_energy_t_x te, md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
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
    status = clReleaseKernel(pimd->cte_kernel);
    if (status != 0)
      return status;
    pimd->cte_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (pimd->cte_kernel != NULL)
    kernel = pimd->cte_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cte_kernel = kernel;
    strcpy(pimd->tekname, kname);
  }
  int size[1];
  size[0] = pimd->N*pimd->P;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  int group = global[0]/local[0];
  //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(md_inter_params_t_x), &params);
  status |= clSetKernelArg(kernel, 5, sizeof(nd_rect_t_x), &rect2);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x)*local[0], NULL);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    if (status != 0)
      return status;
    sim->add_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  md_num_t_x mult = 1.0;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->e_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &group);
  status |= clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
  status = clReleaseMemObject(out_mem);
  if (status != 0)
    return status;
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
  return 0;
}

int md_pimd_calc_pair_force_x(md_pimd_t_x *pimd, md_pair_force_t_x pf) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
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
    status = clReleaseKernel(pimd->cpf_kernel);
    if (status != 0)
      return status;
    pimd->cpf_kernel = NULL;
  }
  int j;
  cl_mem forces = NULL;
  if (pimd->forces != NULL)
    forces = pimd->forces;
  else {
    forces = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*pimd->N*pimd->N*MD_DIMENSION_X, NULL, &status);
    if (status != 0)
      return status;
    pimd->forces = forces;
  }
  cl_kernel kernel = NULL;
  if (pimd->cpf_kernel != NULL)
    kernel = pimd->cpf_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->pair_ex_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->pcount_ex);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cpf_kernel = kernel;
    strcpy(pimd->pfkname, kname);
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &forces);
  status |= clSetKernelArg(kernel, 7, sizeof(md_inter_params_t_x), &params);
  status |= clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (pimd->upf_kernel != NULL)
    kernel2 = pimd->upf_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_update_pair_force_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel2, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->upf_kernel = kernel2;
  }
  status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &forces);
  if (status != 0)
    return status;
  for (j = 1; j <= pimd->P; ++j) {
    int size[1];
    size[0] = pimd->pcount_ex;
    size_t local[1], global[1];
    status = md_get_work_size_x(kernel, device, 1, size, global, local);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 6, sizeof(int), &j);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = pimd->N;
    status = md_get_work_size_x(kernel2, device, 1, size, global, local);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 4, sizeof(int), &j);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
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
  return 0;
}

int md_pimd_calc_pair_energy_x(md_pimd_t_x *pimd, md_pair_energy_t_x pe, md_stats_t_x *stats) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
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
    status = clReleaseKernel(pimd->cpe_kernel);
    if (status != 0)
      return status;
    pimd->cpe_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (pimd->cpe_kernel != NULL)
    kernel = pimd->cpe_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->pair_ex_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->pcount_ex);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cpe_kernel = kernel;
    strcpy(pimd->pekname, kname);
  }
  status = clSetKernelArg(kernel, 6, sizeof(md_inter_params_t_x), &params);
  status |= clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    if (status != 0)
      return status;
    sim->add_kernel = kernel2;
  }
  md_num_t_x mult = 1.0;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->e_mem);
  status |= clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  if (status != 0)
    return status;
  int j;
  for (j = 1; j <= pimd->P; ++j) {
    int size[1];
    size[0] = pimd->pcount_ex;
    size_t local[1], global[1];
    status = md_get_work_size_x(kernel, device, 1, size, global, local);
    if (status != 0)
      return status;
    int group = global[0]/local[0];
    //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &j);
    status |= clSetKernelArg(kernel, 8, sizeof(md_num_t_x)*local[0], NULL);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel2, 2, sizeof(int), &group);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    //cl_event events[1];
    //clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, sizeof(md_num_t_x)*group, output, 0, NULL, &events[0]);
    //clWaitForEvents(1, events);
    //md_num_t_x res = 0;
    //for (i = 0; i < group; ++i)
      //res += output[i];
    //stats->es[stats->N-1] += res;
    //clReleaseKernel(kernel);
    status = clReleaseMemObject(out_mem);
    if (status != 0)
      return status;
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
  return 0;
}

int md_pimd_calc_density_distribution_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmax, int image, md_num_t_x *center) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
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
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cdd_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  if (pimd->den_mem != NULL && pimd->points != stats->points) {
    status = clReleaseMemObject(pimd->den_mem);
    if (status != 0)
      return status;
    pimd->den_mem = NULL;
  }
  cl_mem den_mem = NULL;
  if (pimd->den_mem != NULL)
    den_mem = pimd->den_mem;
  else {
    den_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*stats->points*pimd->N, NULL, &status);
    if (status != 0)
      return status;
    pimd->den_mem = den_mem;
    pimd->points = stats->points;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &image);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &rmax);
  status |= clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  status |= clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect3);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (sim->aden_kernel != NULL)
    kernel2 = sim->aden_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_density_kx", &status);
    if (status != 0)
      return status;
    sim->aden_kernel = kernel2;
  }
  size[0] = stats->points;
  status = md_get_work_size_x(kernel2, device, 1, size, global, local);
  if (status != 0)
    return status;
  //cl_mem fx_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(md_num_t_x)*stats->points, stats->fxs[stats->N-1], &status);
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &stats->fx_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
  status |= clSetKernelArg(kernel2, 3, sizeof(int), &stats->points);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
        continue; //index = stats->points-1;
      stats->fxs[stats->N-1][index] += 1.0/pimd->P;
    }
#endif
  return 0;
}

int md_pimd_fast_fill_ENk_x(md_pimd_t_x *pimd, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
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
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 6, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->ffillE_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  cl_mem Eint_mem = NULL;
  if (pimd->Eint_mem != NULL)
    Eint_mem = pimd->Eint_mem;
  else {
    Eint_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*pimd->N, NULL, &status);
    if (status != 0)
      return status;
    pimd->Eint_mem = Eint_mem;
  }
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &Eint_mem);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &image);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  int j;
  cl_kernel kernel2 = NULL;
  if (pimd->ffillE2_kernel != NULL)
    kernel2 = pimd->ffillE2_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_ENk_2_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel2, 5, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel2, 7, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->ffillE2_kernel = kernel2;
  }
  status = clSetKernelArg(kernel2, 3, sizeof(cl_mem), &Eint_mem);
  status |= clSetKernelArg(kernel2, 6, sizeof(int), &image);
  status |= clSetKernelArg(kernel2, 8, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel2, 9, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  for (j = 2; j <= pimd->N; ++j) {
    size[0] = pimd->N-j+1;
    status = md_get_work_size_x(kernel2, device, 1, size, global, local);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 4, sizeof(int), &j);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
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
  //pimd->Eint = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->N);
  md_num_t_x *Eint = pimd->Eint;
  if (Eint == NULL)
    return -1;
  int index, index2;
  md_num_t_x d;
  int i;
  int L0[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    L0[i] = 0;
  for (l = 1; l <= pimd->N; ++l) {
    if (pimd->worm_index > 0 && l == pimd->N)
      continue;
    md_num_t_x res = 0;
    for (j = 1; j < pimd->P; ++j) {
      if (j == pimd->cut_Pj)
        continue;
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, j+1, pimd->P);
      if (pimd->worm_index > 0 && l >= pimd->worm_index) {
        index += pimd->P;
        index2 += pimd->P;
      }
      if (pimd->pbc && j == 1)
        d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, L0, &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (pimd->pbc)
        d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      res += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }
    /*if (l == pimd->worm_index) {
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      d = md_minimum_image_distance_2_x(sim->particles[index].x, pimd->worm_pos, L, &pimd->pbW[index*MD_DIMENSION_X], L0);
      res += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }*/
    Eint[l-1] = res;
    /*if (l == pimd->worm_index) {
      pimd->ENk[l-1][0] = res;
      continue;
    }*/
    index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
    if (pimd->worm_index > 0 && l >= pimd->worm_index) {
      index += pimd->P;
      index2 += pimd->P;
    }
    if (pimd->pbc)
      d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
    else if (image)
      d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
    else
      d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
    pimd->ENk[l-1][0] = res+0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
  }
  for (l = 1; l <= pimd->N; ++l)
    for (j = 2; j <= l; ++j) {
      if (pimd->worm_index > 0 && l == pimd->N)
        continue;
      pimd->ENk[l-1][j-1] = pimd->ENk[l-1][j-2]+Eint[l-j];
      //if (l == pimd->worm_index)
        //continue;
      if (pimd->P == pimd->cut_Pj)
        continue;
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      if (pimd->worm_index > 0 && l >= pimd->worm_index)
        index += pimd->P;
      if (pimd->worm_index > 0 && l-j+2 >= pimd->worm_index)
        index2 += pimd->P;
      if (pimd->pbc && l == l-j+2)
        d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (pimd->pbc)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L); //d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      //if (l == pimd->worm_index || l-j+2 == pimd->worm_index)
        //d = 0;
      pimd->ENk[l-1][j-1] -= 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
      index = MD_PIMD_INDEX_X(l-j+1, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      if (pimd->worm_index > 0 && l-j+1 >= pimd->worm_index)
        index += pimd->P;
      if (pimd->worm_index > 0 && l-j+2 >= pimd->worm_index)
        index2 += pimd->P;
      if (pimd->pbc)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L); //d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      //if (l-j+1 == pimd->worm_index || l-j+2 == pimd->worm_index)
        //d = 0;
      pimd->ENk[l-1][j-1] += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      if (pimd->worm_index > 0 && l >= pimd->worm_index)
        index += pimd->P;
      if (pimd->worm_index > 0 && l-j+1 >= pimd->worm_index)
        index2 += pimd->P;
      if (pimd->pbc)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L); //d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      //if (l == pimd->worm_index || l-j+1 == pimd->worm_index)
        //d = 0;
      pimd->ENk[l-1][j-1] += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }
  //free(Eint);
#endif
  return 0;
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
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  //cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  //cl_mem V_mem = clCreateBuffer(context, flags, sizeof(md_num_t_x)*(pimd->N+1), V, &status);
  cl_kernel kernel = NULL;
  if (pimd->fminE_kernel != NULL)
    kernel = pimd->fminE_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fast_xminE_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    if (status != 0)
      return status;
    pimd->fminE_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N-u+1;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  int group = global[0]/local[0];
  //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->V_mem);
  status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &u);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  status |= clSetKernelArg(kernel, 8, sizeof(md_num_t_x)*local[0], NULL);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_mem minE_mem = NULL;
  if (pimd->fminE_mem != NULL)
    minE_mem = pimd->fminE_mem;
  else {
    minE_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    if (status != 0)
      return status;
    pimd->fminE_mem = minE_mem;
  }
  cl_kernel kernel2 = NULL;
  if (pimd->min_kernel != NULL)
    kernel2 = pimd->min_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_min_kx", &status);
    if (status != 0)
      return status;
    pimd->min_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &minE_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &group);
  status |= clSetKernelArg(kernel2, 3, sizeof(int), &u);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
  status = clReleaseMemObject(out_mem);
  if (status != 0)
    return status;
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
      if (index+1 != index2 || (index%pimd->P)+1 != pimd->cut_Pj)
        dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
      if (index-1 != index3 || (index%pimd->P)+1 != pimd->cut_Pj+1)
        dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index3].x[i], L[i]);
    }
    else {
      if (index+1 != index2 || (index%pimd->P)+1 != pimd->cut_Pj)
        dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*(sim->particles[index].x[i]-sim->particles[index2].x[i]);
      if (index-1 != index3 || (index%pimd->P)+1 != pimd->cut_Pj+1)
        dENk[i] += sim->particles[index].m*pimd->omegaP*pimd->omegaP*(sim->particles[index].x[i]-sim->particles[index3].x[i]);
    }
  }
}

int md_pimd_fast_fill_force_VB_x(md_pimd_t_x *pimd, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
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
    if (status != 0)
      return status;
    pimd->V_mem = tmpV_mem;
  }
  status = clEnqueueWriteBuffer(queue, pimd->V_mem, CL_FALSE, pimd->N*sizeof(md_num_t_x), sizeof(md_num_t_x), &md_VBN_0_x, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->ffillV_kernel != NULL)
    kernel = pimd->ffillV_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_VB_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    if (status != 0)
      return status;
    pimd->ffillV_kernel = kernel;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmpV_mem);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (pimd->faddV_kernel != NULL)
    kernel2 = pimd->faddV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_add_VBN_kx", &status);
    if (status != 0)
      return status;
    pimd->faddV_kernel = kernel2;
  }
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &tmpV_mem);
  status |= clSetKernelArg(kernel2, 5, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  int u;
  for (u = pimd->N; u >= 1; --u) {
    //md_num_t_x tmp = md_pimd_fast_xminE_x(pimd, u, tmpV);
    status = (cl_int)md_pimd_fast_xminE_x(pimd, u, NULL);
    if (status != 0)
      return status;
    int size[1];
    size[0] = pimd->N-u+1;
    size_t local[1], global[1];
    status = md_get_work_size_x(kernel, device, 1, size, global, local);
    if (status != 0)
      return status;
    int group = global[0]/local[0];
    //md_num_t_x *output = (md_num_t_x *)malloc(sizeof(md_num_t_x)*group);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &u);
    status |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &pimd->fminE_mem);
    status |= clSetKernelArg(kernel, 9, sizeof(md_num_t_x)*local[0], NULL);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &pimd->fminE_mem);
    status |= clSetKernelArg(kernel2, 3, sizeof(int), &group);
    status |= clSetKernelArg(kernel2, 4, sizeof(int), &u);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
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
    status = clReleaseMemObject(out_mem);
    if (status != 0)
      return status;
    //clReleaseEvent(events[0]);
    //free(output);
  }
  cl_mem G_mem = NULL;
  if (pimd->G_mem != NULL)
    G_mem = pimd->G_mem;
  else {
    G_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*pimd->N*pimd->N, NULL, &status);
    if (status != 0)
      return status;
    pimd->G_mem = G_mem;
  }
  if (pimd->ffillG_kernel != NULL)
    kernel = pimd->ffillG_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_G_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->VBN_mem);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->N);
    if (status != 0)
      return status;
    pimd->ffillG_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N*pimd->N;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &tmpV_mem);
  status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &G_mem);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->vi);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  if (pimd->ffillFV_kernel != NULL)
    kernel2 = pimd->ffillFV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_force_VB_1_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel2, 1, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel2, 2, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->ffillFV_kernel = kernel2;
  }
  size[0] = pimd->N*(pimd->P-2);
  status = md_get_work_size_x(kernel2, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel2, 3, sizeof(int), &image);
  status |= clSetKernelArg(kernel2, 4, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel2, 5, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel3 = NULL;
  if (pimd->ffillFV2_kernel != NULL)
    kernel3 = pimd->ffillFV2_kernel;
  else {
    kernel3 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_force_VB_2_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel3, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel3, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->ffillFV2_kernel = kernel3;
  }
  size[0] = pimd->N;
  status = md_get_work_size_x(kernel3, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel3, 1, sizeof(cl_mem), &G_mem);
  status |= clSetKernelArg(kernel3, 4, sizeof(int), &image);
  status |= clSetKernelArg(kernel3, 5, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel3, 6, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel4 = NULL;
  if (pimd->ffillFV3_kernel != NULL)
    kernel4 = pimd->ffillFV3_kernel;
  else {
    kernel4 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_force_VB_3_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel4, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel4, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel4, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->ffillFV3_kernel = kernel4;
  }
  size[0] = pimd->N;
  status = md_get_work_size_x(kernel4, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel4, 1, sizeof(cl_mem), &G_mem);
  status |= clSetKernelArg(kernel4, 4, sizeof(int), &image);
  status |= clSetKernelArg(kernel4, 5, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel4, 6, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel4, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
  if (tmpV == NULL)
    return -1;
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
  if (G == NULL)
    return -1;
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
  return 0;
}

int md_pimd_calc_ITCF_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmin, md_num_t_x rmax, int pi) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->cITCF_kernel != NULL)
    kernel = pimd->cITCF_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_ITCF_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cITCF_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  if (pimd->den_mem != NULL && pimd->points != stats->points) {
    status = clReleaseMemObject(pimd->den_mem);
    if (status != 0)
      return status;
    pimd->den_mem = NULL;
  }
  cl_mem den_mem = NULL;
  if (pimd->den_mem != NULL)
    den_mem = pimd->den_mem;
  else {
    den_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*stats->points*pimd->N, NULL, &status);
    if (status != 0)
      return status;
    pimd->den_mem = den_mem;
    pimd->points = stats->points;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &pi);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &rmin);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &rmax);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (sim->aden_kernel != NULL)
    kernel2 = sim->aden_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_density_kx", &status);
    if (status != 0)
      return status;
    sim->aden_kernel = kernel2;
  }
  size[0] = stats->points;
  status = md_get_work_size_x(kernel2, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &stats->fx_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
  status |= clSetKernelArg(kernel2, 3, sizeof(int), &stats->points);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
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
      stats->fxs[stats->N-1][idx] += 1.0/pimd->P;
    }
#endif
  return 0;
}

int md_pimd_calc_ITCF_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_stats_t_x *stats, md_num_t_x rmin, md_num_t_x rmax, int pi) {
  if (pimd->P != pimd2->P)
    return -1;
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim2, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->cITCF2_kernel != NULL)
    kernel = pimd->cITCF2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_ITCF_2_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cITCF2_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  if (pimd->den_mem != NULL && pimd->points != stats->points) {
    status = clReleaseMemObject(pimd->den_mem);
    if (status != 0)
      return status;
    pimd->den_mem = NULL;
  }
  cl_mem den_mem = NULL;
  if (pimd->den_mem != NULL)
    den_mem = pimd->den_mem;
  else {
    den_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*stats->points*pimd->N, NULL, &status);
    if (status != 0)
      return status;
    pimd->den_mem = den_mem;
    pimd->points = stats->points;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &pi);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &rmin);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &rmax);
  status |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &sim2->particles_mem);
  status |= clSetKernelArg(kernel, 9, sizeof(int), &pimd2->N);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (sim->aden_kernel != NULL)
    kernel2 = sim->aden_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_density_kx", &status);
    if (status != 0)
      return status;
    sim->aden_kernel = kernel2;
  }
  size[0] = stats->points;
  status = md_get_work_size_x(kernel2, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &stats->fx_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
  status |= clSetKernelArg(kernel2, 3, sizeof(int), &stats->points);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  sim->queue = queue;
  sim2->queue = queue;
  stats->queue = queue;
#else
  int l, j;
  int index, index2;
  int n = (int)sqrt(stats->points);
  md_num_t_x incre = (rmax-rmin)/n;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd2->N; ++j) {
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(j, pi, pimd->P);
      int i1 = (int)((pimd->sim->particles[index].x[0]-rmin)/incre);
      int i2 = (int)((pimd2->sim->particles[index2].x[0]-rmin)/incre);
      int idx = i1*n+i2;
      if (idx < 0)
        idx = 0;
      if (idx >= stats->points)
        idx = stats->points-1;
      stats->fxs[stats->N-1][idx] += 1.0/pimd->P;
      idx = i2*n+i1;
      if (idx < 0)
        idx = 0;
      if (idx >= stats->points)
        idx = stats->points-1;
      stats->fxs[stats->N-1][idx] += 1.0/pimd->P;
    }
#endif
  return 0;
}

int md_pimd_calc_pair_correlation_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmax, md_num_t_x norm, int pi, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
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
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->cpc_kernel != NULL)
    kernel = pimd->cpc_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_pair_correlation_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cpc_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  if (pimd->den_mem != NULL && pimd->points != stats->points) {
    status = clReleaseMemObject(pimd->den_mem);
    if (status != 0)
      return status;
    pimd->den_mem = NULL;
  }
  cl_mem den_mem = NULL;
  if (pimd->den_mem != NULL)
    den_mem = pimd->den_mem;
  else {
    den_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*stats->points*pimd->N, NULL, &status);
    if (status != 0)
      return status;
    pimd->den_mem = den_mem;
    pimd->points = stats->points;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &pi);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &rmax);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &norm);
  status |= clSetKernelArg(kernel, 8, sizeof(int), &image);
  status |= clSetKernelArg(kernel, 9, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (sim->aden_kernel != NULL)
    kernel2 = sim->aden_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_density_kx", &status);
    if (status != 0)
      return status;
    sim->aden_kernel = kernel2;
  }
  size[0] = stats->points;
  status = md_get_work_size_x(kernel2, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &stats->fx_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
  status |= clSetKernelArg(kernel2, 3, sizeof(int), &stats->points);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  sim->queue = queue;
  stats->queue = queue;
#else
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  md_num_t_x incre = rmax/stats->points;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, j;
  int index, index2;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->N; ++j) {
      if (l == j)
        continue;
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(j, pi, pimd->P);
      md_num_t_x r;
      if (image)
        r = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        r = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      int index3 = (int)(r/incre);
      if (index3 >= stats->points)
        continue; //index3 = stats->points-1;
      stats->fxs[stats->N-1][index3] += 1.0/norm;
    }
#endif
  return 0;
}

int md_pimd_calc_pair_correlation_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_stats_t_x *stats, md_num_t_x rmax, md_num_t_x norm, int pi, int image) {
  if (pimd->P != pimd2->P)
    return -1;
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  nd_rect_t_x rect2;
  int i;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rect2.L[i] = L[i];
  }
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim2, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->cpc2_kernel != NULL)
    kernel = pimd->cpc2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_pair_correlation_2_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cpc2_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  if (pimd->den_mem != NULL && pimd->points != stats->points) {
    status = clReleaseMemObject(pimd->den_mem);
    if (status != 0)
      return status;
    pimd->den_mem = NULL;
  }
  cl_mem den_mem = NULL;
  if (pimd->den_mem != NULL)
    den_mem = pimd->den_mem;
  else {
    den_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*stats->points*pimd->N, NULL, &status);
    if (status != 0)
      return status;
    pimd->den_mem = den_mem;
    pimd->points = stats->points;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &pi);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &rmax);
  status |= clSetKernelArg(kernel, 7, sizeof(md_num_t_x), &norm);
  status |= clSetKernelArg(kernel, 8, sizeof(int), &image);
  status |= clSetKernelArg(kernel, 9, sizeof(nd_rect_t_x), &rect2);
  status |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &sim2->particles_mem);
  status |= clSetKernelArg(kernel, 11, sizeof(int), &pimd2->N);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (sim->aden_kernel != NULL)
    kernel2 = sim->aden_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_density_kx", &status);
    if (status != 0)
      return status;
    sim->aden_kernel = kernel2;
  }
  size[0] = stats->points;
  status = md_get_work_size_x(kernel2, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &den_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &stats->fx_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &pimd->N);
  status |= clSetKernelArg(kernel2, 3, sizeof(int), &stats->points);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  sim->queue = queue;
  sim2->queue = queue;
  stats->queue = queue;
#else
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_num_t_x *L = NULL;
  md_num_t_x incre = rmax/stats->points;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, j;
  int index, index2;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd2->N; ++j) {
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(j, pi, pimd->P);
      md_num_t_x r;
      if (image)
        r = md_minimum_image_distance_x(sim->particles[index].x, sim2->particles[index2].x, L);
      else
        r = md_distance_x(sim->particles[index].x, sim2->particles[index2].x);
      int index3 = (int)(r/incre);
      if (index3 >= stats->points)
        continue; //index3 = stats->points-1;
      stats->fxs[stats->N-1][index3] += 1.0/norm;
    }
#endif
  return 0;
}

int md_pimd_calc_pair_correlation_2d_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmax, md_num_t_x norm, md_num_t_x *box) {
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x incre = rmax/stats->points;
  int l, j;
  int pj;
  int index, index2;
  for (pj = 1; pj <= pimd->P; ++pj)
    for (l = 1; l <= pimd->N; ++l)
      for (j = 1; j <= pimd->N; ++j) {
        if (l == j)
          continue;
        index = MD_PIMD_INDEX_X(l, pj, pimd->P);
        index2 = MD_PIMD_INDEX_X(j, pj, pimd->P);
        if (sim->particles[index].x[0] < box[0] || sim->particles[index].x[0] > box[1] || sim->particles[index].x[1] < box[2] || sim->particles[index].x[1] > box[3])
          continue;
        if (sim->particles[index2].x[0] < box[0] || sim->particles[index2].x[0] > box[1] || sim->particles[index2].x[1] < box[2] || sim->particles[index2].x[1] > box[3])
          continue;
        md_num_t_x r = sqrt(pow(sim->particles[index].x[0]-sim->particles[index2].x[0], 2)+pow(sim->particles[index].x[1]-sim->particles[index2].x[1], 2));
        int index3 = (int)(r/incre);
        if (index3 >= stats->points)
          continue; //index3 = stats->points-1;
        stats->fxs[stats->N-1][index3] += 1.0/norm;
      }
  return 0;
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
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim2, context);
  if (status != 0)
    return status;
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
    status = clReleaseKernel(pimd->cpf2_kernel);
    if (status != 0)
      return status;
    pimd->cpf2_kernel = NULL;
  }
  int j;
  if (pimd->forces2 != NULL && pimd->N2 != pimd2->N) {
    status = clReleaseMemObject(pimd->forces2);
    if (status != 0)
      return status;
    pimd->forces2 = NULL;
  }
  cl_mem forces = NULL;
  if (pimd->forces2 != NULL)
    forces = pimd->forces2;
  else {
    forces = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*pimd->N*pimd2->N*MD_DIMENSION_X, NULL, &status);
    if (status != 0)
      return status;
    pimd->forces2 = forces;
    pimd->N2 = pimd2->N;
  }
  cl_kernel kernel = NULL;
  if (pimd->cpf2_kernel != NULL)
    kernel = pimd->cpf2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cpf2_kernel = kernel;
    strcpy(pimd->pf2kname, kname);
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &sim2->particles_mem);
  status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &forces);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd2->N);
  status |= clSetKernelArg(kernel, 7, sizeof(md_inter_params_t_x), &params);
  status |= clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (pimd->upf21_kernel != NULL)
    kernel2 = pimd->upf21_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_update_pair_force_2_1_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel2, 3, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel2, 4, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->upf21_kernel = kernel2;
  }
  status = clSetKernelArg(kernel2, 2, sizeof(int), &pimd2->N);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &forces);
  if (status != 0)
    return status;
  cl_kernel kernel3 = NULL;
  if (pimd->upf22_kernel != NULL)
    kernel3 = pimd->upf22_kernel;
  else {
    kernel3 = clCreateKernel(md_programs_x[plat], "md_pimd_update_pair_force_2_2_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel3, 3, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel3, 4, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->upf22_kernel = kernel3;
  }
  status = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &sim2->particles_mem);
  status |= clSetKernelArg(kernel3, 2, sizeof(int), &pimd2->N);
  status |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &forces);
  if (status != 0)
    return status;
  for (j = 1; j <= pimd->P; ++j) {
    int size[1];
    size[0] = pimd->N*pimd2->N;
    size_t local[1], global[1];
    status = md_get_work_size_x(kernel, device, 1, size, global, local);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 6, sizeof(int), &j);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = pimd->N;
    status = md_get_work_size_x(kernel2, device, 1, size, global, local);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 5, sizeof(int), &j);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = pimd2->N;
    status = md_get_work_size_x(kernel3, device, 1, size, global, local);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel3, 5, sizeof(int), &j);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
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
            sim2->particles[index2].f[i] -= sim->particles[index].m*f[i]/pimd->P/sim2->particles[index2].m;
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
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim2, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
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
    status = clReleaseKernel(pimd->cpe2_kernel);
    if (status != 0)
      return status;
    pimd->cpe2_kernel = NULL;
  }
  cl_kernel kernel = NULL;
  if (pimd->cpe2_kernel != NULL)
    kernel = pimd->cpe2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], kname, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cpe2_kernel = kernel;
    strcpy(pimd->pe2kname, kname);
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &sim2->particles_mem);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd2->N);
  status |= clSetKernelArg(kernel, 7, sizeof(md_inter_params_t_x), &params);
  status |= clSetKernelArg(kernel, 8, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    if (status != 0)
      return status;
    sim->add_kernel = kernel2;
  }
  md_num_t_x mult = 1.0;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->e_mem);
  status |= clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  if (status != 0)
    return status;
  int j;
  for (j = 1; j <= pimd->P; ++j) {
    int size[1];
    size[0] = pimd->N*pimd2->N;
    size_t local[1], global[1];
    status = md_get_work_size_x(kernel, device, 1, size, global, local);
    if (status != 0)
      return status;
    int group = global[0]/local[0];
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel, 6, sizeof(int), &j);
    status |= clSetKernelArg(kernel, 9, sizeof(md_num_t_x)*local[0], NULL);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel2, 2, sizeof(int), &group);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    status = clReleaseMemObject(out_mem);
    if (status != 0)
      return status;
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

int md_pimd_calc_pair_force_pa_x(md_pimd_t_x *pimd, md_pa_table_t_x *pa, int image) {
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int i, k;
  md_num_t_x r[MD_DIMENSION_X];
  md_num_t_x r2[MD_DIMENSION_X];
  md_num_t_x f[MD_DIMENSION_X];
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 < l; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r[i] = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
            else
              r[i] = sim->particles[index].x[i]-sim->particles[index2].x[i];
          }
          k = j+1;
          if (k == pimd->P+1)
            k = 1;
          index = MD_PIMD_INDEX_X(l, k, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, k, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r2[i] = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
            else
              r2[i] = sim->particles[index].x[i]-sim->particles[index2].x[i];
          }
          md_calc_pa_deri_1_x(pa, r, r2, f, (pa->r[1]-pa->r[0])/2.0);
          index = MD_PIMD_INDEX_X(l, j, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            sim->particles[index].f[i] -= f[i]/sim->particles[index].m/pimd->P;
            sim->particles[index2].f[i] += f[i]/sim->particles[index2].m/pimd->P;
          }
          md_calc_pa_deri_2_x(pa, r, r2, f, (pa->r[1]-pa->r[0])/2.0);
          k = j+1;
          if (k == pimd->P+1)
            k = 1;
          index = MD_PIMD_INDEX_X(l, k, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, k, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            sim->particles[index].f[i] -= f[i]/sim->particles[index].m/pimd->P;
            sim->particles[index2].f[i] += f[i]/sim->particles[index2].m/pimd->P;
          }
        }
  return 0;
}

int md_pimd_calc_pair_force_pa_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_pa_table_t_x *pa, int image) {
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int i, k;
  md_num_t_x r[MD_DIMENSION_X];
  md_num_t_x r2[MD_DIMENSION_X];
  md_num_t_x f[MD_DIMENSION_X];
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 <= pimd2->N; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r[i] = md_minimum_image_x(sim->particles[index].x[i]-sim2->particles[index2].x[i], L[i]);
            else
              r[i] = sim->particles[index].x[i]-sim2->particles[index2].x[i];
          }
          k = j+1;
          if (k == pimd->P+1)
            k = 1;
          index = MD_PIMD_INDEX_X(l, k, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, k, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r2[i] = md_minimum_image_x(sim->particles[index].x[i]-sim2->particles[index2].x[i], L[i]);
            else
              r2[i] = sim->particles[index].x[i]-sim2->particles[index2].x[i];
          }
          md_calc_pa_deri_1_x(pa, r, r2, f, (pa->r[1]-pa->r[0])/2.0);
          index = MD_PIMD_INDEX_X(l, j, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            sim->particles[index].f[i] -= f[i]/sim->particles[index].m/pimd->P;
            sim2->particles[index2].f[i] += f[i]/sim2->particles[index2].m/pimd->P;
          }
          md_calc_pa_deri_2_x(pa, r, r2, f, (pa->r[1]-pa->r[0])/2.0);
          k = j+1;
          if (k == pimd->P+1)
            k = 1;
          index = MD_PIMD_INDEX_X(l, k, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, k, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            sim->particles[index].f[i] -= f[i]/sim->particles[index].m/pimd->P;
            sim2->particles[index2].f[i] += f[i]/sim2->particles[index2].m/pimd->P;
          }
        }
  return 0;
}

int md_pimd_calc_pair_energy_pa_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_pa_table_t_x *pa, int image) {
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int i, k;
  md_num_t_x r[MD_DIMENSION_X];
  md_num_t_x r2[MD_DIMENSION_X];
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 < l; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r[i] = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
            else
              r[i] = sim->particles[index].x[i]-sim->particles[index2].x[i];
          }
          k = j+1;
          if (k == pimd->P+1)
            k = 1;
          index = MD_PIMD_INDEX_X(l, k, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, k, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r2[i] = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
            else
              r2[i] = sim->particles[index].x[i]-sim->particles[index2].x[i];
          }
          stats->es[stats->N-1] += md_calc_pa_pair_energy_x(pa, r, r2)/pimd->P;
        }
  return 0;
}

int md_pimd_calc_pair_energy_pa_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_stats_t_x *stats, md_pa_table_t_x *pa, int image) {
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int i, k;
  md_num_t_x r[MD_DIMENSION_X];
  md_num_t_x r2[MD_DIMENSION_X];
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 <= pimd2->N; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r[i] = md_minimum_image_x(sim->particles[index].x[i]-sim2->particles[index2].x[i], L[i]);
            else
              r[i] = sim->particles[index].x[i]-sim2->particles[index2].x[i];
          }
          k = j+1;
          if (k == pimd->P+1)
            k = 1;
          index = MD_PIMD_INDEX_X(l, k, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, k, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r2[i] = md_minimum_image_x(sim->particles[index].x[i]-sim2->particles[index2].x[i], L[i]);
            else
              r2[i] = sim->particles[index].x[i]-sim2->particles[index2].x[i];
          }
          stats->es[stats->N-1] += md_calc_pa_pair_energy_x(pa, r, r2)/pimd->P;
        }
  return 0;
}

int md_pimd_calc_pair_pa_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_stats_t_x *stats, md_pa_table_t_x *pa, int image) {
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int i, k;
  md_num_t_x r[MD_DIMENSION_X];
  md_num_t_x r2[MD_DIMENSION_X];
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 <= pimd2->N; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r[i] = md_minimum_image_x(sim->particles[index].x[i]-sim2->particles[index2].x[i], L[i]);
            else
              r[i] = sim->particles[index].x[i]-sim2->particles[index2].x[i];
          }
          k = j+1;
          if (k == pimd->P+1)
            k = 1;
          index = MD_PIMD_INDEX_X(l, k, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, k, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r2[i] = md_minimum_image_x(sim->particles[index].x[i]-sim2->particles[index2].x[i], L[i]);
            else
              r2[i] = sim->particles[index].x[i]-sim2->particles[index2].x[i];
          }
          stats->es[stats->N-1] += md_calc_pa_pair_U_x(pa, r, r2)/pimd->P;
        }
  return 0;
}

int md_pimd_calc_pair_pa_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_pa_table_t_x *pa, int image) {
  int l, l2, j;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int i, k;
  md_num_t_x r[MD_DIMENSION_X];
  md_num_t_x r2[MD_DIMENSION_X];
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l)
        for (l2 = 1; l2 < l; ++l2) {
          int index = MD_PIMD_INDEX_X(l, j, pimd->P);
          int index2 = MD_PIMD_INDEX_X(l2, j, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r[i] = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
            else
              r[i] = sim->particles[index].x[i]-sim->particles[index2].x[i];
          }
          k = j+1;
          if (k == pimd->P+1)
            k = 1;
          index = MD_PIMD_INDEX_X(l, k, pimd->P);
          index2 = MD_PIMD_INDEX_X(l2, k, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if (image)
              r2[i] = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
            else
              r2[i] = sim->particles[index].x[i]-sim->particles[index2].x[i];
          }
          stats->es[stats->N-1] += md_calc_pa_pair_U_x(pa, r, r2)/pimd->P;
        }
  return 0;
}

int md_pimd_calc_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats) {
  pimd->sim = md_simulation_sync_host_x(pimd->sim, 1);
  if (pimd->sim == NULL)
    return -1;
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
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
  return 0;
}

int md_pimd_calc_centroid_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats, int fc) {
  pimd->sim = md_simulation_sync_host_x(pimd->sim, 1);
  if (pimd->sim == NULL)
    return -1;
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  md_num_t_x *centroid = (md_num_t_x *)malloc(MD_DIMENSION_X*sizeof(md_num_t_x)*pimd->N);
  if (centroid == NULL)
    return -1;
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
  return 0;
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

int md_pimd_fill_ENk2_x(md_pimd_t_x *pimd, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
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
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk2_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->pair_in_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->pcount_in);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->fillE2_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->pcount_in;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel, 4, sizeof(int), &image);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  pimd->queue = queue;
#else
  int l, j;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= l; ++j)
      pimd->ENk2[l-1][j-1] = md_pimd_ENk2_x(pimd, l, j, image);
#endif
  return 0;
}

int md_pimd_fprint_ENk2_x(FILE *out, md_pimd_t_x *pimd) {
  pimd = md_pimd_sync_host_x(pimd, 1);
  if (pimd == NULL)
    return -1;
  fprintf(out, "ENk2 %d\n", pimd->N);
  int l, j;
  for (l = 1; l <= pimd->N; ++l) {
    for (j = 1; j <= l; ++j)
      fprintf(out, "%f ", pimd->ENk2[l-1][j-1]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

int md_pimd_calc_vi_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
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
  cl_mem res_mem = NULL;
  if (pimd->eVBN_mem != NULL)
    res_mem = pimd->eVBN_mem;
  else {
    res_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*(pimd->N+1), NULL, &status);
    if (status != 0)
      return status;
    pimd->eVBN_mem = res_mem;
  }
  status = clEnqueueWriteBuffer(queue, pimd->eVBN_mem, CL_FALSE, 0, sizeof(md_num_t_x), &md_VBN_0_x, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->filleV2_kernel != NULL)
    kernel = pimd->filleV2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_VBN2_energy_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pimd->ENk_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk2_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->VBN_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 6, sizeof(int), &pimd->N);
    if (status != 0)
      return status;
    pimd->filleV2_kernel = kernel;
  }
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &res_mem);
  status |= clSetKernelArg(kernel, 8, sizeof(md_num_t_x), &pimd->vi);
  status |= clSetKernelArg(kernel, 9, sizeof(md_num_t_x), &pimd->beta);
  status |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &pimd->minE_mem);
  if (status != 0)
    return status;
  cl_kernel kernel2 = NULL;
  if (pimd->addeV_kernel != NULL)
    kernel2 = pimd->addeV_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_add_eVBN_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &pimd->VBN_mem);
    if (status != 0)
      return status;
    pimd->addeV_kernel = kernel2;
  }
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &pimd->eVBN_mem);
  status |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), &pimd->minE_mem);
  status |= clSetKernelArg(kernel2, 6, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  int N2;
  for (N2 = 1; N2 <= pimd->N; ++N2) {
    int size[1];
    size[0] = N2;
    size_t local[1], global[1];
    status = md_get_work_size_x(kernel, device, 1, size, global, local);
    if (status != 0)
      return status;
    int group = global[0]/local[0];
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel, 7, sizeof(int), &N2);
    status |= clSetKernelArg(kernel, 11, sizeof(md_num_t_x)*local[0], NULL);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    size[0] = 1;
    local[0] = 1;
    global[0] = 1;
    status = clSetKernelArg(kernel2, 2, sizeof(cl_mem), &out_mem);
    status |= clSetKernelArg(kernel2, 4, sizeof(int), &group);
    status |= clSetKernelArg(kernel2, 5, sizeof(int), &N2);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
    status = clReleaseMemObject(out_mem);
    if (status != 0)
      return status;
  }
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel3 = NULL;
  if (pimd->addeVst2_kernel != NULL)
    kernel3 = pimd->addeVst2_kernel;
  else {
    kernel3 = clCreateKernel(md_programs_x[plat], "md_pimd_add_eVBN2_stats_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel3, 2, sizeof(int), &pimd->N);
    if (status != 0)
      return status;
    pimd->addeVst2_kernel = kernel3;
  }
  status = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &stats->e_mem);
  status |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &pimd->eVBN_mem);
  status |= clSetKernelArg(kernel3, 3, sizeof(md_num_t_x), &pimd->beta);
  if (status != 0)
    return status;
  size_t local[1], global[1];
  local[0] = 1;
  global[0] = 1;
  status = clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  cl_kernel kernel4 = NULL;
  if (pimd->cvire_kernel != NULL)
    kernel4 = pimd->cvire_kernel;
  else {
    kernel4 = clCreateKernel(md_programs_x[plat], "md_pimd_calc_virial_energy_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel4, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel4, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel4, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cvire_kernel = kernel4;
  }
  int size[1];
  size[0] = pimd->N*pimd->P;
  status = md_get_work_size_x(kernel4, device, 1, size, global, local);
  if (status != 0)
    return status;
  int group = global[0]/local[0];
  cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(md_num_t_x)*group, NULL, &status);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel4, 1, sizeof(cl_mem), &out_mem);
  status |= clSetKernelArg(kernel4, 4, sizeof(md_num_t_x)*local[0], NULL);
  status |= clSetKernelArg(kernel4, 5, sizeof(int), &image);
  status |= clSetKernelArg(kernel4, 6, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel4, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  kernel2 = NULL;
  if (sim->add_kernel != NULL)
    kernel2 = sim->add_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_add_stats_kx", &status);
    if (status != 0)
      return status;
    sim->add_kernel = kernel2;
  }
  size[0] = 1;
  local[0] = 1;
  global[0] = 1;
  md_num_t_x mult = 1.0/2.0;
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &stats->e_mem);
  status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &out_mem);
  status |= clSetKernelArg(kernel2, 2, sizeof(int), &group);
  status |= clSetKernelArg(kernel2, 3, sizeof(md_num_t_x), &mult);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  pimd->queue = queue;
  sim->queue = queue;
  stats->queue = queue;
  status = clReleaseMemObject(out_mem);
  if (status != 0)
    return status;
#else
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  if (res == NULL)
    return -1;
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
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (j = 2; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (pimd->pbc)
          res2 += -((sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i])-(sim->particles[index2].x[i]/*+pimd->pbW[MD_DIMENSION_X*index2+i]*L[i]*/))*sim->particles[index].f[i]*sim->particles[index].m;
        else if (image)
          res2 += -md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i])*sim->particles[index].f[i]*sim->particles[index].m;
        else
          res2 += -(sim->particles[index].x[i]-sim->particles[index2].x[i])*sim->particles[index].f[i]*sim->particles[index].m;
      }
    }
  stats->es[stats->N-1] += res2/2.0;
#endif
  return 0;
}

int md_pimd_polymer_periodic_boundary_x(md_pimd_t_x *pimd) {
  pimd->sim = md_simulation_sync_host_x(pimd->sim, 0);
  if (pimd->sim == NULL)
    return -1;
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
  return 0;
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
    if (pimd->pbc) {
      if (index == index3)
        d = (sim->particles[index].x[i]+pimd->pbW[index*MD_DIMENSION_X+i]*L[i])-(sim->particles[index2].x[i]+pimd->pbW[index2*MD_DIMENSION_X+i]*L[i]);
      else
        d = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i]);
      int c = 0;
      if (index != index3) {
        if (pimd->sim->particles[index].x[i]-pimd->sim->particles[index2].x[i] > L[i]/2) {
          pimd->sim->particles[index].x[i] -= L[i];
          c = 1;
        }
        else if (pimd->sim->particles[index].x[i]-pimd->sim->particles[index2].x[i] < -L[i]/2) {
          c = -1;
          pimd->sim->particles[index].x[i] += L[i];
        }
      }
      if (index == index3)
        d2 = (sim->particles[index].x[i]+pimd->pbW[index*MD_DIMENSION_X+i]*L[i])-(sim->particles[index3].x[i]/*+pimd->pbW[index3*MD_DIMENSION_X+i]*L[i]*/);
      else
        d2 = (sim->particles[index].x[i]+pimd->pbW[index3*MD_DIMENSION_X+i]*L[i])-(sim->particles[index3].x[i]/*+pimd->pbW[index3*MD_DIMENSION_X+i]*L[i]*/); //d2 = md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index3].x[i], L[i]);
      if (index != index3)
        pimd->sim->particles[index].x[i] += c*L[i];
    }
    else if (image) {
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

int md_pimd_fast_fill_ENk2_x(md_pimd_t_x *pimd, int image) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(pimd->sim, context);
  if (status != 0)
    return status;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
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
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pimd->ENk2_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->ffillE21_kernel = kernel;
  }
  int size[1];
  size[0] = pimd->N;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clSetKernelArg(kernel, 4, sizeof(int), &image);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel, 7, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  int j;
  cl_kernel kernel2 = NULL;
  if (pimd->ffillE22_kernel != NULL)
    kernel2 = pimd->ffillE22_kernel;
  else {
    kernel2 = clCreateKernel(md_programs_x[plat], "md_pimd_fast_fill_ENk2_2_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &pimd->ENk2_mem);
    status |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &pimd->Eindices_mem);
    status |= clSetKernelArg(kernel2, 4, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel2, 6, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->ffillE22_kernel = kernel2;
  }
  status = clSetKernelArg(kernel2, 5, sizeof(int), &image);
  status |= clSetKernelArg(kernel2, 7, sizeof(md_num_t_x), &pimd->omegaP);
  status |= clSetKernelArg(kernel2, 8, sizeof(nd_rect_t_x), &rect2);
  if (status != 0)
    return status;
  for (j = 2; j <= pimd->N; ++j) {
    size[0] = pimd->N-j+1;
    status = md_get_work_size_x(kernel2, device, 1, size, global, local);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel2, 3, sizeof(int), &j);
    if (status != 0)
      return status;
    status = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, global, local, 0, NULL, NULL);
    if (status != 0)
      return status;
    status = md_update_queue_x(queue);
    if (status != 0)
      return status;
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
      //index = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      //index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      //index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+2, pimd->P, pimd->P);
      index3 = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      pimd->ENk2[l-1][j-1] = pimd->ENk2[l-1][j-2]-md_pimd_fast_ENk2_x(pimd, index, index2, index3, image);
      //index = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      //index2 = MD_PIMD_INDEX_X(l-j+1, pimd->P, pimd->P);
      //index3 = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      index = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+2, pimd->P, pimd->P);
      index3 = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      pimd->ENk2[l-1][j-1] += md_pimd_fast_ENk2_x(pimd, index, index2, index3, image);
      //index = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      //index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      //index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+1, pimd->P, pimd->P);
      index3 = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      pimd->ENk2[l-1][j-1] += md_pimd_fast_ENk2_x(pimd, index, index2, index3, image);
    }
#endif
  return 0;
}

int md_reset_stats_x(md_stats_t_x *stats, md_num_t_x e) {
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  stats->es[0] = e;
  return 0;
}

int md_reset_stats_f_x(md_stats_t_x *stats, md_num_t_x e) {
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  int i;
  for (i = 0; i < stats->points; ++i)
    stats->fxs[0][i] = e;
  return 0;
}

int md_stats_add_to_x(md_stats_t_x *stats, md_stats_t_x *stats2) {
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  stats2 = md_stats_sync_host_x(stats2);
  if (stats2 == NULL)
    return -1;
  stats->es[stats->N-1] += stats2->es[0];
  return 0;
}

int md_stats_add_to_f_x(md_stats_t_x *stats, md_stats_t_x *stats2) {
  if (stats->points != stats2->points)
    return -1;
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  stats2 = md_stats_sync_host_x(stats2);
  if (stats2 == NULL)
    return -1;
  int i;
  for (i = 0; i < stats->points; ++i)
    stats->fxs[stats->N-1][i] += stats2->fxs[0][i];
  return 0;
}

int md_stats_copy_to_x(md_stats_t_x *stats, md_stats_t_x *stats2) {
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  stats2 = md_stats_sync_host_x(stats2);
  if (stats2 == NULL)
    return -1;
  stats->es[0] = stats2->es[0];
  return 0;
}

int md_stats_copy_to_f_x(md_stats_t_x *stats, md_stats_t_x *stats2) {
  if (stats->points != stats2->points)
    return -1;
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  stats2 = md_stats_sync_host_x(stats2);
  if (stats2 == NULL)
    return -1;
  int i;
  for (i = 0; i < stats->points; ++i)
    stats->fxs[0][i] = stats2->fxs[0][i];
  return 0;
}

int md_stats_mult_to_x(md_stats_t_x *stats, md_stats_t_x *stats2) {
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  stats2 = md_stats_sync_host_x(stats2);
  if (stats2 == NULL)
    return -1;
  stats->es[0] *= stats2->es[0];
  return 0;
}

int md_stats_mult_to_f_x(md_stats_t_x *stats, md_stats_t_x *stats2) {
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  stats2 = md_stats_sync_host_x(stats2);
  if (stats2 == NULL)
    return -1;
  int i;
  for (i = 0; i < stats->points; ++i)
    stats->fxs[0][i] *= stats2->es[0];
  return 0;
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

int md_pimd_calc_vi_sign_x(md_pimd_t_x *pimd, md_stats_t_x *e_s, md_stats_t_x *s_s, md_num_t_x vi2) {
  pimd = md_pimd_sync_host_x(pimd, 1);
  if (pimd == NULL)
    return -1;
  e_s = md_stats_sync_host_x(e_s);
  if (e_s == NULL)
    return -1;
  s_s = md_stats_sync_host_x(s_s);
  if (s_s == NULL)
    return -1;
  md_num_t_x *VBN2 = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  md_num_t_x *sign = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  if (VBN2 == NULL || sign == NULL)
    return -1;
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
  return 0;
}

void md_calc_uniform_magnet_x(md_simulation_t_x *sim, int index, int index2, md_num_t_x *res, md_num_t_x B) {
  if (MD_DIMENSION_X < 2)
    return;
  res[0] = 0.5*B*(-sim->particles[index].x[1]-sim->particles[index2].x[1]);
  res[1] = 0.5*B*(sim->particles[index].x[0]+sim->particles[index2].x[0]);
  int i;
  for (i = 2; i < MD_DIMENSION_X; ++i)
    res[i] = 0;
}

md_num_t_x md_pimd_ENk3_x(md_pimd_t_x *pimd, int N2, int k, int image, md_num_t_x B) {
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, j, i;
  int index, index2;
  md_num_t_x res = 0;
  md_num_t_x mage[MD_DIMENSION_X];
  for (l = N2-k+1; l <= N2; ++l)
    for (j = 1; j <= pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = md_pimd_next_index_x(l, j, N2, k, pimd->P);
      md_calc_uniform_magnet_x(sim, index, index2, mage, B);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (image)
          res += mage[i]*md_minimum_image_x(sim->particles[index2].x[i]-sim->particles[index].x[i], L[i]);
        else
          res += mage[i]*(sim->particles[index2].x[i]-sim->particles[index].x[i]);
      }
    }
  return res;
}

int md_pimd_fill_ENk3_x(md_pimd_t_x *pimd, int image, md_num_t_x B) {
  pimd->sim = md_simulation_sync_host_x(pimd->sim, 1);
  if (pimd->sim == NULL)
    return -1;
  pimd = md_pimd_sync_host_x(pimd, 0);
  if (pimd == NULL)
    return -1;
  int l, j;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= l; ++j)
      pimd->ENk3[l-1][j-1] = md_pimd_ENk3_x(pimd, l, j, image, B);
  return 0;
}

int md_pimd_fprint_ENk3_x(FILE *out, md_pimd_t_x *pimd) {
  pimd = md_pimd_sync_host_x(pimd, 1);
  if (pimd == NULL)
    return -1;
  fprintf(out, "ENk3 %d\n", pimd->N);
  int l, j;
  for (l = 1; l <= pimd->N; ++l) {
    for (j = 1; j <= l; ++j)
      fprintf(out, "%f ", pimd->ENk3[l-1][j-1]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

int md_pimd_calc_magnet_sign_x(md_pimd_t_x *pimd, md_stats_t_x *e_s, md_stats_t_x *s_s, md_stats_t_x *ie_s, md_stats_t_x *is_s) {
  pimd = md_pimd_sync_host_x(pimd, 1);
  if (pimd == NULL)
    return -1;
  e_s = md_stats_sync_host_x(e_s);
  if (e_s == NULL)
    return -1;
  s_s = md_stats_sync_host_x(s_s);
  if (s_s == NULL)
    return -1;
  ie_s = md_stats_sync_host_x(ie_s);
  if (ie_s == NULL)
    return -1;
  is_s = md_stats_sync_host_x(is_s);
  if (is_s == NULL)
    return -1;
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  if (res == NULL)
    return -1;
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
      sum += (res[N2-k]-pimd->ENk3[N2-1][k-1])*md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
    }
    res[N2] = sum/sum2;
  }
  md_num_t_x s = cos(res[pimd->N]/2.0);
  e_s->es[0] *= s;
  s_s->es[0] *= s;
  s = sin(res[pimd->N]/2.0);
  ie_s->es[0] *= s;
  is_s->es[0] *= s;
  free(res);
  return 0;
}

void md_cross_product_x(md_num_t_x *a, md_num_t_x *b, md_num_t_x *res) {
  if (MD_DIMENSION_X == 2) {
    res[0] += a[0]*b[1]-a[1]*b[0];
  }
  else if (MD_DIMENSION_X == 3) {
    res[0] += a[1]*b[2]-a[2]*b[1];
    res[1] += a[2]*b[0]-a[0]*b[2];
    res[2] += a[0]*b[1]-a[1]*b[0];
  }
}

md_num_t_x md_cross_product_z_x(md_num_t_x *a, md_num_t_x *b) {
  if (MD_DIMENSION_X < 2)
    return 0;
  return a[0]*b[1]-a[1]*b[0];
}

md_num_t_x md_pimd_ENk3b_x(md_pimd_t_x *pimd, int N2, int k, int MD_UNUSED_X(image), md_num_t_x B) {
  md_simulation_t_x *sim = pimd->sim;
  int l, j;
  int index, index2;
  md_num_t_x res = 0;
  /*md_num_t_x mage[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    mage[i] = 0;*/
  for (l = N2-k+1; l <= N2; ++l)
    for (j = 1; j <= pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = md_pimd_next_index_x(l, j, N2, k, pimd->P);
      res += md_cross_product_z_x(sim->particles[index].x, sim->particles[index2].x);
      //md_cross_product_x(sim->particles[index].x, sim->particles[index2].x, mage);
    }
  //for (i = 0; i < MD_DIMENSION_X; ++i)
    //res += mage[i]*mage[i];
  return B*sqrt(res*res)/2.0;
}

int md_pimd_fill_ENk3b_x(md_pimd_t_x *pimd, int image, md_num_t_x B) {
  pimd->sim = md_simulation_sync_host_x(pimd->sim, 1);
  if (pimd->sim == NULL)
    return -1;
  pimd = md_pimd_sync_host_x(pimd, 0);
  if (pimd == NULL)
    return -1;
  int l, j;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= l; ++j)
      pimd->ENk3[l-1][j-1] = md_pimd_ENk3b_x(pimd, l, j, image, B);
  return 0;
}

int md_pimd_calc_2d_pair_correlation_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x rmax, md_num_t_x norm, int pi, int image) {
  if (MD_DIMENSION_X < 2)
    return -1;
  pimd->sim = md_simulation_sync_host_x(pimd->sim, 1);
  if (pimd->sim == NULL)
    return -1;
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, j;
  int index, index2;
  int n = (int)sqrt(stats->points);
  md_num_t_x incre = rmax/n;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd->N; ++j) {
      if (l == j)
        continue;
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(j, pi, pimd->P);
      int i1, i2;
      if (image) {
        i1 = (int)(fabs(md_minimum_image_x(pimd->sim->particles[index].x[0]-pimd->sim->particles[index2].x[0], L[0]))/incre);
        i2 = (int)(fabs(md_minimum_image_x(pimd->sim->particles[index].x[1]-pimd->sim->particles[index2].x[1], L[1]))/incre);
      }
      else {
        i1 = (int)(fabs((pimd->sim->particles[index].x[0]-pimd->sim->particles[index2].x[0]))/incre);
        i2 = (int)(fabs((pimd->sim->particles[index].x[1]-pimd->sim->particles[index2].x[1]))/incre);
      }
      int idx = i1*n+i2;
      if (idx < 0)
        idx = 0;
      if (idx >= stats->points)
        idx = stats->points-1;
      stats->fxs[stats->N-1][idx] += 1.0/norm;
    }
  return 0;
}

int md_pimd_calc_2d_pair_correlation_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_stats_t_x *stats, md_num_t_x rmax, md_num_t_x norm, int pi, int image) {
  if (MD_DIMENSION_X < 2)
    return -1;
  pimd->sim = md_simulation_sync_host_x(pimd->sim, 1);
  if (pimd->sim == NULL)
    return -1;
  pimd2->sim = md_simulation_sync_host_x(pimd2->sim, 1);
  if (pimd2->sim == NULL)
    return -1;
  stats = md_stats_sync_host_x(stats);
  if (stats == NULL)
    return -1;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  int l, j;
  int index, index2;
  int n = (int)sqrt(stats->points);
  md_num_t_x incre = rmax/n;
  for (l = 1; l <= pimd->N; ++l)
    for (j = 1; j <= pimd2->N; ++j) {
      index = MD_PIMD_INDEX_X(l, 1, pimd->P);
      index2 = MD_PIMD_INDEX_X(j, pi, pimd2->P);
      int i1, i2;
      if (image) {
        i1 = (int)(fabs(md_minimum_image_x(pimd->sim->particles[index].x[0]-pimd2->sim->particles[index2].x[0], L[0]))/incre);
        i2 = (int)(fabs(md_minimum_image_x(pimd->sim->particles[index].x[1]-pimd2->sim->particles[index2].x[1], L[1]))/incre);
      }
      else {
        i1 = (int)(fabs((pimd->sim->particles[index].x[0]-pimd2->sim->particles[index2].x[0]))/incre);
        i2 = (int)(fabs((pimd->sim->particles[index].x[1]-pimd2->sim->particles[index2].x[1]))/incre);
      }
      int idx = i1*n+i2;
      if (idx < 0)
        idx = 0;
      if (idx >= stats->points)
        idx = stats->points-1;
      stats->fxs[stats->N-1][idx] += 1.0/norm;
    }
  return 0;
}

int md_pimd_calc_Sk_structure_x(md_pimd_t_x *pimd, md_stats_t_x *stats, md_num_t_x q0, md_num_t_x qincre, int pi) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->cSk_kernel != NULL)
    kernel = pimd->cSk_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_Sk_structure_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cSk_kernel = kernel;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &stats->fx_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  status |= clSetKernelArg(kernel, 5, sizeof(md_num_t_x), &q0);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &qincre);
  status |= clSetKernelArg(kernel, 7, sizeof(int), &pi);
  if (status != 0)
    return status;
  int size[1];
  size[0] = stats->points;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  sim->queue = queue;
  stats->queue = queue;
#else
  int l, i, j;
  for (j = 0; j < stats->points; ++j) {
    md_num_t_x q = q0+j*qincre;
    md_num_t_x res = 0;
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      md_num_t_x sum = 0;
      md_num_t_x sum2 = 0;
      int index;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
        sum += cos(q*pimd->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, pi, pimd->P);
        sum2 += cos(q*pimd->sim->particles[index].x[i]);
      }
      res += sum*sum2/pimd->N;
      sum = 0;
      sum2 = 0;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
        sum += sin(q*pimd->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, pi, pimd->P);
        sum2 += sin(q*pimd->sim->particles[index].x[i]);
      }
      res += sum*sum2/pimd->N;
    }
    res /= MD_DIMENSION_X;
    stats->fxs[stats->N-1][j] += res;
  }
#endif
  return 0;
}

int md_pimd_calc_Sk_structure_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_stats_t_x *stats, md_num_t_x q0, md_num_t_x qincre, int pi) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim2, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->cSk2_kernel != NULL)
    kernel = pimd->cSk2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_Sk_structure_2_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cSk2_kernel = kernel;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &stats->fx_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  status |= clSetKernelArg(kernel, 5, sizeof(md_num_t_x), &q0);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &qincre);
  status |= clSetKernelArg(kernel, 7, sizeof(int), &pi);
  status |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &sim2->particles_mem);
  status |= clSetKernelArg(kernel, 9, sizeof(int), &pimd2->N);
  status |= clSetKernelArg(kernel, 10, sizeof(int), &pimd2->P);
  if (status != 0)
    return status;
  int size[1];
  size[0] = stats->points;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  sim->queue = queue;
  sim2->queue = queue;
  stats->queue = queue;
#else
  int l, i, j;
  for (j = 0; j < stats->points; ++j) {
    md_num_t_x q = q0+j*qincre;
    md_num_t_x res = 0;
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      md_num_t_x sum = 0;
      md_num_t_x sum2 = 0;
      int index;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
        sum += cos(q*pimd->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, pi, pimd2->P);
        sum2 += cos(q*pimd2->sim->particles[index].x[i]);
      }
      res += sum*sum2/pimd->N;
      sum = 0;
      sum2 = 0;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
        sum += sin(q*pimd->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, pi, pimd2->P);
        sum2 += sin(q*pimd2->sim->particles[index].x[i]);
      }
      res += sum*sum2/pimd->N;
    }
    res /= MD_DIMENSION_X;
    stats->fxs[stats->N-1][j] += res;
  }
#endif
  return 0;
}

int md_pimd_calc_Sk_structure_full_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_stats_t_x *stats, md_num_t_x q0, md_num_t_x qincre, int pi) {
#ifdef MD_USE_OPENCL_X
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  int plat;
  cl_int status;
  status = md_get_command_queue_x(&plat, NULL, &context, &device, &queue);
  if (status != 0)
    return status;
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  status = md_pimd_to_context_x(pimd, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim, context);
  if (status != 0)
    return status;
  status = md_simulation_to_context_x(sim2, context);
  if (status != 0)
    return status;
  status = md_stats_to_context_x(stats, context);
  if (status != 0)
    return status;
  cl_kernel kernel = NULL;
  if (pimd->cSkf2_kernel != NULL)
    kernel = pimd->cSkf2_kernel;
  else {
    kernel = clCreateKernel(md_programs_x[plat], "md_pimd_calc_Sk_structure_full_2_kx", &status);
    if (status != 0)
      return status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sim->particles_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &pimd->N);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &pimd->P);
    if (status != 0)
      return status;
    pimd->cSkf2_kernel = kernel;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &stats->fx_mem);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &stats->points);
  status |= clSetKernelArg(kernel, 5, sizeof(md_num_t_x), &q0);
  status |= clSetKernelArg(kernel, 6, sizeof(md_num_t_x), &qincre);
  status |= clSetKernelArg(kernel, 7, sizeof(int), &pi);
  status |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &sim2->particles_mem);
  status |= clSetKernelArg(kernel, 9, sizeof(int), &pimd2->N);
  status |= clSetKernelArg(kernel, 10, sizeof(int), &pimd2->P);
  if (status != 0)
    return status;
  int size[1];
  size[0] = stats->points;
  size_t local[1], global[1];
  status = md_get_work_size_x(kernel, device, 1, size, global, local);
  if (status != 0)
    return status;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
  if (status != 0)
    return status;
  status = md_update_queue_x(queue);
  if (status != 0)
    return status;
  sim->queue = queue;
  sim2->queue = queue;
  stats->queue = queue;
#else
  int l, i, j;
  for (j = 0; j < stats->points; ++j) {
    md_num_t_x q = q0+j*qincre;
    md_num_t_x res = 0;
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      md_num_t_x sum = 0;
      md_num_t_x sum2 = 0;
      int index;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
        sum += cos(q*pimd->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, 1, pimd2->P);
        sum += cos(q*pimd2->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, pi, pimd->P);
        sum2 += cos(q*pimd->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, pi, pimd2->P);
        sum2 += cos(q*pimd2->sim->particles[index].x[i]);
      }
      res += sum*sum2/(pimd->N+pimd2->N);
      sum = 0;
      sum2 = 0;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
        sum += sin(q*pimd->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, 1, pimd2->P);
        sum += sin(q*pimd2->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, pi, pimd->P);
        sum2 += sin(q*pimd->sim->particles[index].x[i]);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, pi, pimd2->P);
        sum2 += sin(q*pimd2->sim->particles[index].x[i]);
      }
      res += sum*sum2/(pimd->N+pimd2->N);
    }
    res /= MD_DIMENSION_X;
    stats->fxs[stats->N-1][j] += res;
  }
#endif
  return 0;
}

int md_pimd_calc_sphere_Sk_structure_full_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_stats_t_x *stats, int pi, md_sk_sphere_t_x *sk_sphere) {
  int l, i, j;
  int N = MD_MIN_X(sk_sphere->N, stats->points);
  for (j = 0; j < N; ++j) {
    int k;
    for (k = 0; k < /*pimd->P*/1; ++k) {
      md_num_t_x res = 0;
      md_num_t_x sum = 0;
      md_num_t_x sum2 = 0;
      int index;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum += cos(tmp);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd2->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd2->sim->particles[index].x[i];
        sum += cos(tmp);
      }
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, (k+pi-1)%pimd->P+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum2 += cos(tmp);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, (k+pi-1)%pimd->P+1, pimd2->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd2->sim->particles[index].x[i];
        sum2 += cos(tmp);
      }
      res += sum*sum2/(pimd->N+pimd2->N);
      sum = 0;
      sum2 = 0;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum += sin(tmp);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd2->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd2->sim->particles[index].x[i];
        sum += sin(tmp);
      }
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, (k+pi-1)%pimd->P+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum2 += sin(tmp);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, (k+pi-1)%pimd->P+1, pimd2->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd2->sim->particles[index].x[i];
        sum2 += sin(tmp);
      }
      res += sum*sum2/(pimd->N+pimd2->N);
      stats->fxs[stats->N-1][j] += res/*/pimd->P*/;
    }
  }
  return 0;
}

int md_pimd_calc_sphere_Sk_structure_3_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, md_pimd_t_x *pimd3, md_stats_t_x *stats, int pi, md_sk_sphere_t_x *sk_sphere) {
  int l, i, j;
  int N = MD_MIN_X(sk_sphere->N, stats->points);
  for (j = 0; j < N; ++j) {
    int k;
    for (k = 0; k < /*pimd->P*/1; ++k) {
      md_num_t_x res = 0;
      md_num_t_x sum = 0;
      md_num_t_x sum2 = 0;
      int index;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum += cos(tmp);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd2->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd2->sim->particles[index].x[i];
        sum += cos(tmp);
      }
      for (l = 0; l < pimd3->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, (k+pi-1)%pimd->P+1, pimd3->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd3->sim->particles[index].x[i];
        sum2 += cos(tmp);
      }
      res += sum*sum2/sqrt((pimd->N+pimd2->N)*pimd3->N);
      sum = 0;
      sum2 = 0;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum += sin(tmp);
      }
      for (l = 0; l < pimd2->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd2->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd2->sim->particles[index].x[i];
        sum += sin(tmp);
      }
      for (l = 0; l < pimd3->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, (k+pi-1)%pimd->P+1, pimd3->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd3->sim->particles[index].x[i];
        sum2 += sin(tmp);
      }
      res += sum*sum2/sqrt((pimd->N+pimd2->N)*pimd3->N);
      stats->fxs[stats->N-1][j] += res/*/pimd->P*/;
    }
  }
  return 0;
}

int md_pimd_calc_sphere_Sk_structure_x(md_pimd_t_x *pimd, md_stats_t_x *stats, int pi, md_sk_sphere_t_x *sk_sphere) {
  int l, i, j;
  int N = MD_MIN_X(sk_sphere->N, stats->points);
  for (j = 0; j < N; ++j) {
    int k;
    for (k = 0; k < /*pimd->P*/1; ++k) {
      md_num_t_x res = 0;
      md_num_t_x sum = 0;
      md_num_t_x sum2 = 0;
      int index;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum += cos(tmp);
      }
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, (k+pi-1)%pimd->P+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum2 += cos(tmp);
      }
      res += sum*sum2/(pimd->N);
      sum = 0;
      sum2 = 0;
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, k+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum += sin(tmp);
      }
      for (l = 0; l < pimd->N; ++l) {
        index = MD_PIMD_INDEX_X(l+1, (k+pi-1)%pimd->P+1, pimd->P);
        md_num_t_x tmp = 0;
        for (i = 0; i < MD_DIMENSION_X; ++i)
          tmp += sk_sphere->qis[MD_DIMENSION_X*j+i]*pimd->sim->particles[index].x[i];
        sum2 += sin(tmp);
      }
      res += sum*sum2/(pimd->N);
      stats->fxs[stats->N-1][j] += res/*/pimd->P*/;
    }
  }
  return 0;
}

int md_pimd_calc_vi_virial_pressure_x(md_pimd_t_x *pimd, md_stats_t_x *stats, int image) {
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  if (res == NULL)
    return -1;
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
  stats->es[stats->N-1] += 2.0*res[pimd->N]/MD_DIMENSION_X;
  stats->es[stats->N-1] += pimd->N/(pimd->beta);
  free(res);
  md_num_t_x res2 = 0;
  int l, j, i;
  int index, index2;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (j = 2; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (pimd->pbc)
          res2 += -((sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i])-(sim->particles[index2].x[i]/*+pimd->pbW[MD_DIMENSION_X*index2+i]*L[i]*/))*sim->particles[index].f[i]*sim->particles[index].m;
        else if (image)
          res2 += -md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i])*sim->particles[index].f[i]*sim->particles[index].m;
        else
          res2 += -(sim->particles[index].x[i]-sim->particles[index2].x[i])*sim->particles[index].f[i]*sim->particles[index].m;
      }
    }
  stats->es[stats->N-1] += res2/MD_DIMENSION_X;
  res2 = 0;
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        res2 += (sim->particles[index].x[i]-L[i]/2.0)*sim->particles[index].f[i]*sim->particles[index].m;
    }
  stats->es[stats->N-1] += res2/(MD_DIMENSION_X);
  return 0;
}

int md_pimd_cut_connect_force_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, int image) {
  //if (pimd->N != pimd2->N)
    //return -1;
  int N = MD_MIN_X(pimd->N, pimd2->N);
  int l, i;
  int index, index2;
  md_num_t_x dE[MD_DIMENSION_X];
  md_simulation_t_x *sim = pimd->sim;
  md_simulation_t_x *sim2 = pimd2->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (l = 1; l <= N; ++l) {
    index = MD_PIMD_INDEX_X(l, pimd->cut_Pj, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, pimd2->cut_Pj, pimd2->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (image)
        dE[i] = sim->particles[index].m*pimd->omegaP*pimd->omegaP*md_minimum_image_x(sim->particles[index].x[i]-sim2->particles[index2].x[i], L[i]);
      else
        dE[i] = sim->particles[index].m*pimd->omegaP*pimd->omegaP*(sim->particles[index].x[i]-sim2->particles[index2].x[i]);
    }
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      sim->particles[index].f[i] -= dE[i]/sim->particles[index].m;
      sim2->particles[index2].f[i] += dE[i]/sim2->particles[index2].m;
    }
  }
  for (l = 1; l <= N; ++l) {
    index = MD_PIMD_INDEX_X(l, pimd->cut_Pj+1, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, pimd2->cut_Pj+1, pimd2->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (image)
        dE[i] = sim->particles[index].m*pimd->omegaP*pimd->omegaP*md_minimum_image_x(sim->particles[index].x[i]-sim2->particles[index2].x[i], L[i]);
      else
        dE[i] = sim->particles[index].m*pimd->omegaP*pimd->omegaP*(sim->particles[index].x[i]-sim2->particles[index2].x[i]);
    }
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      sim->particles[index].f[i] -= dE[i]/sim->particles[index].m;
      sim2->particles[index2].f[i] += dE[i]/sim2->particles[index2].m;
    }
  }
  return 0;
}

int md_pimd_connect_fbeads_force_x(md_simulation_t_x *sim, md_simulation_t_x *sim2, int Nb, md_num_t_x omega, int image) {
  if (sim->N*Nb != sim2->N)
    return -1;
  int l, i, j;
  md_num_t_x dE[MD_DIMENSION_X];
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (l = 0; l < sim->N; ++l) {
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (image)
        dE[i] = sim2->particles[l*Nb].m*omega*omega*md_minimum_image_x(sim->particles[l].x[i]-sim2->particles[l*Nb].x[i], L[i]);
      else
        dE[i] = sim2->particles[l*Nb].m*omega*omega*(sim->particles[l].x[i]-sim2->particles[l*Nb].x[i]);
    }
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      sim->particles[l].f[i] -= dE[i]/sim->particles[l].m;
      sim2->particles[l*Nb].f[i] += dE[i]/sim2->particles[l*Nb].m;
    }
    for (j = 1; j < Nb; ++j) {
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (image)
          dE[i] = sim2->particles[l*Nb+j].m*omega*omega*md_minimum_image_x(sim2->particles[l*Nb+j].x[i]-sim2->particles[l*Nb+j-1].x[i], L[i]);
        else
          dE[i] = sim2->particles[l*Nb+j].m*omega*omega*(sim2->particles[l*Nb+j].x[i]-sim2->particles[l*Nb+j-1].x[i]);
      }
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        sim2->particles[l*Nb+j].f[i] -= dE[i]/sim2->particles[l*Nb+j].m;
        sim2->particles[l*Nb+j-1].f[i] += dE[i]/sim2->particles[l*Nb+j-1].m;
      }
    }
  }
  return 0;
}

int md_pimd_init_fbeads_pos_x(md_simulation_t_x *sim, md_simulation_t_x *sim2, int Nb, md_num_t_x fluc) {
  if (sim->N*Nb != sim2->N)
    return -1;
  int l, i, j;
  for (l = 0; l < sim->N; ++l) {
    for (j = 0; j < Nb; ++j) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        sim2->particles[l*Nb+j].x[i] = sim->particles[l].x[i]+fluc*2*(md_random_uniform_x(0,1)-0.5);
    }
  }
  return 0;
}

int md_pimd_mc_move_beads_x(md_pimd_t_x *pimd, md_num_t_x *trans, int *par, md_num_t_x max_trans) {
  int l = rand()%pimd->N;
  *par = l;
  int j, i;
  int index;
  for (j = 1; j <= pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      trans[(j-1)*MD_DIMENSION_X+i] = md_random_uniform_x(-max_trans, max_trans);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].x[i] += trans[(j-1)*MD_DIMENSION_X+i];
  }
  return 0;
}

int md_pimd_mc_restore_move_beads_x(md_pimd_t_x *pimd, md_num_t_x *trans, int *par) {
  int l = *par;
  int j, i;
  int index;
  for (j = 1; j <= pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].x[i] -= trans[(j-1)*MD_DIMENSION_X+i];
  }
  return 0;
}

void md_pimd_gaussian_sample_x(md_num_t_x *r1, md_num_t_x *r2, md_num_t_x *rm, md_num_t_x sig) {
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    rm[i] = (r1[i]+r2[i])/2.0+sqrt(sig)*md_random_gaussian_x();
}

int md_pimd_mc_gaussian_sample_beads_x(md_pimd_t_x *pimd, int length, int *par, int *jstart, md_num_t_x *rec) {
  int l = rand()%pimd->N;
  *par = l;
  length = MD_MIN_X(length, pimd->P);
  int n = (int)log2(length-1);
  length = (int)(pow(2, n)+1);
  int js = rand()%(pimd->P-length+1);
  *jstart = js;
  int i, j;
  int index, index2;
  for (j = js; j < js+length; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rec[(j-js)*MD_DIMENSION_X+i] = pimd->sim->particles[index].x[i];
  }
  int *fixed = (int *)malloc(sizeof(int)*length);
  for (j = 0; j < length; ++j)
    fixed[j] = 0;
  fixed[0] = 1;
  fixed[length-1] = 1;
  int k;
  int r1;
  md_num_t_x sig = (pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m);
  for (k = 0; k < n; ++k) {
    r1 = 0;
    for (j = 1; j < length; ++j) {
      if (fixed[j] != 0) {
        index = MD_PIMD_INDEX_X(l+1, js+r1+1, pimd->P);
        index2 = MD_PIMD_INDEX_X(l+1, js+j+1, pimd->P);
        md_pimd_gaussian_sample_x(pimd->sim->particles[index].x, pimd->sim->particles[index2].x, pimd->sim->particles[(index+index2)/2].x, sig);
        fixed[(r1+j)/2] = 1;
        r1 = j;
      }
    }
    /*for (j = 0; j < length; ++j)
      printf("%d ", fixed[j]);
    printf("\n");*/
  }
  free(fixed);
  return 0;
}

int md_pimd_mc_restore_gaussian_sample_beads_x(md_pimd_t_x *pimd, int length, int *par, int *jstart, md_num_t_x *rec) {
  int l = *par;
  length = MD_MIN_X(length, pimd->P);
  int n = (int)log2(length-1);
  length = (int)(pow(2, n)+1);
  int js = *jstart;
  int i, j;
  int index;
  for (j = js; j < js+length; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].x[i] = rec[(j-js)*MD_DIMENSION_X+i];
  }
  return 0;
}

int md_pimd_mc_calc_connect_prob_x(md_pimd_t_x *pimd, int par, md_num_t_x *G) {
  md_num_t_x *tmpV = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(pimd->N+1));
  if (tmpV == NULL)
    return -1;
  tmpV[pimd->N] = 0;
  int l, j, u;
  int l2;
  if (par < pimd->N)
    l2 = par+1;
  else
    l2 = par-pimd->N+1;
  if (pimd->worm_index > 0 && l2 > pimd->worm_index)
      --l2;
  if (pimd->worm_index > 0)
    --pimd->N;
  tmpV[pimd->N] = 0;
  for (u = pimd->N; u >= /*l2*/1; --u) {
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
  if (pimd->worm_index > 0)
    ++pimd->N;
  if (par < pimd->N) {
    j = par+1;
    if (pimd->worm_index > 0 && j > pimd->worm_index)
      --j;
    if (pimd->worm_index > 0)
      --pimd->N;
    for (l = 1; l <= pimd->N; ++l) {
      if (j > l+1)
        G[l-1] = 0;
      else if (j == l+1)
        G[l-1] = 1-exp(-pimd->beta*(pimd->VBN[l]+tmpV[l]-pimd->VBN[pimd->N]));
      else {
        if (pimd->vi == 0 && l-j != 0)
          G[l-1] = 0;
        else if (pimd->vi == 0 && l-j == 0)
          G[l-1] = exp(-pimd->beta*(pimd->VBN[j-1]+pimd->ENk[l-1][l-j]+tmpV[l]-pimd->VBN[pimd->N]))/l;
        else
          G[l-1] = exp(-pimd->beta*(pimd->VBN[j-1]+pimd->ENk[l-1][l-j]+tmpV[l]-pimd->VBN[pimd->N])+(l-j)*log(pimd->vi))/l;
      }
    }
    if (pimd->worm_index > 0)
      ++pimd->N;
  }
  else {
    l = par-pimd->N+1;
    if (pimd->worm_index > 0 && l > pimd->worm_index)
      --l;
    if (pimd->worm_index > 0)
      --pimd->N;
    for (j = 1; j <= pimd->N; ++j) {
      if (j > l+1)
        G[j-1] = 0;
      else if (j == l+1)
        G[j-1] = 1-exp(-pimd->beta*(pimd->VBN[l]+tmpV[l]-pimd->VBN[pimd->N]));
      else {
        if (pimd->vi == 0 && l-j != 0)
          G[j-1] = 0;
        else if (pimd->vi == 0 && l-j == 0)
          G[j-1] = exp(-pimd->beta*(pimd->VBN[j-1]+pimd->ENk[l-1][l-j]+tmpV[l]-pimd->VBN[pimd->N]))/l;
        else
          G[j-1] = exp(-pimd->beta*(pimd->VBN[j-1]+pimd->ENk[l-1][l-j]+tmpV[l]-pimd->VBN[pimd->N])+(l-j)*log(pimd->vi))/l;
      }
    }
    if (pimd->worm_index > 0)
      ++pimd->N;
  }
  if (pimd->worm_index > 0) {
    /*G[pimd->worm_index-1] = 0;
    md_num_t_x sum = 0;
    for (j = 1; j <= pimd->N; ++j)
      sum += G[j-1];
    for (j = 1; j <= pimd->N; ++j)
      G[j-1] /= sum;*/
    for (j = pimd->N-1; j >= pimd->worm_index; --j)
      G[j] = G[j-1];
    G[pimd->worm_index-1] = 0;
  }
  free(tmpV);
  return 0;
}

int md_pimd_mc_gaussian_pbW_x(md_pimd_t_x *pimd, int *par, md_num_t_x *G, int *rec2, int W, md_num_t_x *rec) {
  int l = rand()%(pimd->N);
  *par = l;
  int index;
  index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    rec[i] = pimd->sim->particles[index].x[i];
    rec2[i] = pimd->pbW[index*MD_DIMENSION_X+i];
    //pimd->pbW[index*MD_DIMENSION_X+i] = (rand()%(2*W+1))-W;
  }
  i = rand()%MD_DIMENSION_X;
  pimd->pbW[index*MD_DIMENSION_X+i] = (rand()%(2*W+1))-W;
  md_pimd_mc_calc_connect_prob_x(pimd, l, G);
  /*int j;
  md_num_t_x sum = 0;
  md_num_t_x r = md_random_uniform_x(0, 1);
  for (j = 0; j < pimd->N; ++j) {
    sum += G[j];
    if (r <= sum)
      break;
  }
  if (j >= pimd->N)
    j = pimd->N-1;
  int index2, index3;
  index2 = MD_PIMD_INDEX_X(l+1, 2, pimd->P);
  index3 = MD_PIMD_INDEX_X(j+1, pimd->P, pimd->P);
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
    pimd->sim->particles[index3].x[i] += pimd->pbW[MD_DIMENSION_X*index3+i]*L[i];
  }
  md_num_t_x *pW = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(2*W+1));
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    md_num_t_x dis = 0;
    dis = (pimd->sim->particles[index].x[i]-pimd->sim->particles[index3].x[i])*(pimd->sim->particles[index].x[i]-pimd->sim->particles[index3].x[i]);
    for (j = -W; j <= W; ++j) {
      pimd->sim->particles[index].x[i] += j*L[i];
      md_num_t_x dis2 = 0;
      dis2 = (pimd->sim->particles[index].x[i]-pimd->sim->particles[index3].x[i])*(pimd->sim->particles[index].x[i]-pimd->sim->particles[index3].x[i]);
      pW[j+W] = exp(-0.5*pimd->sim->particles[index].m*pimd->omegaP*pimd->omegaP*(dis2-dis));
      pimd->sim->particles[index].x[i] -= j*L[i];
    }
    md_num_t_x sum = 0;
    for (j = -W; j <= W; ++j)
      sum += pW[j+W];
    for (j = -W; j <= W; ++j)
      pW[j+W] /= sum;
    sum = 0;
    md_num_t_x r = md_random_uniform_x(0, 1);
    for (j = -W; j <= W; ++j) {
      sum += pW[j+W];
      if (r <= sum)
        break;
    }
    if (j > W)
      j = W;
    pimd->pbW[index*MD_DIMENSION_X+i] = j;
  }
  free(pW);
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
    pimd->sim->particles[index3].x[i] -= pimd->pbW[MD_DIMENSION_X*index3+i]*L[i];
  }*/
  return 0;
}

int md_pimd_mc_restore_gaussian_pbW_x(md_pimd_t_x *pimd, int *par, int *rec2, md_num_t_x *rec) {
  int l = *par;
  int index;
  index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    pimd->sim->particles[index].x[i] = rec[i];
    pimd->pbW[index*MD_DIMENSION_X+i] = rec2[i];
  }
  return 0;
}

int md_pimd_mc_gaussian_move_heads_x(md_pimd_t_x *pimd, md_num_t_x *rec, int *par, md_num_t_x *G, int image, int *rec2) {
  int l = rand()%(2*pimd->N);
  if (pimd->mc_par >= 0)
    l = pimd->mc_par+pimd->N*(rand()%2);
  *par = l;
  int index;
  if (l < pimd->N)
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  else
    index = MD_PIMD_INDEX_X(l-pimd->N+1, pimd->P, pimd->P);
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    rec[i] = pimd->sim->particles[index].x[i];
    rec2[i] = pimd->pbW[index*MD_DIMENSION_X+i];
  }
  if (l+1 == pimd->worm_index || l-pimd->N+1 == pimd->worm_index)
    return 0;
  md_pimd_mc_calc_connect_prob_x(pimd, l, G);
  int j;
  md_num_t_x sum = 0;
  md_num_t_x r = md_random_uniform_x(0, 1);
  for (j = 0; j < pimd->N; ++j) {
    sum += G[j];
    if (r <= sum)
      break;
  }
  if (j >= pimd->N)
    j = pimd->N-1;
  if (j+1 == pimd->worm_index)
    return 0;
  int index2, index3;
  if (l < pimd->N) {
    index2 = MD_PIMD_INDEX_X(l+1, 2, pimd->P);
    index3 = MD_PIMD_INDEX_X(j+1, pimd->P, pimd->P);
  }
  else {
    index2 = MD_PIMD_INDEX_X(l-pimd->N+1, pimd->P-1, pimd->P);
    index3 = MD_PIMD_INDEX_X(j+1, 1, pimd->P);
  }
  int c[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    c[i] = 0;
  if (pimd->pbc) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    if (l+1 == j+1 || l-pimd->N+1 == j+1) {
      if (l >= pimd->N) {
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          if ((pimd->sim->particles[index3].x[i]+pimd->pbW[MD_DIMENSION_X*index3+i]*L[i])-(pimd->sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i]) > L[i]/2) {
            --pimd->pbW[MD_DIMENSION_X*index3+i];
          }
          else if ((pimd->sim->particles[index3].x[i]+pimd->pbW[MD_DIMENSION_X*index3+i]*L[i])-(pimd->sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i]) < -L[i]/2) {
            ++pimd->pbW[MD_DIMENSION_X*index3+i];
          }
        }
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
          pimd->sim->particles[index3].x[i] += pimd->pbW[MD_DIMENSION_X*index3+i]*L[i];
        }
      }
      else {
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
          pimd->sim->particles[index3].x[i] += (pimd->pbW[MD_DIMENSION_X*index3+i]-pimd->pbW[MD_DIMENSION_X*index+i])*L[i];
        }
      }
    }
    else {
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index].x[i] > L[i]/2) {
          pimd->sim->particles[index3].x[i] -= L[i];
          c[i] = 1;
        }
        else if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index].x[i] < -L[i]/2) {
          c[i] = -1;
          pimd->sim->particles[index3].x[i] += L[i];
        }
      }
      if (l >= pimd->N) {
        int index4 = MD_PIMD_INDEX_X(l-pimd->N+1, 1, pimd->P);
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
          pimd->sim->particles[index3].x[i] += pimd->pbW[MD_DIMENSION_X*index4+i]*L[i];
        }
      }
      else {
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
        }
      }
    }
  }
  else if (image) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (pimd->sim->particles[index2].x[i]-pimd->sim->particles[index3].x[i] > L[i]/2)
        pimd->sim->particles[index2].x[i] -= L[i];
      else if (pimd->sim->particles[index2].x[i]-pimd->sim->particles[index3].x[i] < -L[i]/2)
        pimd->sim->particles[index2].x[i] += L[i];
    }
  }
  md_num_t_x aj = 0.5;
  md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    md_num_t_x r2 = (pimd->sim->particles[index2].x[i]+pimd->sim->particles[index3].x[i])/2.0;
    pimd->sim->particles[index].x[i] = r2+sqrt(sig)*md_random_gaussian_x();
    if (l >= pimd->N) {
      //if (l+1 == j+1 || l-pimd->N+1 == j+1)
        pimd->pbW[MD_DIMENSION_X*index+i] = 0;
      //else
        //pimd->pbW[MD_DIMENSION_X*index+i] = pimd->pbW[MD_DIMENSION_X*index2+i];
    }
  }
  if (pimd->pbc) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    if (l+1 == j+1 || l-pimd->N+1 == j+1) {
      if (l >= pimd->N) {
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
          pimd->sim->particles[index3].x[i] -= pimd->pbW[MD_DIMENSION_X*index3+i]*L[i];
        }
      }
      else {
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
          pimd->sim->particles[index3].x[i] -= (pimd->pbW[MD_DIMENSION_X*index3+i]-pimd->pbW[MD_DIMENSION_X*index+i])*L[i];
        }
      }
    }
    else {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index3].x[i] += c[i]*L[i];
      if (l >= pimd->N) {
        int index4 = MD_PIMD_INDEX_X(l-pimd->N+1, 1, pimd->P);
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
          pimd->sim->particles[index3].x[i] -= pimd->pbW[MD_DIMENSION_X*index4+i]*L[i];
        }
      }
      else {
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
        }
      }
    }
    //int out = 0;
    if (l < pimd->N) {
      int count[MD_DIMENSION_X];
      index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        count[i] = md_periodic_image_count_x(pimd->sim->particles[index].x[i], L[i]);
      for (j = 0; j < pimd->P; ++j) {
        index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
        for (i = 0; i < MD_DIMENSION_X; ++i)
          pimd->sim->particles[index].x[i] += count[i]*L[i];
      }
      /*for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (pimd->sim->particles[index].x[i] < 0 || pimd->sim->particles[index].x[i] > L[i]) {
          out = 1;
          break;
        }
      }*/
    }
    /*if (out) {
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        pimd->sim->particles[index].x[i] = rec[i];
        pimd->pbW[index*MD_DIMENSION_X+i] = rec2[i];
      }
    }*/
  }
  return 0;
}

int md_pimd_mc_restore_gaussian_move_heads_x(md_pimd_t_x *pimd, md_num_t_x *rec, int *par, int *rec2) {
  int l = *par;
  int index;
  if (l < pimd->N)
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  else
    index = MD_PIMD_INDEX_X(l-pimd->N+1, pimd->P, pimd->P);
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    pimd->sim->particles[index].x[i] = rec[i];
    pimd->pbW[index*MD_DIMENSION_X+i] = rec2[i];
  }
  return 0;
}

int md_pimd_mc_calc_gaussian_heads_acc_x(md_pimd_t_x *pimd, int *par, md_num_t_x *G, int image, md_stats_t_x *acc) {
  int l = *par;
  int index;
  if (l < pimd->N)
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  else
    index = MD_PIMD_INDEX_X(l-pimd->N+1, pimd->P, pimd->P);
  if (l+1 == pimd->worm_index || l-pimd->N+1 == pimd->worm_index) {
    acc->es[0] = 1.0;
    return 0;
  }
  md_num_t_x *G2 = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->N);
  md_pimd_mc_calc_connect_prob_x(pimd, l, G2);
  int i, j;
  md_num_t_x sum = 0, sum2 = 0;
  int index2, index3;
  for (j = 0; j < pimd->N; ++j) {
    if (j+1 == pimd->worm_index)
      continue;
    if (l < pimd->N) {
      index2 = MD_PIMD_INDEX_X(l+1, 2, pimd->P);
      index3 = MD_PIMD_INDEX_X(j+1, pimd->P, pimd->P);
    }
    else {
      index2 = MD_PIMD_INDEX_X(l-pimd->N+1, pimd->P-1, pimd->P);
      index3 = MD_PIMD_INDEX_X(j+1, 1, pimd->P);
    }
    int c[MD_DIMENSION_X];
    for (i = 0; i < MD_DIMENSION_X; ++i)
      c[i] = 0;
    if (pimd->pbc) {
      md_num_t_x *L = NULL;
      if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
        nd_rect_t_x *rect = pimd->sim->box->box;
        L = rect->L;
      }
      if (l+1 == j+1 || l-pimd->N+1 == j+1) {
        if (l >= pimd->N) {
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            if ((pimd->sim->particles[index3].x[i]+pimd->pbW[MD_DIMENSION_X*index3+i]*L[i])-(pimd->sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i]) > L[i]/2) {
              --pimd->pbW[MD_DIMENSION_X*index3+i];
            }
            else if ((pimd->sim->particles[index3].x[i]+pimd->pbW[MD_DIMENSION_X*index3+i]*L[i])-(pimd->sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i]) < -L[i]/2) {
              ++pimd->pbW[MD_DIMENSION_X*index3+i];
            }
          }
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            pimd->sim->particles[index].x[i] += pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
            pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
            pimd->sim->particles[index3].x[i] += pimd->pbW[MD_DIMENSION_X*index3+i]*L[i];
          }
        }
        else {
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
            pimd->sim->particles[index3].x[i] += (pimd->pbW[MD_DIMENSION_X*index3+i]-pimd->pbW[MD_DIMENSION_X*index+i])*L[i];
          }
        }
      }
      else {
        for (i = 0; i < MD_DIMENSION_X; ++i) {
          if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index].x[i] > L[i]/2) {
            pimd->sim->particles[index3].x[i] -= L[i];
            c[i] = 1;
          }
          else if (pimd->sim->particles[index3].x[i]-pimd->sim->particles[index].x[i] < -L[i]/2) {
            c[i] = -1;
            pimd->sim->particles[index3].x[i] += L[i];
          }
        }
        if (l >= pimd->N) {
          int index4 = MD_PIMD_INDEX_X(l-pimd->N+1, 1, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            pimd->sim->particles[index].x[i] += pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
            pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
            pimd->sim->particles[index3].x[i] += pimd->pbW[MD_DIMENSION_X*index4+i]*L[i];
          }
        }
        else {
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
          }
        }
      }
    }
    else if (image) {
      md_num_t_x *L = NULL;
      if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
        nd_rect_t_x *rect = pimd->sim->box->box;
        L = rect->L;
      }
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (pimd->sim->particles[index2].x[i]-pimd->sim->particles[index3].x[i] > L[i]/2)
          pimd->sim->particles[index2].x[i] -= L[i];
        else if (pimd->sim->particles[index2].x[i]-pimd->sim->particles[index3].x[i] < -L[i]/2)
          pimd->sim->particles[index2].x[i] += L[i];
      }
    }
    md_num_t_x aj = 0.5;
    md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
    md_num_t_x dis = 0;
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      md_num_t_x r2 = (pimd->sim->particles[index2].x[i]+pimd->sim->particles[index3].x[i])/2.0;
      dis += (pimd->sim->particles[index].x[i]-r2)*(pimd->sim->particles[index].x[i]-r2);
    }
    md_num_t_x mult = (1.0/pow(2*M_PI*sig, MD_DIMENSION_X/2.0))*exp(-dis/(2*sig));
    sum += G2[j]*mult;
    sum2 += G[j]*mult;
    if (pimd->pbc) {
      md_num_t_x *L = NULL;
      if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
        nd_rect_t_x *rect = pimd->sim->box->box;
        L = rect->L;
      }
      if (l+1 == j+1 || l-pimd->N+1 == j+1) {
        if (l >= pimd->N) {
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            pimd->sim->particles[index].x[i] -= pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
            pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
            pimd->sim->particles[index3].x[i] -= pimd->pbW[MD_DIMENSION_X*index3+i]*L[i];
          }
        }
        else {
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
            pimd->sim->particles[index3].x[i] -= (pimd->pbW[MD_DIMENSION_X*index3+i]-pimd->pbW[MD_DIMENSION_X*index+i])*L[i];
          }
        }
      }
      else {
        for (i = 0; i < MD_DIMENSION_X; ++i)
          pimd->sim->particles[index3].x[i] += c[i]*L[i];
        if (l >= pimd->N) {
          int index4 = MD_PIMD_INDEX_X(l-pimd->N+1, 1, pimd->P);
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            pimd->sim->particles[index].x[i] -= pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
            pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
            pimd->sim->particles[index3].x[i] -= pimd->pbW[MD_DIMENSION_X*index4+i]*L[i];
          }
        }
        else {
          for (i = 0; i < MD_DIMENSION_X; ++i) {
            pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
          }
        }
      }
    }
  }
  acc->es[0] = sum/sum2;
  free(G2);
  return 0;
}

int md_pimd_mc_move_heads_x(md_pimd_t_x *pimd, md_num_t_x *trans, int *par, md_num_t_x max_trans) {
  int l = rand()%(2*pimd->N);
  *par = l;
  int index;
  if (l < pimd->N)
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  else
    index = MD_PIMD_INDEX_X(l-pimd->N+1, pimd->P, pimd->P);
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    trans[i] = md_random_uniform_x(-max_trans, max_trans);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index].x[i] += trans[i];
  return 0;
}

int md_pimd_mc_restore_move_heads_x(md_pimd_t_x *pimd, md_num_t_x *trans, int *par) {
  int l = *par;
  int index;
  if (l < pimd->N)
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  else
    index = MD_PIMD_INDEX_X(l-pimd->N+1, pimd->P, pimd->P);
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index].x[i] -= trans[i];
  return 0;
}

int md_pimd_mc_pbw_normal_gaussian_sample_beads_x(md_pimd_t_x *pimd, int *par, md_num_t_x *rec, int *rec2) {
  int l = *par;
  int i, j;
  int index;
  for (j = 1; j < pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rec[j*MD_DIMENSION_X+i] = pimd->sim->particles[index].x[i];
    if (pimd->pbc) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        rec2[j*MD_DIMENSION_X+i] = pimd->pbW[index*MD_DIMENSION_X+i];
    }
  }
  int js = 0;
  int length = pimd->P+1;
  md_num_t_x endx[MD_DIMENSION_X];
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    endx[i] = pimd->sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
  }
  for (j = js+1; j < js+length-1; ++j) {
    md_num_t_x j1 = js+length-1;
    md_num_t_x aj = (j1-j)/(j1-j+1);
    md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      md_num_t_x r2 = (endx[i]+(j1-j)*pimd->sim->particles[index-1].x[i])/(j1-j+1);
      pimd->sim->particles[index].x[i] = r2+sqrt(sig)*md_random_gaussian_x();
      pimd->pbW[MD_DIMENSION_X*index+i] = 0;
    }
  }
  return 0;
}

int md_pimd_mc_pbw_restore_normal_gaussian_sample_beads_x(md_pimd_t_x *pimd, int *par, md_num_t_x *rec, int *rec2) {
  int l = *par;
  int i, j;
  int index;
  for (j = 1; j < pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].x[i] = rec[j*MD_DIMENSION_X+i];
    if (pimd->pbc) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->pbW[index*MD_DIMENSION_X+i] = rec2[j*MD_DIMENSION_X+i];
    }
  }
  return 0;
}

int md_pimd_mc_normal_gaussian_sample_beads_x(md_pimd_t_x *pimd, int length, int *par, int *jstart, md_num_t_x *rec, int image, int *rec2) {
  int l = rand()%pimd->N;
  if (pimd->mc_par >= 0)
    l = pimd->mc_par;
  *par = l;
  length = MD_MIN_X(length, pimd->P);
  int js = rand()%(pimd->P-length+1);
  if (l+1 == pimd->worm_index) {
    length = MD_MIN_X(length, pimd->P+1);
    js = rand()%(pimd->P+1-length+1);
  }
  *jstart = js;
  int i, j;
  int index, index2;
  for (j = js; j < js+length; ++j) {
    if (j >= pimd->P)
      continue;
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rec[(j-js)*MD_DIMENSION_X+i] = pimd->sim->particles[index].x[i];
    if (pimd->pbc) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        rec2[(j-js)*MD_DIMENSION_X+i] = pimd->pbW[index*MD_DIMENSION_X+i];
    }
  }
  if (pimd->pbc) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    index = MD_PIMD_INDEX_X(l+1, js+1, pimd->P);
    index2 = MD_PIMD_INDEX_X(l+1, js+length, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (js != 0)
        pimd->sim->particles[index].x[i] += pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
      if (js+length <= pimd->P)
        pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
    }
  }
  else if (image) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    index = MD_PIMD_INDEX_X(l+1, js+1, pimd->P);
    index2 = MD_PIMD_INDEX_X(l+1, js+length, pimd->P);
    if (js+length <= pimd->P) {
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (pimd->sim->particles[index2].x[i]-pimd->sim->particles[index].x[i] > L[i]/2)
          pimd->sim->particles[index2].x[i] -= L[i];
        else if (pimd->sim->particles[index2].x[i]-pimd->sim->particles[index].x[i] < -L[i]/2)
          pimd->sim->particles[index2].x[i] += L[i];
      }
    }
  }
  if (l+1 == pimd->worm_index && js == 0) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
    index2 = MD_PIMD_INDEX_X(l+1, js+length, pimd->P);
    md_num_t_x aj = js+length-1;
    md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
    if (js+length <= pimd->P) {
      //for (i = 0; i < MD_DIMENSION_X; ++i)
        //pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        pimd->sim->particles[index].x[i] = pimd->sim->particles[index2].x[i]+sqrt(sig)*md_random_gaussian_x();
        pimd->pbW[MD_DIMENSION_X*index+i] = 0;
      }
      //for (i = 0; i < MD_DIMENSION_X; ++i)
        //pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
    }
    else {
      index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        pimd->sim->particles[index].x[i] = pimd->worm_pos[i]+sqrt(sig)*md_random_gaussian_x();
        pimd->pbW[MD_DIMENSION_X*index+i] = 0;
      }
    }
    /*index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
    int out = 0;
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (pimd->sim->particles[index].x[i] < 0 || pimd->sim->particles[index].x[i] > L[i]) {
        out = 1;
        break;
      }
    }
    if (out) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] = rec[i];
      if (pimd->pbc) {
        for (i = 0; i < MD_DIMENSION_X; ++i)
          pimd->pbW[index*MD_DIMENSION_X+i] = rec2[i];
      }
    }*/
    int count[MD_DIMENSION_X];
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      count[i] = md_periodic_image_count_x(pimd->sim->particles[index].x[i], L[i]);
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] += count[i]*L[i];
    }
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->worm_pos[i] += count[i]*L[i];
  }
  else if (l+1 == pimd->worm_index && js+length == pimd->P+1) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    index = MD_PIMD_INDEX_X(l+1, js+1, pimd->P);
    md_num_t_x aj = pimd->P+1-(js+1);
    md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
    /*if (js != 0) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] += pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
    }*/
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->worm_pos[i] = pimd->sim->particles[index].x[i]+sqrt(sig)*md_random_gaussian_x();
    /*if (js != 0) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] -= pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
    }*/
  }
  for (j = js+1; j < js+length-1; ++j) {
    md_num_t_x j1 = js+length-1;
    md_num_t_x aj = (j1-j)/(j1-j+1);
    md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      index2 = MD_PIMD_INDEX_X(l+1, js+length, pimd->P);
      md_num_t_x r2;
      if (js+length <= pimd->P)
        r2 = (pimd->sim->particles[index2].x[i]+(j1-j)*pimd->sim->particles[index-1].x[i])/(j1-j+1);
      else
        r2 = (pimd->worm_pos[i]+(j1-j)*pimd->sim->particles[index-1].x[i])/(j1-j+1);
      pimd->sim->particles[index].x[i] = r2+sqrt(sig)*md_random_gaussian_x();
      pimd->pbW[MD_DIMENSION_X*index+i] = 0;
    }
  }
  if (pimd->pbc) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    index = MD_PIMD_INDEX_X(l+1, js+1, pimd->P);
    index2 = MD_PIMD_INDEX_X(l+1, js+length, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (js != 0)
        pimd->sim->particles[index].x[i] -= pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
      if (js+length <= pimd->P)
        pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
    }
  }
  return 0;
}

int md_pimd_mc_restore_normal_gaussian_sample_beads_x(md_pimd_t_x *pimd, int length, int *par, int *jstart, md_num_t_x *rec, int *rec2) {
  int l = *par;
  length = MD_MIN_X(length, pimd->P);
  int js = *jstart;
  int i, j;
  int index;
  for (j = js; j < js+length; ++j) {
    if (j >= pimd->P)
      continue;
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].x[i] = rec[(j-js)*MD_DIMENSION_X+i];
    if (pimd->pbc) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->pbW[index*MD_DIMENSION_X+i] = rec2[(j-js)*MD_DIMENSION_X+i];
    }
  }
  return 0;
}

int md_pimd_mc_move_ring_x(md_pimd_t_x *pimd, md_num_t_x *trans, int *par, md_num_t_x max_trans) {
  int l = rand()%(pimd->N);
  *par = l;
  int index;
  int i, j;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    trans[i] = md_random_uniform_x(-max_trans, max_trans);
  for (j = 0; j < pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].x[i] += trans[i];
  }
  if (pimd->pbc) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    int count[MD_DIMENSION_X];
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      count[i] = md_periodic_image_count_x(pimd->sim->particles[index].x[i], L[i]);
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] += count[i]*L[i];
    }
    if (l+1 == pimd->worm_index) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->worm_pos[i] += count[i]*L[i];
    }
    /*int out = 0;
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      if (pimd->sim->particles[index].x[i] < 0 || pimd->sim->particles[index].x[i] > L[i]) {
        out = 1;
        break;
      }
    }
    if (out) {
      for (j = 0; j < pimd->P; ++j) {
        index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
        for (i = 0; i < MD_DIMENSION_X; ++i)
          pimd->sim->particles[index].x[i] -= trans[i];
      }
      for (i = 0; i < MD_DIMENSION_X; ++i)
        trans[i] = 0;
    }*/
  }
  if (l+1 == pimd->worm_index) {
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->worm_pos[i] += trans[i];
  }
  return 0;
}

int md_pimd_mc_restore_move_ring_x(md_pimd_t_x *pimd, md_num_t_x *trans, int *par) {
  int l = *par;
  int index;
  int i, j;
  for (j = 0; j < pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].x[i] -= trans[i];
  }
  if (pimd->pbc) {
    md_num_t_x *L = NULL;
    if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
      nd_rect_t_x *rect = pimd->sim->box->box;
      L = rect->L;
    }
    int count[MD_DIMENSION_X];
    index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      count[i] = md_periodic_image_count_x(pimd->sim->particles[index].x[i], L[i]);
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] += count[i]*L[i];
    }
  }
  return 0;
}

int md_pimd_mc_open_worm_x(md_pimd_t_x *pimd, md_num_t_x *rec, int *par, md_stats_t_x *acc, int length, int *rec2, md_num_t_x C) {
  int l = rand()%pimd->N;
  *par = l;
  acc->es[0] = 1.0;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  md_num_t_x del[MD_DIMENSION_X];
  int i, j;
  int index, index2;
  for (j = 0; j < pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rec[j*MD_DIMENSION_X+i] = pimd->sim->particles[index].x[i];
    if (pimd->pbc) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        rec2[j*MD_DIMENSION_X+i] = pimd->pbW[index*MD_DIMENSION_X+i];
    }
  }
  md_num_t_x *G2 = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->N);
  md_pimd_mc_calc_connect_prob_x(pimd, l, G2);
  if (md_random_uniform_x(0, 1) > G2[l]) {
    free(G2);
    return 0;
  }
  free(G2);
  index2 = MD_PIMD_INDEX_X(l+1, pimd->P-length+1, pimd->P);
  index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  md_num_t_x dis = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
  pimd->worm_index = l+1;
  md_num_t_x aj = length;
  md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    del[i] = MD_MIN_X(sqrt(sig), L[i]/2.0);
    pimd->worm_pos[i] = md_random_uniform_x(-del[i],del[i])+pimd->sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i];
    pimd->pbW[MD_DIMENSION_X*index+i] = 0;
  }
  int L0[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    L0[i] = 0;
  md_num_t_x dis2 = md_minimum_image_distance_2_x(pimd->worm_pos, sim->particles[index2].x, L, L0, &pimd->pbW[index2*MD_DIMENSION_X]);
  md_num_t_x mult = 1.0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    mult *= 2*del[i]/L[i];
  acc->es[0] = C*pimd->N*mult*exp(-(dis2-dis)/(2*sig));
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
  for (j = pimd->P-length+1; j < pimd->P; ++j) {
    md_num_t_x j1 = pimd->P;
    md_num_t_x aj = (j1-j)/(j1-j+1);
    md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      md_num_t_x r2;
      r2 = (pimd->worm_pos[i]+(j1-j)*pimd->sim->particles[index-1].x[i])/(j1-j+1);
      pimd->sim->particles[index].x[i] = r2+sqrt(sig)*md_random_gaussian_x();
      pimd->pbW[MD_DIMENSION_X*index+i] = 0;
    }
  }
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
  return 0;
}

int md_pimd_mc_close_worm_x(md_pimd_t_x *pimd, md_num_t_x *rec, int *par, md_stats_t_x *acc, int length, int *rec2, md_num_t_x C) {
  int l = pimd->worm_index-1;
  *par = l;
  acc->es[0] = 1.0;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  md_num_t_x del[MD_DIMENSION_X];
  int i, j;
  int index, index2;
  for (j = 0; j < pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      rec[j*MD_DIMENSION_X+i] = pimd->sim->particles[index].x[i];
    if (pimd->pbc) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        rec2[j*MD_DIMENSION_X+i] = pimd->pbW[index*MD_DIMENSION_X+i];
    }
  }
  index2 = MD_PIMD_INDEX_X(l+1, pimd->P-length+1, pimd->P);
  index = MD_PIMD_INDEX_X(l+1, 1, pimd->P);
  int L0[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    L0[i] = 0;
  md_num_t_x dis = md_minimum_image_distance_2_x(pimd->worm_pos, sim->particles[index2].x, L, L0, &pimd->pbW[index2*MD_DIMENSION_X]);
  md_num_t_x aj = length;
  md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    del[i] = MD_MIN_X(sqrt(sig), L[i]/2.0);
    int ii;
    for (ii = -5; ii <= 5; ++ii) {
      md_num_t_x tmp = fabs(pimd->worm_pos[i]-(pimd->sim->particles[index].x[i]+ii*L[i]));
      if (tmp <= L[i]/2.0) {
        if (tmp > del[i]) {
          acc->es[0] = 0.0;
          return 0;
        }
        pimd->pbW[index*MD_DIMENSION_X+i] = ii;
        break;
      }
    }
    pimd->worm_pos[i] = pimd->sim->particles[index].x[i]+pimd->pbW[index*MD_DIMENSION_X+i]*L[i];
  }
  md_num_t_x dis2 = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
  md_num_t_x mult = 1.0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    mult *= 2*del[i]/L[i];
  acc->es[0] = (1.0/(C*pimd->N*mult))*exp(-(dis2-dis)/(2*sig));
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index2].x[i] += pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
  for (j = pimd->P-length+1; j < pimd->P; ++j) {
    md_num_t_x j1 = pimd->P;
    md_num_t_x aj = (j1-j)/(j1-j+1);
    md_num_t_x sig = 2*((pimd->beta/pimd->P)/(2*pimd->sim->particles[0].m))*aj;
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      md_num_t_x r2;
      r2 = (pimd->worm_pos[i]+(j1-j)*pimd->sim->particles[index-1].x[i])/(j1-j+1);
      pimd->sim->particles[index].x[i] = r2+sqrt(sig)*md_random_gaussian_x();
      pimd->pbW[MD_DIMENSION_X*index+i] = 0;
    }
  }
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->sim->particles[index2].x[i] -= pimd->pbW[MD_DIMENSION_X*index2+i]*L[i];
  pimd->worm_index = -1;
  md_pimd_periodic_boundary_x(pimd);
  md_pimd_mc_recalc_ENk_x(pimd, *par, 0);
  md_pimd_mc_recalc_VB_x(pimd, *par);
  md_num_t_x *G2 = (md_num_t_x *)malloc(sizeof(md_num_t_x)*pimd->N);
  md_pimd_mc_calc_connect_prob_x(pimd, l, G2);
  if (md_random_uniform_x(0, 1) > G2[l]) {
    free(G2);
    acc->es[0] = 0.0;
    return 0;
  }
  free(G2);
  return 0;
}

int md_pimd_mc_restore_oc_worm_x(md_pimd_t_x *pimd, int *par, md_num_t_x *rec, int *rec2) {
  int l = *par;
  int i, j;
  int index;
  for (j = 0; j < pimd->P; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].x[i] = rec[j*MD_DIMENSION_X+i];
    if (pimd->pbc) {
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->pbW[index*MD_DIMENSION_X+i] = rec2[j*MD_DIMENSION_X+i];
    }
  }
  return 0;
}

int md_pimd_save_worm_x(md_pimd_t_x *pimd) {
  pimd->old_worm_index = pimd->worm_index;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->old_worm_pos[i] = pimd->worm_pos[i];
  return 0;
}

int md_pimd_restore_worm_x(md_pimd_t_x *pimd) {
  pimd->worm_index = pimd->old_worm_index;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    pimd->worm_pos[i] = pimd->old_worm_pos[i];
  return 0;
}

int md_pimd_mc_recalc_pair_energy_x(md_pimd_t_x *pimd, int par, int jstart, int length, md_num_t_x *old, md_pair_energy_t_x pe, md_stats_t_x *stats) {
  int l = par;
  int j, l2;
  int index, index2;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int count[MD_DIMENSION_X], count2[MD_DIMENSION_X];
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    count[i] = md_periodic_image_count_x(pimd->old_worm_pos[i], L[i]);
    count2[i] = md_periodic_image_count_x(pimd->worm_pos[i], L[i]);
    pimd->old_worm_pos[i] += count[i]*L[i];
    pimd->worm_pos[i] += count2[i]*L[i];
  }
  for (j = jstart; j < jstart+length; ++j) {
    if (j+1 > pimd->P)
      continue;
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (l2 = 0; l2 < pimd->N; ++l2) {
      if (l == l2)
        continue;
      index2 = MD_PIMD_INDEX_X(l2+1, j+1, pimd->P);
      if (j == 0 && l+1 == pimd->old_worm_index) {
        stats->es[stats->N-1] -= 0.5*pe(&old[(j-jstart)*MD_DIMENSION_X], pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] -= 0.5*pe(pimd->old_worm_pos, pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else if (j == 0 && l2+1 == pimd->worm_index) {
        stats->es[stats->N-1] -= 0.5*pe(&old[(j-jstart)*MD_DIMENSION_X], pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] -= 0.5*pe(&old[(j-jstart)*MD_DIMENSION_X], pimd->worm_pos, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else
        stats->es[stats->N-1] -= pe(&old[(j-jstart)*MD_DIMENSION_X], pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
      if (j == 0 && l+1 == pimd->worm_index) {
        stats->es[stats->N-1] += 0.5*pe(pimd->sim->particles[index].x, pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] += 0.5*pe(pimd->worm_pos, pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else if (j == 0 && l2+1 == pimd->worm_index) {
        stats->es[stats->N-1] += 0.5*pe(pimd->sim->particles[index].x, pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] += 0.5*pe(pimd->sim->particles[index].x, pimd->worm_pos, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else
        stats->es[stats->N-1] += pe(pimd->sim->particles[index].x, pimd->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
    }
  }
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    pimd->old_worm_pos[i] -= count[i]*L[i];
    pimd->worm_pos[i] -= count2[i]*L[i];
  }
  return 0;
}

int md_pimd_mc_recalc_pair_energy_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, int par, int jstart, int length, md_num_t_x *old, md_pair_energy_t_x pe, md_stats_t_x *stats) {
  if (pimd->P != pimd2->P)
    return -1;
  int l = par;
  int j, l2;
  int index, index2;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  int count[MD_DIMENSION_X], count2[MD_DIMENSION_X], count3[MD_DIMENSION_X];
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    count[i] = md_periodic_image_count_x(pimd->old_worm_pos[i], L[i]);
    count2[i] = md_periodic_image_count_x(pimd->worm_pos[i], L[i]);
    count3[i] = md_periodic_image_count_x(pimd2->worm_pos[i], L[i]);
    pimd->old_worm_pos[i] += count[i]*L[i];
    pimd->worm_pos[i] += count2[i]*L[i];
    pimd2->worm_pos[i] += count3[i]*L[i];
  }
  for (j = jstart; j < jstart+length; ++j) {
    if (j+1 > pimd->P)
      continue;
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (l2 = 0; l2 < pimd2->N; ++l2) {
      index2 = MD_PIMD_INDEX_X(l2+1, j+1, pimd->P);
      if (j == 0 && l+1 == pimd->old_worm_index && l2+1 == pimd2->worm_index) {
        stats->es[stats->N-1] -= 0.5*pe(&old[(j-jstart)*MD_DIMENSION_X], pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] -= 0.5*pe(pimd->old_worm_pos, pimd2->worm_pos, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else if (j == 0 && l+1 == pimd->old_worm_index && l2+1 != pimd2->worm_index) {
        stats->es[stats->N-1] -= 0.5*pe(&old[(j-jstart)*MD_DIMENSION_X], pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] -= 0.5*pe(pimd->old_worm_pos, pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else if (j == 0 && l+1 != pimd->old_worm_index && l2+1 == pimd2->worm_index) {
        stats->es[stats->N-1] -= 0.5*pe(&old[(j-jstart)*MD_DIMENSION_X], pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] -= 0.5*pe(&old[(j-jstart)*MD_DIMENSION_X], pimd2->worm_pos, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else
        stats->es[stats->N-1] -= pe(&old[(j-jstart)*MD_DIMENSION_X], pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
      if (j == 0 && l+1 == pimd->worm_index && l2+1 == pimd2->worm_index) {
        stats->es[stats->N-1] += 0.5*pe(pimd->sim->particles[index].x, pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] += 0.5*pe(pimd->worm_pos, pimd2->worm_pos, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else if (j == 0 && l+1 == pimd->worm_index && l2+1 != pimd2->worm_index) {
        stats->es[stats->N-1] += 0.5*pe(pimd->sim->particles[index].x, pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] += 0.5*pe(pimd->worm_pos, pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else if (j == 0 && l+1 != pimd->worm_index && l2+1 == pimd2->worm_index) {
        stats->es[stats->N-1] += 0.5*pe(pimd->sim->particles[index].x, pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
        stats->es[stats->N-1] += 0.5*pe(pimd->sim->particles[index].x, pimd2->worm_pos, L, pimd->sim->particles[index].m)/pimd->P;
      }
      else
        stats->es[stats->N-1] += pe(pimd->sim->particles[index].x, pimd2->sim->particles[index2].x, L, pimd->sim->particles[index].m)/pimd->P;
    }
  }
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    pimd->old_worm_pos[i] -= count[i]*L[i];
    pimd->worm_pos[i] -= count2[i]*L[i];
    pimd2->worm_pos[i] -= count3[i]*L[i];
  }
  return 0;
}

int md_pimd_mc_recalc_trap_energy_x(md_pimd_t_x *pimd, int par, int jstart, int length, md_num_t_x *old, md_trap_energy_t_x te, md_stats_t_x *stats) {
  int l = par;
  int j;
  int index;
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  for (j = jstart; j < jstart+length; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    stats->es[stats->N-1] -= te(&old[(j-jstart)*MD_DIMENSION_X], L, pimd->sim->particles[index].m)/pimd->P;
    stats->es[stats->N-1] += te(pimd->sim->particles[index].x, L, pimd->sim->particles[index].m)/pimd->P;
  }
  return 0;
}

int md_pimd_mc_recalc_pair_force_x(md_pimd_t_x *pimd, int par, int jstart, int length, md_num_t_x *old, md_pair_force_t_x pf) {
  int l = par;
  int i, j, l2;
  int index, index2;
  md_num_t_x f[MD_DIMENSION_X];
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  for (j = jstart; j < jstart+length; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (l2 = 0; l2 < pimd->N; ++l2) {
      if (l == l2)
        continue;
      index2 = MD_PIMD_INDEX_X(l2+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        f[i] = 0;
      pf(&old[(j-jstart)*MD_DIMENSION_X], pimd->sim->particles[index2].x, f, L, pimd->sim->particles[index].m);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        pimd->sim->particles[index].f[i] -= f[i]/pimd->P;
        pimd->sim->particles[index2].f[i] += f[i]/pimd->P;
      }
      for (i = 0; i < MD_DIMENSION_X; ++i)
        f[i] = 0;
      pf(pimd->sim->particles[index].x, pimd->sim->particles[index2].x, f, L, pimd->sim->particles[index].m);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        pimd->sim->particles[index].f[i] += f[i]/pimd->P;
        pimd->sim->particles[index2].f[i] -= f[i]/pimd->P;
      }
    }
  }
  return 0;
}

int md_pimd_mc_recalc_pair_force_2_x(md_pimd_t_x *pimd, md_pimd_t_x *pimd2, int par, int jstart, int length, md_num_t_x *old, md_pair_force_t_x pf) {
  if (pimd->P != pimd2->P)
    return -1;
  int l = par;
  int i, j, l2;
  int index, index2;
  md_num_t_x f[MD_DIMENSION_X];
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  for (j = jstart; j < jstart+length; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (l2 = 0; l2 < pimd2->N; ++l2) {
      index2 = MD_PIMD_INDEX_X(l2+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        f[i] = 0;
      pf(&old[(j-jstart)*MD_DIMENSION_X], pimd2->sim->particles[index2].x, f, L, pimd->sim->particles[index].m);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        pimd->sim->particles[index].f[i] -= f[i]/pimd->P;
        pimd2->sim->particles[index2].f[i] += f[i]/pimd->P;
      }
      for (i = 0; i < MD_DIMENSION_X; ++i)
        f[i] = 0;
      pf(pimd->sim->particles[index].x, pimd2->sim->particles[index2].x, f, L, pimd->sim->particles[index].m);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        pimd->sim->particles[index].f[i] += f[i]/pimd->P;
        pimd2->sim->particles[index2].f[i] -= f[i]/pimd->P;
      }
    }
  }
  return 0;
}

int md_pimd_mc_recalc_trap_force_x(md_pimd_t_x *pimd, int par, int jstart, int length, md_num_t_x *old, md_trap_force_t_x tf) {
  int l = par;
  int i, j;
  int index;
  md_num_t_x f[MD_DIMENSION_X];
  md_num_t_x *L = NULL;
  if (pimd->sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = pimd->sim->box->box;
    L = rect->L;
  }
  for (j = jstart; j < jstart+length; ++j) {
    index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      f[i] = 0;
    tf(&old[(j-jstart)*MD_DIMENSION_X], f, L, pimd->sim->particles[index].m);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].f[i] -= f[i]/pimd->P;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      f[i] = 0;
    tf(pimd->sim->particles[index].x, f, L, pimd->sim->particles[index].m);
    for (i = 0; i < MD_DIMENSION_X; ++i)
      pimd->sim->particles[index].f[i] += f[i]/pimd->P;
  }
  return 0;
}

int md_pimd_mc_recalc_ENk_x(md_pimd_t_x *pimd, int par, int image) {
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  if (pimd->worm_index > 0 && par+1 > pimd->worm_index)
    --par;
  int l, j;
  int l2 = par+1;
  md_num_t_x *Eint = pimd->Eint;
  if (Eint == NULL)
    return -1;
  int index, index2;
  md_num_t_x d;
  int i;
  int L0[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    L0[i] = 0;
  for (l = l2-1; l <= /*l2*/pimd->N; ++l) {
    if (l <= 0)
      continue;
    if (pimd->worm_index > 0 && l == pimd->N)
      continue;
    md_num_t_x res = 0;
    for (j = 1; j < pimd->P; ++j) {
      if (j == pimd->cut_Pj)
        continue;
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, j+1, pimd->P);
      if (pimd->worm_index > 0 && l >= pimd->worm_index) {
        index += pimd->P;
        index2 += pimd->P;
      }
      if (pimd->pbc && j == 1)
        d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, L0, &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (pimd->pbc)
        d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      res += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }
    /*if (l == pimd->worm_index) {
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      d = md_minimum_image_distance_2_x(sim->particles[index].x, pimd->worm_pos, L, &pimd->pbW[index*MD_DIMENSION_X], L0);
      res += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }*/
    Eint[l-1] = res;
    /*if (l == pimd->worm_index) {
      pimd->ENk[l-1][0] = res;
      continue;
    }*/
    index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
    if (pimd->worm_index > 0 && l >= pimd->worm_index) {
      index += pimd->P;
      index2 += pimd->P;
    }
    if (pimd->pbc)
      d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
    else if (image)
      d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
    else
      d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
    pimd->ENk[l-1][0] = res+0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
  }
  for (l = l2-1; l <= pimd->N; ++l)
    for (j = 2; j <= l; ++j) {
      if (l <= 0)
        continue;
      if (pimd->worm_index > 0 && l == pimd->N)
        continue;
      pimd->ENk[l-1][j-1] = pimd->ENk[l-1][j-2]+Eint[l-j];
      if (pimd->P == pimd->cut_Pj)
        continue;
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      if (pimd->worm_index > 0 && l >= pimd->worm_index)
        index += pimd->P;
      if (pimd->worm_index > 0 && l-j+2 >= pimd->worm_index)
        index2 += pimd->P;
      if (pimd->pbc && l == l-j+2)
        d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (pimd->pbc)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L); //d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      //if (l == pimd->worm_index || l-j+2 == pimd->worm_index)
        //d = 0;
      pimd->ENk[l-1][j-1] -= 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
      index = MD_PIMD_INDEX_X(l-j+1, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+2, 1, pimd->P);
      if (pimd->worm_index > 0 && l-j+1 >= pimd->worm_index)
        index += pimd->P;
      if (pimd->worm_index > 0 && l-j+2 >= pimd->worm_index)
        index2 += pimd->P;
      if (pimd->pbc)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L); //d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      //if (l-j+1 == pimd->worm_index || l-j+2 == pimd->worm_index)
        //d = 0;
      pimd->ENk[l-1][j-1] += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
      index = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
      index2 = MD_PIMD_INDEX_X(l-j+1, 1, pimd->P);
      if (pimd->worm_index > 0 && l >= pimd->worm_index)
        index += pimd->P;
      if (pimd->worm_index > 0 && l-j+1 >= pimd->worm_index)
        index2 += pimd->P;
      if (pimd->pbc)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L); //d = md_minimum_image_distance_2_x(sim->particles[index].x, sim->particles[index2].x, L, &pimd->pbW[index*MD_DIMENSION_X], &pimd->pbW[index2*MD_DIMENSION_X]);
      else if (image)
        d = md_minimum_image_distance_x(sim->particles[index].x, sim->particles[index2].x, L);
      else
        d = md_distance_x(sim->particles[index].x, sim->particles[index2].x);
      //if (l == pimd->worm_index || l-j+1 == pimd->worm_index)
        //d = 0;
      pimd->ENk[l-1][j-1] += 0.5*sim->particles[index].m*pimd->omegaP*pimd->omegaP*d*d;
    }
  return 0;
}

int md_pimd_mc_recalc_ENk2_x(md_pimd_t_x *pimd, int par, int image) {
  int l, j;
  int l2 = par+1;
  int index, index2, index3;
  for (l = l2; l <= l2; ++l) {
    index = MD_PIMD_INDEX_X(l, 1, pimd->P);
    index2 = MD_PIMD_INDEX_X(l, pimd->P, pimd->P);
    index3 = MD_PIMD_INDEX_X(l, 1, pimd->P);
    pimd->ENk2[l-1][0] = md_pimd_fast_ENk2_x(pimd, index, index2, index3, image);
  }
  for (l = l2; l <= pimd->N; ++l)
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
  return 0;
}

int md_pimd_mc_recalc_VB_x(md_pimd_t_x *pimd, int par) {
  int N2, k;
  if (pimd->worm_index > 0 && par+1 > pimd->worm_index)
    --par;
  pimd->VBN[0] = 0.0;
  for (N2 = /*par+1*/par; N2 <= pimd->N; ++N2) {
    if (N2 <= 0)
      continue;
    if (pimd->worm_index > 0 && N2 == pimd->N) {
      pimd->VBN[N2] = pimd->VBN[N2-1];
      continue;
    }
    md_num_t_x sum = 0;
    md_num_t_x tmp = md_pimd_xminE_x(pimd, N2);
    for (k = 1; k <= N2; ++k) {
      if (pimd->vi == 0 && k-1 != 0)
        continue;
      sum += md_pimd_xexp_x(k, pimd->ENk[N2-1][k-1]+pimd->VBN[N2-k], tmp, pimd->beta, pimd->vi);
    }
    pimd->VBN[N2] = (tmp-log(sum)+log(N2))/pimd->beta;
  }
  return 0;
}

int md_pimd_fill_Evir_x(md_pimd_t_x *pimd) {
  md_num_t_x *res = pimd->Evir;
  if (res == NULL)
    return -1;
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
  }
  return 0;
}

int md_pimd_mc_recalc_Evir_x(md_pimd_t_x *pimd, int par) {
  md_num_t_x *res = pimd->Evir;
  if (res == NULL)
    return -1;
  res[0] = 0;
  int N2, k;
  for (N2 = par+1; N2 <= pimd->N; ++N2) {
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
  }
  return 0;
}

int md_pimd_mc_calc_vi_virial_energy_x(md_pimd_t_x *pimd, md_stats_t_x *stats, int image) {
  md_num_t_x *res = pimd->Evir;
  if (res == NULL)
    return -1;
  stats->es[stats->N-1] += res[pimd->N];
  stats->es[stats->N-1] += MD_DIMENSION_X*pimd->N/(2*pimd->beta);
  md_num_t_x res2 = 0;
  int l, j, i;
  int index, index2;
  md_simulation_t_x *sim = pimd->sim;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (j = 1; j <= pimd->P; ++j)
    for (l = 1; l <= pimd->N; ++l) {
      index = MD_PIMD_INDEX_X(l, j, pimd->P);
      index2 = MD_PIMD_INDEX_X(l, 1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i) {
        if (pimd->pbc)
          res2 += -((sim->particles[index].x[i]+pimd->pbW[MD_DIMENSION_X*index+i]*L[i])-(sim->particles[index2].x[i]+pimd->pbW[MD_DIMENSION_X*index2+i]*L[i]))*sim->particles[index].f[i]*sim->particles[index].m;
        else if (image)
          res2 += -md_minimum_image_x(sim->particles[index].x[i]-sim->particles[index2].x[i], L[i])*sim->particles[index].f[i]*sim->particles[index].m;
        else
          res2 += -(sim->particles[index].x[i]-sim->particles[index2].x[i])*sim->particles[index].f[i]*sim->particles[index].m;
      }
    }
  stats->es[stats->N-1] += res2/2.0;
  return 0;
}

int md_pimd_mc_rearrange_x(md_pimd_t_x *pimd, int par) {
  int l, j, i;
  int index, index2;
  md_num_t_x *tmpx = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_DIMENSION_X*pimd->N*pimd->P);
  md_num_t_x *tmpf = (md_num_t_x *)malloc(sizeof(md_num_t_x)*MD_DIMENSION_X*pimd->N*pimd->P);
  if (tmpx == NULL || tmpf == NULL)
    return -1;
  for (l = 0; l < pimd->N*pimd->P; ++l) {
    for (i = 0; i < MD_DIMENSION_X; ++i)
      tmpx[l*MD_DIMENSION_X+i] = pimd->sim->particles[l].x[i];
    for (i = 0; i < MD_DIMENSION_X; ++i)
      tmpf[l*MD_DIMENSION_X+i] = pimd->sim->particles[l].f[i];
  }
  for (l = 0; l < pimd->N; ++l) {
    for (j = 0; j < pimd->P; ++j) {
      index = MD_PIMD_INDEX_X(l+1, j+1, pimd->P);
      index2 = MD_PIMD_INDEX_X(((l+par)%pimd->N)+1, j+1, pimd->P);
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].x[i] = tmpx[index2*MD_DIMENSION_X+i];
      for (i = 0; i < MD_DIMENSION_X; ++i)
        pimd->sim->particles[index].f[i] = tmpf[index2*MD_DIMENSION_X+i];
    }
  }
  free(tmpx);
  free(tmpf);
  return 0;
}

int md_pimd_periodic_boundary_x(md_pimd_t_x *pimd) {
  md_simulation_t_x *sim = pimd->sim;
  int l, i;
  md_num_t_x *L = NULL;
  if (sim->box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = sim->box->box;
    L = rect->L;
  }
  for (l = 0; l < sim->N; ++l)
    for (i = 0; i < MD_DIMENSION_X; ++i) {
      int count = md_periodic_image_count_x(sim->particles[l].x[i], L[i]);
      sim->particles[l].x[i] += count*L[i];
      pimd->pbW[l*MD_DIMENSION_X+i] += -count;
    }
  return 0;
}