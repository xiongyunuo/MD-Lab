#include "md_util_x.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef MD_USE_OPENCL_X
cl_uint md_num_platforms_x;
cl_int *md_num_devices_x;
cl_int *md_num_queues_x;
cl_platform_id *md_platforms_x;
cl_context *md_contexts_x;
cl_device_id **md_devices_x;
cl_command_queue **md_command_queues_x;
cl_device_id **md_command_devices_x;
cl_program *md_programs_x;
int md_que_cur_i_x, md_que_cur_j_x;

int md_get_command_queue_x(int *i, int *j, cl_context *context, cl_device_id *device, cl_command_queue *queue) {
  int ii = md_que_cur_i_x;
  int jj = md_que_cur_j_x;
  if (i != NULL)
    *i = md_que_cur_i_x;
  if (j != NULL)
    *j = md_que_cur_j_x;
  if (context != NULL)
    *context = md_contexts_x[ii];
  if (device != NULL)
    *device = md_command_devices_x[ii][jj];
  if (queue != NULL)
    *queue = md_command_queues_x[ii][jj];
  return 0;
}

int md_roundup_x(int a, int b) {
  if (a%b == 0)
    return a/b;
  return (int)(a/b)+1;
}

int md_get_work_size_x(cl_kernel kernel, cl_device_id device, int d, int *size, size_t *global, size_t *local) {
  cl_uint max_units;
  size_t max_gsize;
  cl_int err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_units, NULL);
  //err |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_gsize, NULL);
  err |= clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_gsize, NULL);
  if (err != CL_SUCCESS)
    return err;
  if (max_gsize >= 512)
    max_gsize = 512;
  if (d == 1) {
    if (size[0] <= max_gsize*max_units/2)
      local[0] = MD_MIN_X(size[0], max_gsize);
    else
      local[0] = MD_MIN_X(md_roundup_x(size[0], max_units), max_gsize);
    if (local[0] <= 0)
      local[0] = 1;
    global[0] = local[0]*md_roundup_x(size[0], local[0]);
    return 0;
  }
  return -1;
}

static int md_queue_size_x = 0;

cl_int md_update_queue_x(cl_command_queue queue) {
  ++md_queue_size_x;
  if (md_queue_size_x >= MD_MAX_QUEUE_SIZE_X) {
    cl_int status;
    cl_event evt = NULL;
    status = clEnqueueMarker(queue, &evt);
    if (status != CL_SUCCESS)
      return status;
    status = clWaitForEvents(1, &evt);
    if (status != CL_SUCCESS)
      return status;
    status = clReleaseEvent(evt);
    if (status != CL_SUCCESS)
      return status;
    md_queue_size_x = 0;
  }
  return 0;
}
#endif

md_attr_pair_t_x *md_alloc_attr_pair_x(void) {
  md_attr_pair_t_x *res = (md_attr_pair_t_x *)malloc(sizeof(md_attr_pair_t_x));
  if (res == NULL)
    return NULL;
  res->N = 0;
  res->names = NULL;
  res->values = NULL;
  return res;
}

md_attr_pair_t_x *md_read_attr_pair_x(FILE *in, md_attr_pair_t_x *attr) {
  md_attr_pair_t_x *res = attr;
  if (attr == NULL)
    res = md_alloc_attr_pair_x();
  if (res == NULL)
    return NULL;
  char name[MD_MAX_NAME_LENGTH_X+1];
  double value;
  while (fscanf(in, "%" MD_STRINGIFY_X(MD_MAX_NAME_LENGTH_X) "s", name) == 1) {
    if (strlen(name) == 0)
      break;
    if (fscanf(in, "%lf", &value) != 1)
      return NULL;
    res->N++;
    res->names = (md_name_t_x *)realloc(res->names, sizeof(md_name_t_x)*res->N);
    res->values = (md_num_t_x *)realloc(res->values, sizeof(md_num_t_x)*res->N);
    if (res->names == NULL || res->values == NULL)
      return NULL;
    strcpy(res->names[res->N-1], name);
    res->values[res->N-1] = (md_num_t_x)value;
    name[0] = '\0';
  }
  return res;
}

md_num_t_x md_get_attr_value_x(md_attr_pair_t_x *attr, const char *name, int *found) {
  int i;
  if (found)
    *found = 0;
  md_num_t_x res = 0;
  for (i = 0; i < attr->N; ++i) {
    if (strcmp(name, attr->names[i]) == 0) {
      if (found)
        *found = 1;
      res = attr->values[i];
      break;
    }
  }
  return res;
}

int md_init_setup_x(FILE *out, void *config) {
  md_init_config_x *con = (md_init_config_x *)config;
  md_set_seed_x(con->seed);
#ifdef MD_USE_OPENCL_X
  if (out)
    fprintf(out, "Using OpenCL\nInitialising...\n");
  cl_int status;
  status = clGetPlatformIDs(0, NULL, &md_num_platforms_x);
  md_platforms_x = (cl_platform_id *)malloc(sizeof(cl_platform_id)*md_num_platforms_x);
  md_num_devices_x = (cl_int *)malloc(sizeof(cl_int)*md_num_platforms_x);
  md_num_queues_x = (cl_int *)malloc(sizeof(cl_int)*md_num_platforms_x);
  md_contexts_x = (cl_context *)malloc(sizeof(cl_context)*md_num_platforms_x);
  md_programs_x = (cl_program *)malloc(sizeof(cl_program)*md_num_platforms_x);
  md_devices_x = (cl_device_id **)malloc(sizeof(cl_device_id *)*md_num_platforms_x);
  md_command_queues_x = (cl_command_queue **)malloc(sizeof(cl_command_queue *)*md_num_platforms_x);
  md_command_devices_x = (cl_device_id **)malloc(sizeof(cl_device_id *)*md_num_platforms_x);
  if (md_platforms_x == NULL || md_num_devices_x == NULL || md_contexts_x == NULL || md_devices_x == NULL || md_command_queues_x == NULL || md_command_devices_x == NULL || md_num_queues_x == NULL || md_programs_x == NULL) {
    if (out)
      fprintf(out, "malloc error\n");
    return -1;
  }
  status |= clGetPlatformIDs(md_num_platforms_x, md_platforms_x, NULL);
  if (status != CL_SUCCESS || md_num_platforms_x <= 0) {
    if (out)
      fprintf(out, "clGetPlatformIDs error\n");
    return status;
  }
  if (out)
    fprintf(out, "Available platform number: %d\n", md_num_platforms_x);
  int i, j, k;
  for (i = 0; i < md_num_platforms_x; ++i) {
    if (out)
      fprintf(out, "Getting info for platform #%d...\n", i+1);
    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)md_platforms_x[i], 0 };
    //printf("%d\n", rand());
    if (con->acc_count == 0 && con->gpu_count == 0)
      md_contexts_x[i] = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &status);
    else if (con->acc_count == 0 && con->cpu_count == 0)
      md_contexts_x[i] = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    else
      md_contexts_x[i] = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_ALL, NULL, NULL, &status);
    //srand(con->seed);
    //printf("%d\n", rand());
    if (status != CL_SUCCESS) {
      if (out)
        fprintf(out, "clCreateContextFromType error\n");
      return status;
    }
    size_t device_size;
    status = clGetContextInfo(md_contexts_x[i], CL_CONTEXT_DEVICES, 0, NULL, &device_size);
    if (status != CL_SUCCESS || device_size <= 0) {
      if (out)
        fprintf(out, "clGetContextInfo error\n");
      return status;
    }
    md_num_devices_x[i] = device_size/sizeof(cl_device_id);
    md_devices_x[i] = (cl_device_id *)malloc(sizeof(cl_device_id)*md_num_devices_x[i]);
    if (md_devices_x[i] == NULL) {
      if (out)
        fprintf(out, "malloc error\n");
      return -1;
    }
    if (out)
      fprintf(out, "Number of available devices: %d\n", md_num_devices_x[i]);
    status = clGetContextInfo(md_contexts_x[i], CL_CONTEXT_DEVICES, device_size, md_devices_x[i], NULL);
    if (status != CL_SUCCESS) {
      if (out)
        fprintf(out, "clGetContextInfo error\n");
      return status;
    }
    int count = 0;
    //printf("%d\n", rand());
    for (j = 0; j < md_num_devices_x[i]; ++j) {
      cl_device_type type;
      status = clGetDeviceInfo(md_devices_x[i][j], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
      if (status != CL_SUCCESS) {
        if (out)
          fprintf(out, "clGetDeviceInfo error\n");
        return status;
      }
      if (type == CL_DEVICE_TYPE_CPU)
        count += con->cpu_count;
      else if (type == CL_DEVICE_TYPE_GPU)
        count += con->gpu_count;
      else if (type == CL_DEVICE_TYPE_ACCELERATOR)
        count += con->acc_count;
    }
    if (out)
      fprintf(out, "Total command queues created: %d\n", count);
    md_num_queues_x[i] = count;
    md_command_queues_x[i] = (cl_command_queue *)malloc(sizeof(cl_command_queue)*count);
    md_command_devices_x[i] = (cl_device_id *)malloc(sizeof(cl_device_id)*count);
    if (count == 0)
      continue;
    if (md_command_queues_x[i] == NULL || md_command_devices_x[i] == NULL) {
      if (out)
        fprintf(out, "malloc error\n");
      return -1;
    }
    count = 0;
    for (j = 0; j < md_num_devices_x[i]; ++j) {
      cl_device_type type;
      status = clGetDeviceInfo(md_devices_x[i][j], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
      if (status != CL_SUCCESS) {
        if (out)
          fprintf(out, "clGetDeviceInfo error\n");
        return status;
      }
      if (type == CL_DEVICE_TYPE_CPU) {
        for (k = 0; k < con->cpu_count; ++k) {
          md_command_devices_x[i][count] = md_devices_x[i][j];
          md_command_queues_x[i][count] = clCreateCommandQueue(md_contexts_x[i], md_devices_x[i][j], 0, &status);
          if (status != CL_SUCCESS) {
            if (out)
              fprintf(out, "clCreateCommandQueue error\n");
            return status;
          }
          ++count;
        }
      }
      else if (type == CL_DEVICE_TYPE_GPU) {
        for (k = 0; k < con->gpu_count; ++k) {
          md_command_devices_x[i][count] = md_devices_x[i][j];
          md_command_queues_x[i][count] = clCreateCommandQueue(md_contexts_x[i], md_devices_x[i][j], 0, &status);
          if (status != CL_SUCCESS) {
            if (out)
              fprintf(out, "clCreateCommandQueue error\n");
            return status;
          }
          ++count;
        }
      }
      else if (type == CL_DEVICE_TYPE_ACCELERATOR) {
        for (k = 0; k < con->acc_count; ++k) {
          md_command_devices_x[i][count] = md_devices_x[i][j];
          md_command_queues_x[i][count] = clCreateCommandQueue(md_contexts_x[i], md_devices_x[i][j], 0, &status);
          if (status != CL_SUCCESS) {
            if (out)
              fprintf(out, "clCreateCommandQueue error\n");
            return status;
          }
          ++count;
        }
      }
    }
    if (out)
      fprintf(out, "Compiling CL programs...\n");
    char filename[2048];
    filename[0] = '\0';
    strcat(filename, con->cl_path);
    strcat(filename, "md_cl_lib_x.cl");
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      if (out)
        fprintf(out, "fopen error\n");
      return -1;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *fcontent = (char *)malloc((size+1)*sizeof(char));
    if (fcontent == NULL) {
      if (out)
        fprintf(out, "malloc error\n");
      return -1;
    }
    long ret = fread(fcontent, 1, size, fp);
    if (ret != size) {
      if (out)
        fprintf(out, "fread error\n");
      return -1;
    }
    fcontent[size] = '\0';
    fclose(fp);
    const char *src = (const char *)fcontent;
    md_programs_x[i] = clCreateProgramWithSource(md_contexts_x[i], 1, (const char **)&src, NULL, &status);
    if (status != CL_SUCCESS) {
      if (out)
        fprintf(out, "clCreateProgramWithSource error\n");
      return status;
    }
    //printf("%d\n", rand());
    status = clBuildProgram(md_programs_x[i], 0, NULL, MD_CL_COMPILER_OPTIONS_X, NULL, NULL);
    if (status != CL_SUCCESS) {
      char buildLog[2048];
      clGetProgramBuildInfo(md_programs_x[i], md_devices_x[i][0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
      if (out)
        fprintf(out, "Error in kernel: %s\n", buildLog);
      clReleaseProgram(md_programs_x[i]);
      return status;
    }
    //printf("%d\n", rand());
    free(fcontent);
  }
  //md_que_cur_i_x = md_que_cur_j_x = 0;
  int found = 0;
  for (i = 0; i < md_num_platforms_x; ++i) {
    if (md_num_queues_x[i] > 0) {
      md_que_cur_i_x = i;
      md_que_cur_j_x = 0;
      found = 1;
      break;
    }
  }
  if (!found) {
    if (out)
      fprintf(out, "No devices found\n");
    return -1;
  }
#endif
  if (out)
    fprintf(out, "Initialization completed\n");
  //printf("%d\n", rand());
  srand(con->seed);
  return 0;
}

int md_pimd_mode_x = 0;
void md_set_pimd_mode_x(int pimd_mode) {
  md_pimd_mode_x = pimd_mode;
}

#ifdef MD_USE_OPENCL_X
cl_int md_simulation_to_context_x(md_simulation_t_x *sim, cl_context context) {
  cl_int status;
  if (context == sim->context && sim->particles_mem != NULL)
    return 0;
  else if (context != sim->context && sim->particles_mem != NULL) {
    status = clReleaseMemObject(sim->particles_mem);
    status |= clReleaseMemObject(sim->fs_mem);
    status |= clReleaseMemObject(sim->f0s_mem);
    status |= clReleaseMemObject(sim->nhcs_mem);
    if (!md_pimd_mode_x) {
      status |= clReleaseMemObject(sim->pair_in_mem);
      status |= clReleaseMemObject(sim->pair_ex_mem);
    }
    status |= md_simulation_clear_cache_x(sim);
    sim->particles_mem = NULL;
    sim->fs_mem = NULL;
    sim->f0s_mem = NULL;
    sim->nhcs_mem = NULL;
    sim->pair_in_mem = NULL;
    sim->pair_ex_mem = NULL;
    sim->context = NULL;
    sim->queue = NULL;
    if (status != CL_SUCCESS)
      return status;
  }
  cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
  sim->particles_mem = clCreateBuffer(context, flags, sizeof(md_particle_t_x)*sim->N, sim->particles, &status);
  if (status != CL_SUCCESS)
    return status;
  sim->fs_mem = clCreateBuffer(context, flags, sizeof(int)*sim->Nf, sim->fs, &status);
  if (status != CL_SUCCESS)
    return status;
  sim->f0s_mem = clCreateBuffer(context, flags, sizeof(int)*sim->Nf, sim->f0s, &status);
  if (status != CL_SUCCESS)
    return status;
  sim->nhcs_mem = clCreateBuffer(context, flags, sizeof(md_nhc_t_x)*sim->Nf, sim->nhcs, &status);
  if (status != CL_SUCCESS)
    return status;
  if (!md_pimd_mode_x) {
    sim->pair_in_mem = clCreateBuffer(context, flags, sizeof(md_index_pair_t_x)*sim->pcount_in, sim->pair_in, &status);
    if (status != CL_SUCCESS)
      return status;
    sim->pair_ex_mem = clCreateBuffer(context, flags, sizeof(md_index_pair_t_x)*sim->pcount_ex, sim->pair_ex, &status);
    if (status != CL_SUCCESS)
      return status;
  }
  sim->context = context;
  sim->queue = NULL;
  return 0;
}

cl_event md_simulation_sync_queue_x(md_simulation_t_x *sim, cl_command_queue queue, cl_int *err) {
  sim->queue = queue;
  cl_event evt = NULL;
  *err = clEnqueueMarker(queue, &evt);
  if (*err != CL_SUCCESS)
    return NULL;
  return evt;
}

cl_int md_simulation_clear_cache_x(md_simulation_t_x *sim) {
  cl_int status = 0;
  if (sim->rf_kernel != NULL) {
    status |= clReleaseKernel(sim->rf_kernel);
    sim->rf_kernel = NULL;
  }
  if (sim->cpf_kernel != NULL) {
    status |= clReleaseKernel(sim->cpf_kernel);
    sim->cpf_kernel = NULL;
  }
  if (sim->forces != NULL) {
    status |= clReleaseMemObject(sim->forces);
    sim->forces = NULL;
  }
  if (sim->cpf_kernel2 != NULL) {
    status |= clReleaseKernel(sim->cpf_kernel2);
    sim->cpf_kernel2 = NULL;
  }
  if (sim->uVV1_kernel != NULL) {
    status |= clReleaseKernel(sim->uVV1_kernel);
    sim->uVV1_kernel = NULL;
  }
  if (sim->uVV2_kernel != NULL) {
    status |= clReleaseKernel(sim->uVV2_kernel);
    sim->uVV2_kernel = NULL;
  }
  if (sim->pb_kernel != NULL) {
    status |= clReleaseKernel(sim->pb_kernel);
    sim->pb_kernel = NULL;
  }
  if (sim->cpe_kernel != NULL) {
    status |= clReleaseKernel(sim->cpe_kernel);
    sim->cpe_kernel = NULL;
  }
  if (sim->add_kernel != NULL) {
    status |= clReleaseKernel(sim->add_kernel);
    sim->add_kernel = NULL;
  }
  if (sim->ct_kernel != NULL) {
    status |= clReleaseKernel(sim->ct_kernel);
    sim->ct_kernel = NULL;
  }
  if (sim->cpc_kernel != NULL) {
    status |= clReleaseKernel(sim->cpc_kernel);
    sim->cpc_kernel = NULL;
  }
  if (sim->aden_kernel != NULL) {
    status |= clReleaseKernel(sim->aden_kernel);
    sim->aden_kernel = NULL;
  }
  if (sim->pc_mem != NULL) {
    status |= clReleaseMemObject(sim->pc_mem);
    sim->pc_mem = NULL;
  }
  if (sim->cSk_kernel != NULL) {
    status |= clReleaseKernel(sim->cSk_kernel);
    sim->cSk_kernel = NULL;
  }
  return status;
}
#endif

#ifdef MD_USE_OPENCL_X
md_simulation_t_x *md_simulation_sync_host_x(md_simulation_t_x *sim, int read_only) {
#else
md_simulation_t_x *md_simulation_sync_host_x(md_simulation_t_x *sim, int MD_UNUSED_X(read_only)) {
#endif
#ifdef MD_USE_OPENCL_X
  if (sim->particles_mem == NULL)
    return sim;
  cl_command_queue queue = sim->queue;
  if (queue == NULL) {
    int i;
    for (i = 0; i < md_num_platforms_x; ++i)
      if (md_contexts_x[i] == sim->context) {
        queue = md_command_queues_x[i][0];
        break;
      }
  }
  cl_int status;
  cl_event events[6];
  status = clEnqueueReadBuffer(queue, sim->particles_mem, CL_FALSE, 0, sizeof(md_particle_t_x)*sim->N, sim->particles, 0, NULL, &events[0]);
  status |= clEnqueueReadBuffer(queue, sim->fs_mem, CL_FALSE, 0, sizeof(int)*sim->Nf, sim->fs, 0, NULL, &events[1]);
  status |= clEnqueueReadBuffer(queue, sim->nhcs_mem, CL_FALSE, 0, sizeof(md_nhc_t_x)*sim->Nf, sim->nhcs, 0, NULL, &events[2]);
  status |= clEnqueueReadBuffer(queue, sim->f0s_mem, CL_FALSE, 0, sizeof(int)*sim->Nf, sim->f0s, 0, NULL, &events[3]);
  if (!md_pimd_mode_x) {
    status |= clEnqueueReadBuffer(queue, sim->pair_in_mem, CL_FALSE, 0, sizeof(md_index_pair_t_x)*sim->pcount_in, sim->pair_in, 0, NULL, &events[4]);
    status |= clEnqueueReadBuffer(queue, sim->pair_ex_mem, CL_FALSE, 0, sizeof(md_index_pair_t_x)*sim->pcount_ex, sim->pair_ex, 0, NULL, &events[5]);
  }
  if (status != CL_SUCCESS)
    return NULL;
  if (!md_pimd_mode_x) {
    status = clWaitForEvents(6, events);
    status |= clReleaseEvent(events[0]);
    status |= clReleaseEvent(events[1]);
    status |= clReleaseEvent(events[2]);
    status |= clReleaseEvent(events[3]);
    status |= clReleaseEvent(events[4]);
    status |= clReleaseEvent(events[5]);
  }
  else {
    status = clWaitForEvents(4, events);
    status |= clReleaseEvent(events[0]);
    status |= clReleaseEvent(events[1]);
    status |= clReleaseEvent(events[2]);
    status |= clReleaseEvent(events[3]);
  }
  if (status != CL_SUCCESS)
    return NULL;
  if (!read_only) {
    status = clReleaseMemObject(sim->particles_mem);
    status |= clReleaseMemObject(sim->fs_mem);
    status |= clReleaseMemObject(sim->f0s_mem);
    status |= clReleaseMemObject(sim->nhcs_mem);
    if (!md_pimd_mode_x) {
      status |= clReleaseMemObject(sim->pair_in_mem);
      status |= clReleaseMemObject(sim->pair_ex_mem);
    }
    status |= md_simulation_clear_cache_x(sim);
    sim->particles_mem = NULL;
    sim->fs_mem = NULL;
    sim->f0s_mem = NULL;
    sim->nhcs_mem = NULL;
    sim->pair_in_mem = NULL;
    sim->pair_ex_mem = NULL;
    sim->context = NULL;
    sim->queue = NULL;
    if (status != CL_SUCCESS)
      return NULL;
  }
#endif
  return sim;
}

md_simulation_t_x *md_alloc_simulation_x(int N, int f, int fc, int box_type) {
  md_simulation_t_x *res = (md_simulation_t_x *)malloc(sizeof(md_simulation_t_x));
  if (res == NULL)
    return NULL;
  res->N = N;
  res->particles = (md_particle_t_x *)malloc(sizeof(md_particle_t_x)*N);
  if (res->particles == NULL) {
    free(res);
    return NULL;
  }
  res->fc = MD_DIMENSION_X*N-fc;
  int Nf = (MD_DIMENSION_X*N-fc)/f;
  if ((MD_DIMENSION_X*N-fc)%f != 0)
    ++Nf;
  res->Nf = Nf;
  res->fs = (int *)malloc(sizeof(int)*Nf);
  res->f0s = (int *)malloc(sizeof(int)*Nf);
  if (res->fs == NULL || res->f0s == NULL) {
    free(res->particles);
    free(res);
    return NULL;
  }
  int i;
  int f0 = 0;
  for (i = 0; i < Nf; ++i) {
    res->fs[i] = f;
    if ((i+1)*f > MD_DIMENSION_X*N-fc)
      res->fs[i] = MD_DIMENSION_X*N-fc-i*f;
    res->f0s[i] = f0;
    f0 += res->fs[i];
  }
  res->nhcs = (md_nhc_t_x *)malloc(sizeof(md_nhc_t_x)*Nf);
  if (res->nhcs == NULL) {
    free(res->fs);
    free(res->particles);
    free(res);
    return NULL;
  }
  res->box = md_alloc_box_x(box_type);
  if (res->box == NULL) {
    free(res->nhcs);
    free(res->fs);
    free(res->particles);
    free(res);
    return NULL;
  }
  int j;
  res->pcount_in = res->N*(res->N+1)/2;
  res->pcount_ex = res->N*(res->N-1)/2;
  if (!md_pimd_mode_x) {
    res->pair_in = (md_index_pair_t_x *)malloc(sizeof(md_index_pair_t_x)*res->pcount_in);
    res->pair_ex = (md_index_pair_t_x *)malloc(sizeof(md_index_pair_t_x)*res->pcount_ex);
    if (res->pair_in == NULL || res->pair_ex == NULL) {
      free(res);
      return NULL;
    }
    int count = 0;
    for (i = 0; i < res->N; ++i)
      for (j = 0; j <= i; ++j) {
        res->pair_in[count].i = i;
        res->pair_in[count].j = j;
        ++count;
      }
    count = 0;
    for (i = 0; i < res->N; ++i)
      for (j = 0; j < i; ++j) {
        res->pair_ex[count].i = i;
        res->pair_ex[count].j = j;
        ++count;
      }
  }
#ifdef MD_USE_OPENCL_X
  res->particles_mem = NULL;
  res->fs_mem = NULL;
  res->f0s_mem = NULL;
  res->nhcs_mem = NULL;
  res->pair_in_mem = NULL;
  res->pair_ex_mem = NULL;
  res->context = NULL;
  res->queue = NULL;
  res->rf_kernel = NULL;
  res->cpf_kernel = NULL;
  res->kname[0] = '\0';
  res->forces = NULL;
  res->cpf_kernel2 = NULL;
  res->uVV1_kernel = NULL;
  res->uVV2_kernel = NULL;
  res->pb_kernel = NULL;
  res->cpe_kernel = NULL;
  res->kname2[0] = '\0';
  res->add_kernel = NULL;
  res->ct_kernel = NULL;
  res->cpc_kernel = NULL;
  res->aden_kernel = NULL;
  res->points = 0;
  res->pc_mem = NULL;
  res->cSk_kernel = NULL;
#endif
  return res;
}

simulation_box_t_x *md_alloc_box_x(int box_type) {
  simulation_box_t_x *res = (simulation_box_t_x *)malloc(sizeof(simulation_box_t_x));
  if (res == NULL)
    return NULL;
  res->type = box_type;
  if (box_type == MD_ND_RECT_BOX_X) {
    res->box = malloc(sizeof(nd_rect_t_x));
    if (res->box == NULL) {
      free(res);
      return NULL;
    }
    return res;
  }
  free(res);
  return NULL;
}

void md_set_seed_x(unsigned int seed) {
  srand(seed);
}

md_num_t_x md_random_uniform_x(md_num_t_x a, md_num_t_x b) {
  md_num_t_x r = ((md_num_t_x)rand())/RAND_MAX;
  return a+(b-a)*r;
}

md_num_t_x md_random_gaussian_x(void) {
  static md_num_t_x first, second;
  static int has = 0;
  if (has) {
    has = 0;
    return second;
  }
  has = 1;
  md_num_t_x r = sqrt(-2*log(md_random_uniform_x(0, 1)));
  md_num_t_x angle = 2*M_PI*md_random_uniform_x(0, 1);
  first = r*cos(angle);
  second = r*sin(angle);
  return first;
}

int md_init_maxwell_vel_x(md_simulation_t_x *sim, md_num_t_x T) {
  sim = md_simulation_sync_host_x(sim, 0);
  if (sim == NULL)
    return -1;
  int i, j;
  md_num_t_x mean[MD_DIMENSION_X];
  for (j = 0; j < MD_DIMENSION_X; ++j)
    mean[j] = 0;
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j) {
      if (j)
        sim->particles[i].v[j] = md_random_gaussian_x()*sqrt(MD_kB_X*T/sim->particles[i].m);
      else
        sim->particles[i].v[j] = md_random_gaussian_x()*sqrt(MD_kB_X*T/sim->particles[i].m);
      mean[j] += sim->particles[i].m*sim->particles[i].v[j];
    }
  for (j = 0; j < MD_DIMENSION_X; ++j)
    mean[j] /= sim->N;
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j)
      sim->particles[i].v[j] -= mean[j]/sim->particles[i].m;
  return 0;
}

int md_init_particle_mass_x(md_simulation_t_x *sim, md_num_t_x m, int start, int end) {
  sim = md_simulation_sync_host_x(sim, 0);
  if (sim == NULL)
    return -1;
  int i;
  for (i = start; i < end; ++i)
    sim->particles[i].m = m;
  return 0;
}

int md_init_particle_face_center_3d_lattice_pos_x(md_simulation_t_x *sim, md_num_t_x *center, md_num_t_x *length, md_num_t_x fluc, md_num_t_x eps) {
  if (MD_DIMENSION_X != 3)
    return -1;
  sim = md_simulation_sync_host_x(sim, 0);
  if (sim == NULL)
    return -1;
  md_num_t_x M2 = pow(sim->N/4.0, 1.0/MD_DIMENSION_X);
  int M = (int)M2;
  if (M2-M != 0)
    ++M;
  int i, j, k, l;
  int count = 0;
  int cell[MD_DIMENSION_X];
  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      for (k = 0; k < M; ++k) {
        if (count >= sim->N)
          break;
        cell[0] = i;
        cell[1] = j;
        cell[2] = k;
        for (l = 0; l < MD_DIMENSION_X; ++l)
          sim->particles[count].x[l] = center[l]-length[l]/2.0+cell[l]*length[l]/M+eps+md_random_uniform_x(-fluc, fluc);
        ++count;
      }
  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      for (k = 0; k < M; ++k) {
        if (count >= sim->N)
          break;
        cell[0] = i;
        cell[1] = j;
        cell[2] = k;
        for (l = 0; l < MD_DIMENSION_X; ++l) {
          if (l != 0)
            sim->particles[count].x[l] = center[l]-length[l]/2.0+cell[l]*length[l]/M+length[l]/M/2.0+eps+md_random_uniform_x(-fluc, fluc);
          else
            sim->particles[count].x[l] = center[l]-length[l]/2.0+cell[l]*length[l]/M+eps+md_random_uniform_x(-fluc, fluc);
        }
        ++count;
      }
  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      for (k = 0; k < M; ++k) {
        if (count >= sim->N)
          break;
        cell[0] = i;
        cell[1] = j;
        cell[2] = k;
        for (l = 0; l < MD_DIMENSION_X; ++l) {
          if (l != 1)
            sim->particles[count].x[l] = center[l]-length[l]/2.0+cell[l]*length[l]/M+length[l]/M/2.0+eps+md_random_uniform_x(-fluc, fluc);
          else
            sim->particles[count].x[l] = center[l]-length[l]/2.0+cell[l]*length[l]/M+eps+md_random_uniform_x(-fluc, fluc);
        }
        ++count;
      }
  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      for (k = 0; k < M; ++k) {
        if (count >= sim->N)
          break;
        cell[0] = i;
        cell[1] = j;
        cell[2] = k;
        for (l = 0; l < MD_DIMENSION_X; ++l) {
          if (l != 2)
            sim->particles[count].x[l] = center[l]-length[l]/2.0+cell[l]*length[l]/M+length[l]/M/2.0+eps+md_random_uniform_x(-fluc, fluc);
          else
            sim->particles[count].x[l] = center[l]-length[l]/2.0+cell[l]*length[l]/M+eps+md_random_uniform_x(-fluc, fluc);
        }
        ++count;
      }
  return 0;
}

int md_init_nhc_x(md_simulation_t_x *sim, md_num_t_x Q, int start, int end, md_num_t_x fp) {
  sim = md_simulation_sync_host_x(sim, 0);
  if (sim == NULL)
    return -1;
  int i, j;
  for (i = start; i < end; ++i) {
    sim->nhcs[i].f = sim->fs[i];
    for (j = 0; j < MD_NHC_LENGTH_X; ++j) {
      sim->nhcs[i].theta[j] = 0;
      sim->nhcs[i].vtheta[j] = 1;
      if (j == 0) {
        if (fp == 0.0)
          sim->nhcs[i].Q[j] = sim->nhcs[i].f*Q;
        else
          sim->nhcs[i].Q[j] = fp*Q;
      }
      else
        sim->nhcs[i].Q[j] = Q;
    }
  }
  return 0;
}

int md_init_nd_rect_box_x(simulation_box_t_x *box, md_num_t_x *L) {
  if (box->type != MD_ND_RECT_BOX_X)
    return -1;
  nd_rect_t_x *rect = (nd_rect_t_x *)box->box;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    rect->L[i] = L[i];
  return 0;
}

int md_fprint_particle_pos_x(FILE *out, md_simulation_t_x *sim) {
  fprintf(out, "x %d %d\n", sim->N, MD_DIMENSION_X);
  sim = md_simulation_sync_host_x(sim, 1);
  if (sim == NULL)
    return -1;
  int i, j;
  for (i = 0; i < sim->N; ++i) {
    for (j = 0; j < MD_DIMENSION_X; ++j)
      fprintf(out, "%f ", sim->particles[i].x[j]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

int md_fprint_particle_vel_x(FILE *out, md_simulation_t_x *sim) {
  fprintf(out, "v %d %d\n", sim->N, MD_DIMENSION_X);
  sim = md_simulation_sync_host_x(sim, 1);
  if (sim == NULL)
    return -1;
  int i, j;
  for (i = 0; i < sim->N; ++i) {
    for (j = 0; j < MD_DIMENSION_X; ++j)
      fprintf(out, "%f ", sim->particles[i].v[j]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

int md_fprint_particle_force_x(FILE *out, md_simulation_t_x *sim) {
  fprintf(out, "f %d %d\n", sim->N, MD_DIMENSION_X);
  sim = md_simulation_sync_host_x(sim, 1);
  if (sim == NULL)
    return -1;
  int i, j;
  for (i = 0; i < sim->N; ++i) {
    for (j = 0; j < MD_DIMENSION_X; ++j)
      fprintf(out, "%f ", sim->particles[i].f[j]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

int md_fprint_nhcs_x(FILE *out, md_simulation_t_x *sim) {
  fprintf(out, "nhc %d %d\n", sim->Nf, MD_NHC_LENGTH_X);
  sim = md_simulation_sync_host_x(sim, 1);
  if (sim == NULL)
    return -1;
  int i, j;
  for (i = 0; i < sim->Nf; ++i) {
    fprintf(out, "%d\n", sim->nhcs[i].f);
    for (j = 0; j < MD_NHC_LENGTH_X; ++j)
      fprintf(out, "%f ", sim->nhcs[i].theta[j]);
    fprintf(out, "\n");
    for (j = 0; j < MD_NHC_LENGTH_X; ++j)
      fprintf(out, "%f ", sim->nhcs[i].vtheta[j]);
    fprintf(out, "\n");
    for (j = 0; j < MD_NHC_LENGTH_X; ++j)
      fprintf(out, "%f ", sim->nhcs[i].Q[j]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

int md_fprint_sim_box_x(FILE *out, simulation_box_t_x *box) {
  fprintf(out, "box %d %d\n", box->type, MD_DIMENSION_X);
  if (box->type == MD_ND_RECT_BOX_X) {
    nd_rect_t_x *rect = (nd_rect_t_x *)box->box;
    int i;
    for (i = 0; i < MD_DIMENSION_X; ++i)
      fprintf(out, "%f ", rect->L[i]);
    fprintf(out, "\n");
  }
  return ferror(out);
}

md_num_t_x md_get_quick_Q_x(md_num_t_x T, md_num_t_x omega) {
  return T*MD_kB_X/(omega*omega);
}

int md_fread_particle_pos_x(FILE *in, md_simulation_t_x *sim) {
  sim = md_simulation_sync_host_x(sim, 0);
  if (sim == NULL)
    return -1;
  char name[32];
  if (fscanf(in, "%31s", name) != 1)
    return -1;
  int N, d;
  if (fscanf(in, "%d", &N) != 1)
    return -1;
  if (N != sim->N)
    return -1;
  if (fscanf(in, "%d", &d) != 1)
    return -1;
  if (d != MD_DIMENSION_X)
    return -1;
  int i, j;
  double num;
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j) {
      if (fscanf(in, "%lf", &num) != 1)
        return -1;
      sim->particles[i].x[j] = num;
    }
  return ferror(in);
}

int md_fread_particle_vel_x(FILE *in, md_simulation_t_x *sim) {
  sim = md_simulation_sync_host_x(sim, 0);
  if (sim == NULL)
    return -1;
  char name[32];
  if (fscanf(in, "%31s", name) != 1)
    return -1;
  int N, d;
  if (fscanf(in, "%d", &N) != 1)
    return -1;
  if (N != sim->N)
    return -1;
  if (fscanf(in, "%d", &d) != 1)
    return -1;
  if (d != MD_DIMENSION_X)
    return -1;
  int i, j;
  double num;
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j) {
      if (fscanf(in, "%lf", &num) != 1)
        return -1;
      sim->particles[i].v[j] = num;
    }
  return ferror(in);
}

int md_fread_nhcs_x(FILE *in, md_simulation_t_x *sim) {
  sim = md_simulation_sync_host_x(sim, 0);
  if (sim == NULL)
    return -1;
  char name[32];
  if (fscanf(in, "%31s", name) != 1)
    return -1;
  int N, d;
  if (fscanf(in, "%d", &N) != 1)
    return -1;
  if (N != sim->Nf)
    return -1;
  if (fscanf(in, "%d", &d) != 1)
    return -1;
  if (d != MD_NHC_LENGTH_X)
    return -1;
  int i, j;
  double num;
  for (i = 0; i < sim->Nf; ++i) {
    int f;
    if (fscanf(in, "%d", &f) != 1)
      return -1;
    sim->nhcs[i].f = f;
    for (j = 0; j < MD_NHC_LENGTH_X; ++j) {
      if (fscanf(in, "%lf", &num) != 1)
        return -1;
      sim->nhcs[i].theta[j] = num;
    }
    for (j = 0; j < MD_NHC_LENGTH_X; ++j) {
      if (fscanf(in, "%lf", &num) != 1)
        return -1;
      sim->nhcs[i].vtheta[j] = num;
    }
    for (j = 0; j < MD_NHC_LENGTH_X; ++j) {
      if (fscanf(in, "%lf", &num) != 1)
        return -1;
      sim->nhcs[i].Q[j] = num;
    }
  }
  return ferror(in);
}

md_num_t_x md_laguerre_poly_x(int n, md_num_t_x x) {
  int i;
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(n+1));
  res[0] = 1;
  res[1] = 1-x;
  for (i = 1; i < n; ++i) {
    res[i+1] = ((2*i+1-x)*res[i]-i*res[i-1])/(i+1);
  }
  md_num_t_x l = res[n];
  free(res);
  return l;
}

md_num_t_x md_coulomb_Cn_x(int n, md_num_t_x Z) {
  //return pow(fabs(Z)/n, 3.0/2.0)*sqrt(1.0/((md_num_t_x)(2*n)*n*(n+1)));
  return sqrt(0.5*pow(fabs(Z), 3)/pow(n, 5));
}

md_num_t_x md_coulomb_En_x(int n, md_num_t_x Z, md_num_t_x ke) {
  return -ke*Z*Z/(4*n*n);
}

md_num_t_x md_r_magnitude_x(md_num_t_x *r) {
  md_num_t_x res = 0;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    res += r[i]*r[i];
  return sqrt(res);
}

md_num_t_x md_coulomb_rho_b_x(int nmax, md_num_t_x Z, md_num_t_x ke, md_num_t_x beta, md_num_t_x *r, md_num_t_x *r2) {
  md_num_t_x mr = md_r_magnitude_x(r);
  md_num_t_x mr2 = md_r_magnitude_x(r2);
  md_num_t_x dr[MD_DIMENSION_X];
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    dr[i] = r[i]-r2[i];
  md_num_t_x mdr = md_r_magnitude_x(dr);
  md_num_t_x x = (mr+mr2+mdr)/2.0;
  md_num_t_x y = (mr+mr2-mdr)/2.0;
  md_num_t_x res = 0;
  int n;
  for (n = 1; n <= nmax; ++n) {
    md_num_t_x En = md_coulomb_En_x(n, Z, ke);
    md_num_t_x Cn = md_coulomb_Cn_x(n, Z);
    md_num_t_x mult = exp(-beta*En)*Cn*Cn*(n*n*n/fabs(Z))*exp(-(fabs(Z)/(2*n))*(mr+mr2));
    md_num_t_x mult2 = md_laguerre_poly_x(n-1, fabs(Z)*x/n)*md_laguerre_poly_x(n, fabs(Z)*y/n)-md_laguerre_poly_x(n-1, fabs(Z)*y/n)*md_laguerre_poly_x(n, fabs(Z)*x/n);
    res += mult*mult2;
  }
  return res/(4*M_PI*mdr);
}

md_num_t_x md_coulomb_diag_rho_b_x(int nmax, md_num_t_x Z, md_num_t_x ke, md_num_t_x beta, md_num_t_x *r) {
  md_num_t_x mr = md_r_magnitude_x(r);
  md_num_t_x res = 0;
  int n;
  for (n = 1; n <= nmax; ++n) {
    md_num_t_x En = md_coulomb_En_x(n, Z, ke);
    md_num_t_x Cn = md_coulomb_Cn_x(n, Z);
    md_num_t_x mult = exp(-beta*En)*Cn*Cn*n*n*exp(-(fabs(Z)/n)*mr);
    md_num_t_x tmp = md_laguerre_poly_x(n-1, fabs(Z)*mr/n)-md_laguerre_poly_x(n, fabs(Z)*mr/n);
    md_num_t_x mult2 = (n*n/(mr*fabs(Z)))*tmp*tmp+md_laguerre_poly_x(n, fabs(Z)*mr/n)*md_laguerre_poly_x(n-1, fabs(Z)*mr/n);
    res += mult*mult2;
  }
  return res/(4*M_PI);
}

md_num_t_x md_coulomb_wave_Ak_x(int k, md_num_t_x eta) {
  int i;
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(k+2));
  res[1] = 1;
  res[2] = eta;
  for (i = 3; i <= k; ++i) {
    res[i] = (2*eta*res[i-1]-res[i-2])/(i*(i-1));
  }
  md_num_t_x tmp = res[k];
  free(res);
  return tmp;
}

md_num_t_x md_coulomb_wave_Bk_x(int k, md_num_t_x eta, md_num_t_x rho) {
  int i;
  md_num_t_x *res = (md_num_t_x *)malloc(sizeof(md_num_t_x)*(k+2));
  res[0] = 1;
  res[1] = eta*rho;
  for (i = 1; i < k; ++i) {
    res[i+1] = (2*eta*rho*res[i]-rho*rho*res[i-1])/((i+1)*(i+2));
  }
  md_num_t_x tmp = res[k];
  free(res);
  return tmp;
}

md_num_t_x md_coulomb_wave_func_x(md_num_t_x eta, md_num_t_x rho, int kmax) {
  int k;
  md_num_t_x res = 0;
  /*for (k = 1; k <= kmax; ++k) {
    res += md_coulomb_wave_Ak_x(k, eta)*pow(rho, k-1);
  }*/
  if (rho <= 10) {
    for (k = 0; k <= kmax; ++k) {
      res += md_coulomb_wave_Bk_x(k, eta, rho);
    }
    return res;
  }
  md_num_t_x tmp, tmp2;
  md_1F1_x(1, -eta, 2, 0, 0, 2*rho, &tmp, &tmp2);
  res = cos(-rho)*tmp-sin(-rho)*tmp2;
  return res;
}

void md_log_gamma_x(md_num_t_x re, md_num_t_x im, md_num_t_x *res, md_num_t_x *res2) {
  *res = 0.5*log(2*M_PI);
  md_num_t_x r = sqrt(re*re+im*im);
  md_num_t_x theta = atan2(im, re);
  *res += (re-0.5)*log(r)-im*theta;
  *res2 = (re-0.5)*theta+im*log(r);
  *res -= re;
  *res2 -= im;
  *res += re/(12*r*r);
  *res2 += -im/(12*r*r);
  md_num_t_x r2 = pow(r,3);
  md_num_t_x theta2 = 3*theta;
  md_num_t_x x = r2*cos(theta2);
  md_num_t_x y = r2*sin(theta2);
  md_num_t_x r3 = sqrt(x*x+y*y);
  *res -= x/(360*r3*r3);
  *res2 -= -y/(360*r3*r3);
  r2 = pow(r,5);
  theta2 = 5*theta;
  x = r2*cos(theta2);
  y = r2*sin(theta2);
  r3 = sqrt(x*x+y*y);
  *res += x/(1260*r3*r3);
  *res2 += -y/(1260*r3*r3);
}

md_num_t_x md_1F1_a_x;
md_num_t_x md_1F1_ia_x;
md_num_t_x md_1F1_b_x;
md_num_t_x md_1F1_ib_x;
md_num_t_x md_1F1_z_x;
md_num_t_x md_1F1_iz_x;

md_num_t_x md_1F1_integrand_x(md_num_t_x u) {
  md_num_t_x a = md_1F1_a_x;
  md_num_t_x ia = md_1F1_ia_x;
  md_num_t_x b = md_1F1_b_x;
  md_num_t_x ib = md_1F1_ib_x;
  md_num_t_x z = md_1F1_z_x;
  md_num_t_x iz = md_1F1_iz_x;
  md_num_t_x res = exp(z*u)*cos(iz*u);
  md_num_t_x res2 = exp(z*u)*sin(iz*u);
  md_num_t_x mult = exp(log(u)*(a-1))*cos(log(u)*ia);
  md_num_t_x mult2 = exp(log(u)*(a-1))*sin(log(u)*ia);
  md_num_t_x tmp = res*mult-res2*mult2;
  md_num_t_x tmp2 = res*mult2+res2*mult;
  res = tmp;
  res2 = tmp2;
  mult = exp(log(1-u)*(b-a-1))*cos(log(1-u)*(ib-ia));
  mult2 = exp(log(1-u)*(b-a-1))*sin(log(1-u)*(ib-ia));
  tmp = res*mult-res2*mult2;
  tmp2 = res*mult2+res2*mult;
  res = tmp;
  res2 = tmp2;
  return res;
}

md_num_t_x md_1F1_im_integrand_x(md_num_t_x u) {
  md_num_t_x a = md_1F1_a_x;
  md_num_t_x ia = md_1F1_ia_x;
  md_num_t_x b = md_1F1_b_x;
  md_num_t_x ib = md_1F1_ib_x;
  md_num_t_x z = md_1F1_z_x;
  md_num_t_x iz = md_1F1_iz_x;
  md_num_t_x res = exp(z*u)*cos(iz*u);
  md_num_t_x res2 = exp(z*u)*sin(iz*u);
  //md_num_t_x mult = exp(log(u)*(a-1))*cos(log(u)*ia);
  //md_num_t_x mult2 = exp(log(u)*(a-1))*sin(log(u)*ia);
  md_num_t_x mult = pow(u, a-1)*cos(log(u)*ia);
  md_num_t_x mult2 = pow(u, a-1)*sin(log(u)*ia);
  md_num_t_x tmp = res*mult-res2*mult2;
  md_num_t_x tmp2 = res*mult2+res2*mult;
  res = tmp;
  res2 = tmp2;
  //mult = exp(log(1-u)*(b-a-1))*cos(log(1-u)*(ib-ia));
  //mult2 = exp(log(1-u)*(b-a-1))*sin(log(1-u)*(ib-ia));
  mult = pow(1-u, b-a-1)*cos(log(1-u)*(ib-ia));
  mult2 = pow(1-u, b-a-1)*sin(log(1-u)*(ib-ia));
  tmp = res*mult-res2*mult2;
  tmp2 = res*mult2+res2*mult;
  res = tmp;
  res2 = tmp2;
  return res2;
}

void md_1F1_x(md_num_t_x a, md_num_t_x ia, md_num_t_x b, md_num_t_x ib, md_num_t_x z, md_num_t_x iz, md_num_t_x *res, md_num_t_x *res2) {
  md_1F1_a_x = a;
  md_1F1_ia_x = ia;
  md_1F1_b_x = b;
  md_1F1_ib_x = ib;
  md_1F1_z_x = z;
  md_1F1_iz_x = iz;
  md_num_t_x inte = md_romberg_integral_x(md_1F1_integrand_x, 1e-6, 1-1e-6, 10, 1e-8);
  md_num_t_x inte2 = md_romberg_integral_x(md_1F1_im_integrand_x, 1e-6, 1-1e-6, 10, 1e-8);
  //printf("%f %f\n", inte, inte2);
  md_num_t_x fac, fac2;
  md_num_t_x tmp, tmp2;
  md_log_gamma_x(b, ib, &tmp, &tmp2);
  fac = tmp;
  fac2 = tmp2;
  md_log_gamma_x(a, ia, &tmp, &tmp2);
  fac -= tmp;
  fac2 -= tmp2;
  md_log_gamma_x(b-a, ib-ia, &tmp, &tmp2);
  fac -= tmp;
  fac2 -= tmp2;
  md_num_t_x efac, efac2;
  efac = exp(fac)*cos(fac2);
  efac2 = exp(fac)*sin(fac2);
  *res = efac*inte-efac2*inte2;
  *res2 = efac*inte2+efac2*inte;
}

md_num_t_x md_coulomb_wave_C_x(md_num_t_x eta) {
  /*md_num_t_x mult = exp(-M_PI*eta/2.0);
  eta = fabs(eta);
  if (eta > 8)
    return 0;
  md_num_t_x g = 1.00418-0.00103694*eta-1.09567*pow(eta,2)+0.896133*pow(eta,3)-0.343346*pow(eta,4)+0.074048*pow(eta,5)-0.00920664*pow(eta,6)+0.000616196*pow(eta,7)-0.000017199*pow(eta,8);
  return mult*g;*/
  md_num_t_x res = 0.5*log(2*M_PI);
  md_num_t_x r = sqrt(1+eta*eta);
  md_num_t_x theta = atan2(eta, 1);
  res += (1-0.5)*log(r)-eta*theta-1;
  res += 1.0/(12*r*r);
  md_num_t_x r2 = pow(r,3);
  md_num_t_x theta2 = 3*theta;
  md_num_t_x x = r2*cos(theta2);
  md_num_t_x y = r2*sin(theta2);
  md_num_t_x r3 = sqrt(x*x+y*y);
  res -= x/(360*r3*r3);
  r2 = pow(r,5);
  theta2 = 5*theta;
  x = r2*cos(theta2);
  y = r2*sin(theta2);
  r3 = sqrt(x*x+y*y);
  res += x/(1260*r3*r3);
  //return exp(res);
  return exp(-M_PI*eta/2.0+res);
}

md_num_t_x md_coulomb_wave_F0_x(md_num_t_x eta, md_num_t_x rho, int kmax) {
  return md_coulomb_wave_C_x(eta)*rho*md_coulomb_wave_func_x(eta, rho, kmax);
}

md_num_t_x md_coulomb_wave_func_star_x(md_num_t_x eta, md_num_t_x rho, int kmax) {
  int k;
  md_num_t_x res = 0;
  /*for (k = 1; k <= kmax; ++k) {
    res += k*md_coulomb_wave_Ak_x(k, eta)*pow(rho, k-1);
  }*/
  for (k = 0; k <= kmax; ++k) {
    res += md_coulomb_wave_Bk_x(k, eta, rho);
  }
  for (k = 0; k <= kmax; ++k) {
    res += k*md_coulomb_wave_Bk_x(k, eta, rho);
  }
  return res;
}

md_num_t_x md_coulomb_wave_dF0_x(md_num_t_x eta, md_num_t_x rho, int kmax) {
  if (rho <= 10) {
    return md_coulomb_wave_C_x(eta)*md_coulomb_wave_func_star_x(eta, rho, kmax);
  }
  md_num_t_x tmp, tmp2;
  md_1F1_x(1, -eta, 2, 0, 0, 2*rho, &tmp, &tmp2);
  md_num_t_x res = md_coulomb_wave_C_x(eta)*(cos(-rho)*tmp-sin(-rho)*tmp2);
  res += md_coulomb_wave_C_x(eta)*rho*(cos(-rho)*tmp2+sin(-rho)*tmp);
  md_1F1_x(2, -eta, 3, 0, 0, 2*rho, &tmp, &tmp2);
  md_num_t_x mult, mult2;
  mult = cos(-rho)*tmp-sin(-rho)*tmp2;
  mult2 = cos(-rho)*tmp2+sin(-rho)*tmp;
  res += md_coulomb_wave_C_x(eta)*rho*(mult*eta-mult2);
  return res;
}

static md_num_t_x md_coulomb_wave_beta_x;
static md_num_t_x md_coulomb_wave_ke_x;
static md_num_t_x md_coulomb_wave_Z_x;
static md_num_t_x md_coulomb_wave_x_x;
static md_num_t_x md_coulomb_wave_y_x;
static md_num_t_x md_coulomb_wave_r_x;
static int md_coulomb_wave_kmax_x;

md_num_t_x md_coulomb_wave_integrand_x(md_num_t_x k) {
  md_num_t_x beta = md_coulomb_wave_beta_x;
  md_num_t_x ke = md_coulomb_wave_ke_x;
  md_num_t_x Z = md_coulomb_wave_Z_x;
  md_num_t_x x = md_coulomb_wave_x_x;
  md_num_t_x y = md_coulomb_wave_y_x;
  int kmax = md_coulomb_wave_kmax_x;
  md_num_t_x mult = k*exp(-beta*ke*k*k);
  md_num_t_x mult2 = md_coulomb_wave_F0_x(Z/(2*k), k*x, kmax)*md_coulomb_wave_dF0_x(Z/(2*k), k*y, kmax)-md_coulomb_wave_F0_x(Z/(2*k), k*y, kmax)*md_coulomb_wave_dF0_x(Z/(2*k), k*x, kmax);
  //printf("%f %f\n", mult*mult2, k);
  return mult*mult2;
}

md_num_t_x md_coulomb_wave_diag_integrand_x(md_num_t_x k) {
  md_num_t_x beta = md_coulomb_wave_beta_x;
  md_num_t_x ke = md_coulomb_wave_ke_x;
  md_num_t_x Z = md_coulomb_wave_Z_x;
  md_num_t_x r = md_coulomb_wave_r_x;
  int kmax = md_coulomb_wave_kmax_x;
  md_num_t_x mult = k*k*exp(-beta*ke*k*k);
  md_num_t_x tmp = md_coulomb_wave_dF0_x(Z/(2*k), k*r, kmax);
  md_num_t_x tmp2 = md_coulomb_wave_F0_x(Z/(2*k), k*r, kmax);
  md_num_t_x mult2 = tmp*tmp+(1-Z/(k*k*r))*tmp2*tmp2;
  return mult*mult2;
}

md_num_t_x md_romberg_integral_x(md_num_t_x (*f)(md_num_t_x), md_num_t_x a, md_num_t_x b, int max_steps, md_num_t_x acc) {
  //md_num_t_x R1[max_steps], R2[max_steps];
  md_num_t_x *R1 = (md_num_t_x *)malloc(sizeof(md_num_t_x)*max_steps);
  md_num_t_x *R2 = (md_num_t_x *)malloc(sizeof(md_num_t_x)*max_steps);
  md_num_t_x *Rp = &R1[0], *Rc = &R2[0];
  md_num_t_x h = b-a;
  Rp[0] = (f(a) + f(b))*h*0.5;
  int i, j;
  for (i = 1; i < max_steps; ++i) {
    h /= 2.0;
    md_num_t_x c = 0;
    int ep = 1 << (i-1);
    for (j = 1; j <= ep; ++j) {
      c += f(a + (2*j-1) * h);
    }
    Rc[0] = h*c + 0.5*Rp[0];

    for (j = 1; j <= i; ++j) {
      md_num_t_x n_k = pow(4, j);
      Rc[j] = (n_k*Rc[j-1] - Rp[j-1]) / (n_k-1);
    }

    if (i > 1 && fabs(Rp[i-1]-Rc[i]) < acc) {
      md_num_t_x res = Rc[i];
      free(R1);
      free(R2);
      return res;
    }
    md_num_t_x *rt = Rp;
    Rp = Rc;
    Rc = rt;
  }
  md_num_t_x res = Rp[max_steps-1];
  free(R1);
  free(R2);
  return res;
}

md_num_t_x md_coulomb_rho_c_x(int kmax, md_num_t_x Z, md_num_t_x ke, md_num_t_x beta, md_num_t_x *r, md_num_t_x *r2) {
  md_num_t_x mr = md_r_magnitude_x(r);
  md_num_t_x mr2 = md_r_magnitude_x(r2);
  md_num_t_x dr[MD_DIMENSION_X];
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    dr[i] = r[i]-r2[i];
  md_num_t_x mdr = md_r_magnitude_x(dr);
  md_num_t_x x = (mr+mr2+mdr)/2.0;
  md_num_t_x y = (mr+mr2-mdr)/2.0;
  md_coulomb_wave_beta_x = beta;
  md_coulomb_wave_ke_x = ke;
  md_coulomb_wave_Z_x = Z;
  md_coulomb_wave_x_x = x;
  md_coulomb_wave_y_x = y;
  md_coulomb_wave_kmax_x = kmax;
  md_num_t_x limit = 2.0*sqrt(5.0/(beta*ke));
  return md_romberg_integral_x(md_coulomb_wave_integrand_x, 1e-4, limit, 10, 1e-8)/(2*M_PI*M_PI*mdr);
}

md_num_t_x md_coulomb_diag_rho_c_x(int kmax, md_num_t_x Z, md_num_t_x ke, md_num_t_x beta, md_num_t_x *r) {
  md_num_t_x mr = md_r_magnitude_x(r);
  md_coulomb_wave_beta_x = beta;
  md_coulomb_wave_ke_x = ke;
  md_coulomb_wave_Z_x = Z;
  md_coulomb_wave_r_x = mr;
  md_coulomb_wave_kmax_x = kmax;
  md_num_t_x limit = 2.0*sqrt(5.0/(beta*ke));
  return md_romberg_integral_x(md_coulomb_wave_diag_integrand_x, 1e-4, limit, 10, 1e-8)/(2*M_PI*M_PI);
}

md_num_t_x md_coulomb_free_rho_x(md_num_t_x ke, md_num_t_x beta, md_num_t_x *r, md_num_t_x *r2) {
  md_num_t_x dr[MD_DIMENSION_X];
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    dr[i] = r[i]-r2[i];
  md_num_t_x mdr = md_r_magnitude_x(dr);
  return exp(-mdr*mdr/(4*ke*beta))/pow(4*M_PI*ke*beta, 1.5);
}

md_pa_table_t_x *md_read_pa_table_x(FILE *in) {
  md_pa_table_t_x *res = (md_pa_table_t_x *)malloc(sizeof(md_pa_table_t_x));
  if (res == NULL)
    return NULL;
  res->pa_func = NULL;
  double tmp;
  if (fscanf(in, "%lf", &tmp) != 1) {
    free(res);
    return NULL;
  }
  res->beta = tmp;
  if (fscanf(in, "%lf", &tmp) != 1) {
    free(res);
    return NULL;
  }
  res->ka = tmp;
  if (fscanf(in, "%lf", &tmp) != 1) {
    free(res);
    return NULL;
  }
  res->Z = tmp;
  if (fscanf(in, "%d", &res->ucount) != 1 || res->ucount <= 0) {
    free(res);
    return NULL;
  }
  res->r = (md_num_t_x *)malloc(sizeof(md_num_t_x)*res->ucount);
  res->u = (md_num_t_x *)malloc(sizeof(md_num_t_x)*res->ucount);
  res->A = (md_num_t_x *)malloc(sizeof(md_num_t_x)*res->ucount);
  if (res->r == NULL || res->u == NULL || res->A == NULL) {
    free(res);
    return NULL;
  }
  int i;
  for (i = 0; i < res->ucount; ++i) {
    if (fscanf(in, "%lf", &tmp) != 1) {
      free(res);
      return NULL;
    }
    res->r[i] = tmp;
    if (fscanf(in, "%lf", &tmp) != 1) {
      free(res);
      return NULL;
    }
    res->u[i] = tmp;
  }
  int tmp2;
  if (fscanf(in, "%d", &tmp2) != 1 || res->ucount != tmp2) {
    free(res);
    return NULL;
  }
  for (i = 0; i < res->ucount; ++i) {
    if (fscanf(in, "%lf", &tmp) != 1) {
      free(res);
      return NULL;
    }
    res->r[i] = tmp;
    if (fscanf(in, "%lf", &tmp) != 1) {
      free(res);
      return NULL;
    }
    res->A[i] = tmp;
  }
  if (fscanf(in, "%d", &res->ducount) != 1 || res->ducount <= 0) {
    free(res);
    return NULL;
  }
  res->r2 = (md_num_t_x *)malloc(sizeof(md_num_t_x)*res->ducount);
  res->du = (md_num_t_x *)malloc(sizeof(md_num_t_x)*res->ducount);
  res->dA = (md_num_t_x *)malloc(sizeof(md_num_t_x)*res->ducount);
  if (res->r2 == NULL || res->du == NULL || res->dA == NULL) {
    free(res);
    return NULL;
  }
  for (i = 0; i < res->ducount; ++i) {
    if (fscanf(in, "%lf", &tmp) != 1) {
      free(res);
      return NULL;
    }
    res->r2[i] = tmp;
    if (fscanf(in, "%lf", &tmp) != 1) {
      free(res);
      return NULL;
    }
    res->du[i] = tmp;
  }
  if (fscanf(in, "%d", &tmp2) != 1 || res->ducount != tmp2) {
    free(res);
    return NULL;
  }
  for (i = 0; i < res->ducount; ++i) {
    if (fscanf(in, "%lf", &tmp) != 1) {
      free(res);
      return NULL;
    }
    res->r2[i] = tmp;
    if (fscanf(in, "%lf", &tmp) != 1) {
      free(res);
      return NULL;
    }
    res->dA[i] = tmp;
  }
  return res;
}

md_num_t_x md_pa_delta_x(md_num_t_x *r1, md_num_t_x *r2, md_num_t_x beta, int N, md_num_t_x *params) {
  if (N < 2)
    return 0;
  md_num_t_x as = params[0];
  md_num_t_x mu = params[1];
  int i;
  md_num_t_x dr[MD_DIMENSION_X];
  for (i = 0; i < MD_DIMENSION_X; ++i)
    dr[i] = r1[i]-r2[i];
  md_num_t_x mdr = md_r_magnitude_x(dr);
  md_num_t_x mr1 = 0;
  md_num_t_x mr2 = 0;
  md_num_t_x rdot = 0;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    mr1 += r1[i]*r1[i];
  mr1 = sqrt(mr1);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    mr2 += r2[i]*r2[i];
  mr2 = sqrt(mr2);
  for (i = 0; i < MD_DIMENSION_X; ++i)
    rdot += r1[i]*r2[i];
  md_num_t_x ctheta = rdot/(mr1*mr2);
  md_num_t_x v = (mr1+mr2-beta/(mu*as))/sqrt(2*beta/mu);
  md_num_t_x res = 1+(beta/(mu*mr1*mr2))*exp(-(mu*mr1*mr2)*(1+ctheta)/beta)*(1+(1/as)*sqrt(M_PI*beta/(2*mu))*erfc(v)*exp(v*v));
  if (as > 0) {
    md_num_t_x E = -1/(2*mu*as*as);
    md_num_t_x pr1 = (1.0/sqrt(2*M_PI*as*mr1*mr1))*exp(-mr1/as);
    md_num_t_x pr2 = (1.0/sqrt(2*M_PI*as*mr2*mr2))*exp(-mr2/as);
    md_num_t_x rho = exp(-beta*E)*pr1*pr2/(exp(-mdr*mdr/(2*(1.0/mu)*beta))/pow(2*M_PI*(1.0/mu)*beta, MD_DIMENSION_X/2.0));
    res += rho;
  }
  return -log(res);
}

md_num_t_x md_calc_pa_pair_energy_x(md_pa_table_t_x *pa, md_num_t_x *r, md_num_t_x *r2) {
  if (pa->pa_func != NULL) {
    md_num_t_x beta_incre = MD_MIN_X(pa->beta/3.0, 0.01);
    md_num_t_x res = pa->pa_func(r, r2, pa->beta+beta_incre, pa->Npar, pa->params);
    md_num_t_x res2 = pa->pa_func(r, r2, pa->beta-beta_incre, pa->Npar, pa->params);
    return (res-res2)/(2*beta_incre);
  }
  md_num_t_x mr = md_r_magnitude_x(r);
  md_num_t_x mr2 = md_r_magnitude_x(r2);
  md_num_t_x dr[MD_DIMENSION_X];
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    dr[i] = r[i]-r2[i];
  md_num_t_x s = md_r_magnitude_x(dr);
  md_num_t_x q = (mr+mr2)/2.0;
  if (q < pa->r2[pa->ducount-1]) {
    md_num_t_x rincre = pa->r2[1]-pa->r2[0];
    int rindex = (int)(q/rincre);
    if (rindex > pa->ducount-2)
      rindex = pa->ducount-2;
    md_num_t_x frac = 1-(q-pa->r2[rindex])/rincre;
    md_num_t_x u = frac*pa->du[rindex]+(1-frac)*pa->du[rindex+1];
    md_num_t_x A = frac*pa->dA[rindex]+(1-frac)*pa->dA[rindex+1];
    return u+A*s*s;
  }
  md_num_t_x Z = pa->Z*pa->ka;
  return 0.5*Z*(1/mr+1/mr2);
}

md_num_t_x md_calc_pa_pair_U_x(md_pa_table_t_x *pa, md_num_t_x *r, md_num_t_x *r2) {
  if (pa->pa_func != NULL) {
    md_num_t_x res = pa->pa_func(r, r2, pa->beta, pa->Npar, pa->params);
    return res/pa->beta;
  }
  md_num_t_x mr = md_r_magnitude_x(r);
  md_num_t_x mr2 = md_r_magnitude_x(r2);
  md_num_t_x dr[MD_DIMENSION_X];
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    dr[i] = r[i]-r2[i];
  md_num_t_x s = md_r_magnitude_x(dr);
  md_num_t_x q = (mr+mr2)/2.0;
  if (q < pa->r[pa->ucount-1]) {
    md_num_t_x rincre = pa->r[1]-pa->r[0];
    int rindex = (int)(q/rincre);
    if (rindex > pa->ucount-2)
      rindex = pa->ucount-2;
    md_num_t_x frac = 1-(q-pa->r[rindex])/rincre;
    md_num_t_x u = frac*pa->u[rindex]+(1-frac)*pa->u[rindex+1];
    md_num_t_x A = frac*pa->A[rindex]+(1-frac)*pa->A[rindex+1];
    return (u+A*s*s)/pa->beta;
  }
  md_num_t_x Z = pa->Z*pa->ka;
  return 0.5*Z*(1/mr+1/mr2);
}

void md_calc_pa_deri_1_x(md_pa_table_t_x *pa, md_num_t_x *r, md_num_t_x *r2, md_num_t_x *dr, md_num_t_x rincre) {
  md_num_t_x tmpr[MD_DIMENSION_X];
  int i, j;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    for (j = 0; j < MD_DIMENSION_X; ++j)
      tmpr[j] = r[j];
    tmpr[i] = r[i]+rincre;
    md_num_t_x a = md_calc_pa_pair_U_x(pa, tmpr, r2);
    tmpr[i] = r[i]-rincre;
    md_num_t_x b = md_calc_pa_pair_U_x(pa, tmpr, r2);
    dr[i] = (a-b)/(2*rincre);
  }
}

void md_calc_pa_deri_2_x(md_pa_table_t_x *pa, md_num_t_x *r, md_num_t_x *r2, md_num_t_x *dr, md_num_t_x rincre) {
  md_num_t_x tmpr[MD_DIMENSION_X];
  int i, j;
  for (i = 0; i < MD_DIMENSION_X; ++i) {
    for (j = 0; j < MD_DIMENSION_X; ++j)
      tmpr[j] = r2[j];
    tmpr[i] = r2[i]+rincre;
    md_num_t_x a = md_calc_pa_pair_U_x(pa, r, tmpr);
    tmpr[i] = r2[i]-rincre;
    md_num_t_x b = md_calc_pa_pair_U_x(pa, r, tmpr);
    dr[i] = (a-b)/(2*rincre);
  }
}

int md_simulation_copy_to_x(md_simulation_t_x *sim, md_simulation_t_x *sim2) {
  if (sim->N != sim2->N || sim->Nf != sim2->Nf)
    return -1;
  int i, j;
  for (i = 0; i < sim->N; ++i) {
    for (j = 0; j < MD_DIMENSION_X; ++j) {
      sim->particles[i].x[j] = sim2->particles[i].x[j];
      sim->particles[i].v[j] = sim2->particles[i].v[j];
    }
    sim->particles[i].m = sim2->particles[i].m;
  }
  for (i = 0; i < sim->Nf; ++i) {
    for (j = 0; j < MD_NHC_LENGTH_X; ++j) {
      sim->nhcs[i].theta[j] = sim2->nhcs[i].theta[j];
      sim->nhcs[i].vtheta[j] = sim2->nhcs[i].vtheta[j];
      sim->nhcs[i].Q[j] = sim2->nhcs[i].Q[j];
    }
    sim->nhcs[i].f = sim2->nhcs[i].f;
  }
  return 0;
}