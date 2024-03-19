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

void md_init_maxwell_vel_x(md_simulation_t_x *sim, md_num_t_x T) {
  md_simulation_sync_host_x(sim, 0);
  int i, j;
  md_num_t_x mean[MD_DIMENSION_X];
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
}

void md_init_particle_mass_x(md_simulation_t_x *sim, md_num_t_x m, int start, int end) {
  md_simulation_sync_host_x(sim, 0);
  int i;
  for (i = start; i < end; ++i)
    sim->particles[i].m = m;
}

int md_init_particle_face_center_3d_lattice_pos_x(md_simulation_t_x *sim, md_num_t_x *center, md_num_t_x *length, md_num_t_x fluc, md_num_t_x eps) {
  if (MD_DIMENSION_X != 3)
    return -1;
  md_simulation_sync_host_x(sim, 0);
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

void md_init_nhc_x(md_simulation_t_x *sim, md_num_t_x Q, int start, int end, md_num_t_x fp) {
  md_simulation_sync_host_x(sim, 0);
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
}

void md_init_nd_rect_box_x(simulation_box_t_x *box, md_num_t_x *L) {
  if (box->type != MD_ND_RECT_BOX_X)
    return;
  nd_rect_t_x *rect = (nd_rect_t_x *)box->box;
  int i;
  for (i = 0; i < MD_DIMENSION_X; ++i)
    rect->L[i] = L[i];
}

int md_fprint_particle_pos_x(FILE *out, md_simulation_t_x *sim) {
  fprintf(out, "x %d %d\n", sim->N, MD_DIMENSION_X);
  md_simulation_sync_host_x(sim, 1);
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
  md_simulation_sync_host_x(sim, 1);
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
  md_simulation_sync_host_x(sim, 1);
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
  md_simulation_sync_host_x(sim, 1);
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