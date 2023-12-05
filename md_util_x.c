#include "md_util_x.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

md_attr_pair_t_x *md_alloc_attr_pair_x() {
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
  md_num_t_x value;
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
    res->values[res->N-1] = value;
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
  if (res->fs == NULL) {
    free(res->particles);
    free(res);
    return NULL;
  }
  int i;
  for (i = 0; i < Nf; ++i) {
    res->fs[i] = f;
    if ((i+1)*f > MD_DIMENSION_X*N-fc)
      res->fs[i] = MD_DIMENSION_X*N-fc-i*f;
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

md_num_t_x md_random_gaussian_x() {
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
  int i, j;
  md_num_t_x mean[MD_DIMENSION_X];
  for (i = 0; i < sim->N; ++i)
    for (j = 0; j < MD_DIMENSION_X; ++j) {
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
  int i;
  for (i = start; i < end; ++i)
    sim->particles[i].m = m;
}

int md_init_particle_face_center_3d_lattice_pos_x(md_simulation_t_x *sim, md_num_t_x *center, md_num_t_x *length, md_num_t_x fluc, md_num_t_x eps) {
  if (MD_DIMENSION_X != 3)
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

void md_init_nhc_x(md_simulation_t_x *sim, md_num_t_x Q, int start, int end, md_num_t_x fp) {
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