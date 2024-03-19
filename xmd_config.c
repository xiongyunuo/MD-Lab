#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Incorrect usage\n");
    return 1;
  }
  if (!strcmp(argv[1], "--xmd-flags")) {
#ifdef MD_USE_OPENCL_X
    printf("-DMD_USE_OPENCL_X ");
#endif
#ifdef MD_DOUBLE_PREC_X
    printf("-DMD_DOUBLE_PREC_X ");
#endif
  }
  else if (!strcmp(argv[1], "--xmd-libs")) {
#if defined(__APPLE__) && defined(MD_USE_POCL_X)
    printf("-lxmd_x -lOpenCL -lpthread");
#elif defined(__APPLE__)
    printf("-lxmd_x -lpthread -framework OpenCL");
#else
    printf("-lxmd_x -lOpenCL -lm -lpthread");
#endif
  }
  fflush(stdout);
  return 0;
}