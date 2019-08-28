#include <stdio.h>

// kernel code
// __global__: could be called by CPU and runs on GPU
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

// host code
int main(void) {
  // number of elements to be processed
  int N = 1<<20;

  float *x, *y, *d_x, *d_y;

  // allocate host memory
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  // initialize host operands
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // allocate device memory
  // d_ prefix mean on device memory
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  // copy x, y to d_x, d_y respectively from host to device
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on N elements
  // host process instructs the device how to execute the kernel function by assigning:
  // <<< # of blocks/grid, # of threads/block >>>
  // if N == 1, blocks = 1. N == 0, blocks = 0
  // y = 2.0f * x + y = 4.0f
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  // copy d_y to y from device to host
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  // check answer
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  // release memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
