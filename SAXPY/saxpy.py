import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


def kernel_function():
    mod = SourceModule("""
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
""")
    return mod.get_function("saxpy")


if __name__ == "__main__":
    # initialize variables
    n = 2**20
    a = 2.0
    x = np.ones((n), dtype=np.float32)
    y = np.ones((n), dtype=np.float32) * 2.0
    y_gpu = y.copy()

    # compile kernel
    saxpy = kernel_function()

    # copy variable from host to device
    saxpy(np.int32(n), np.float32(a), cuda.In(x), cuda.InOut(y_gpu), block=(256,1,1), grid=(int(np.ceil(n/256)),1,1))

    # check answer
    y_cpu = 2.0 * x + y

    print(f"Max error: {np.max(y_gpu-y_cpu)}")
