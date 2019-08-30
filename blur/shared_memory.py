import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from skimage.io import imread, imsave


def kernel_function(filter_radius=3, active_w=16, active_h=16, active_d=3):
    func_str = f"""
__device__ __constant__ float d_Filter[{int(2*int(filter_radius)+1)**2}];

__global__ void convolution(float *d_Result, float *d_Data, int dataW, int dataH, int dataD) {{

  __shared__ float data[{active_w} + {filter_radius*2}][{active_h} + {filter_radius*2}][{active_d}];
  float sum = 0;

  // tile location
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // active block location
  int bx = threadIdx.x - {filter_radius};
  int by = threadIdx.y - {filter_radius};

  // image location
  int gx = (tx - {filter_radius}) + (blockDim.x - 2*{filter_radius}) * blockIdx.x;
  int gy = (ty - {filter_radius}) + (blockDim.y - 2*{filter_radius}) * blockIdx.y;
  int gz = tz;
  int gLoc = gz + (gx + gy * dataW) * dataD;

  // load data from global memory to shared memory
  if (0 <= gx && gx < dataW && 0 <= gy && gy < dataH)
    data[tx][ty][tz] = d_Data[gLoc];
  else
    data[tx][ty][tz] = 0;

  // wait all threads in the block
  __syncthreads();

  // convolve
  if (0 <= bx && bx < (blockDim.x - 2*{filter_radius}) && 0 <= by && (blockDim.y - 2*{filter_radius})) {{
    for (int i = {-1*filter_radius}; i <= {filter_radius}; i++) {{
      for (int j = {-1*filter_radius}; j <= {filter_radius}; j++) {{
        sum += data[tx + i][ty + j][tz] * d_Filter[{filter_radius} + i + ({filter_radius} + j) * {2*filter_radius+1}];
      }}
    }}
    d_Result[gLoc] = sum;
  }}
}}
"""
    return SourceModule(func_str)


if __name__ == "__main__":
    # load input image
    image = imread("lenna.png")
    image = (image / 255).astype(np.float32)

    # initialize output image
    image_blur = np.zeros_like(image)

    # initialize filter
    filter_radius = 5
    filter_size = 2*filter_radius + 1
    h_filter = np.ones((filter_size, filter_size), dtype=np.float32) / np.square(filter_size)

    # compile kernel 
    mod = kernel_function(filter_radius)
    convolve = mod.get_function("convolution")

    # copy constant from host to device
    d_filter, _ = mod.get_global("d_Filter")
    cuda.memcpy_htod(d_filter, h_filter)

    # copy variable from host to device
    active_w = 8
    active_h = 8
    block = (active_w + 2*filter_radius, active_h + 2*filter_radius, 3)
    grid = (
        int(np.ceil(image.shape[1]/active_w)),
        int(np.ceil(image.shape[0]/active_h)),
        int(np.ceil(image.shape[2]/block[2]))
    )
#     print(block, grid)
    convolve(cuda.Out(image_blur), cuda.In(image), np.int32(image.shape[1]), np.int32(image.shape[0]),
             np.int32(image.shape[2]), block=block, grid=grid)

    # dump image concat image_blur
    image_concat = np.concatenate([image, image_blur], axis=1)
    image_concat = (image_concat * 255).astype(np.uint8)
    imsave("lenna+blur_shared.png", image_concat)
