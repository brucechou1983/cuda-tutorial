import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from skimage.io import imread, imsave


def kernel_function(filter_radius=3):
    func_str = f"""
__device__ __constant__ float d_Filter[{int(2*int(filter_radius)+1)**2}];

__global__ void convolution(float *d_Result, float *d_Data, int dataW, int dataH, int dataD ) {{
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;
  const int z = threadIdx.z + blockDim.z * blockIdx.z;
  const int gLoc = z + (x + y * dataW) * dataD ;

  float sum = 0;
  float value = 0;

  // check location
  if ( 0 <= x && x < dataW && 0 <= y && y < dataH && 0 <= z && z <= dataD ) {{
    for (int i = {-1*filter_radius}; i <= {filter_radius}; i++) {{
      for (int j = {-1*filter_radius}; j <= {filter_radius}; j++) {{
        if ((x + i) < 0)                       // left padding
          value = 0;
        else if ( (x + i) >= dataW )           // right padding
          value = 0;
        else {{
          if ((y + j) < 0)                     // top padding
            value = 0;
          else if ( (y + j) >= dataH )         // bottom padding
            value = 0;
          else {{
            value = d_Data[gLoc + (i + j * dataW) * dataD];
          }}
        }}
        sum += value * d_Filter[{filter_radius} + i + ({filter_radius} + j) * {2*filter_radius+1}];
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
    block = (16, 16, 3)
    grid = (
        int(np.ceil(image.shape[1]/block[0])),
        int(np.ceil(image.shape[0]/block[1])),
        int(np.ceil(image.shape[2]/block[2]))
    )
    convolve(cuda.Out(image_blur), cuda.In(image), np.int32(image.shape[1]), np.int32(image.shape[0]),
             np.int32(image.shape[2]), block=block, grid=grid)

    # dump image concat image_blur
    image_concat = np.concatenate([image, image_blur], axis=1)
    image_concat = (image_concat * 255).astype(np.uint8)
    imsave("lenna+blur_3d.png", image_concat)
