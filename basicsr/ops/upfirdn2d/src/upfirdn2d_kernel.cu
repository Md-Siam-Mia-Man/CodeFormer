// from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn2d_kernel.cu

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
static __global__ void upfirdn2d_kernel(
    scalar_t* out,
    const scalar_t* input,
    const scalar_t* kernel,
    int n,
    int in_h,
    int in_w,
    int k_h,
    int k_w,
    int up_x,
    int up_y,
    int down_x,
    int down_y,
    int pad_x0,
    int pad_x1,
    int pad_y0,
    int pad_y1,
    int out_h,
    int out_w
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_w || y >= out_h)
        return;

    const int in_x = x * down_x - pad_x0;
    const int in_y = y * down_y - pad_y0;

    scalar_t v = 0.0;

    for (int j = 0; j < k_h; j++) {
        const int ky = in_y + j;
        if (ky % up_y == 0) {
            for (int i = 0; i < k_w; i++) {
                const int kx = in_x + i;
                if (kx % up_x == 0) {
                    const int real_y = ky / up_y;
                    const int real_x = kx / up_x;

                    if (real_x >= 0 && real_x < in_w && real_y >= 0 && real_y < in_h) {
                        v += input[real_y * in_w + real_x] * kernel[j * k_w + i];
                    }
                }
            }
        }
    }

    out[y * out_w + x] = v;
}

template <typename scalar_t>
static __global__ void upfirdn2d_kernel_large(
    scalar_t* out,
    const scalar_t* input,
    const scalar_t* kernel,
    int n,
    int in_h,
    int in_w,
    int k_h,
    int k_w,
    int up_x,
    int up_y,
    int down_x,
    int down_y,
    int pad_x0,
    int pad_x1,
    int pad_y0,
    int pad_y1,
    int out_h,
    int out_w,
    int tile_h,
    int tile_w
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_w || y >= out_h)
        return;

    const int in_x = x * down_x - pad_x0;
    const int in_y = y * down_y - pad_y0;

    scalar_t v = 0.0;

    for (int j = 0; j < k_h; j++) {
        const int ky = in_y + j;
        if (ky % up_y == 0) {
            for (int i = 0; i < k_w; i++) {
                const int kx = in_x + i;
                if (kx % up_x == 0) {
                    const int real_y = ky / up_y;
                    const int real_x = kx / up_x;

                    if (real_x >= 0 && real_x < in_w && real_y >= 0 && real_y < in_h) {
                        v += input[real_y * in_w + real_x] * kernel[j * k_w + i];
                    }
                }
            }
        }
    }

    out[y * out_w + x] = v;
}

torch::Tensor upfirdn2d_op(const torch::Tensor& input, const torch::Tensor& kernel,
                           int up_x, int up_y, int down_x, int down_y,
                           int pad_x0, int pad_x1, int pad_y0, int pad_y1) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int n = input.size(0);
    int channel = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int k_h = kernel.size(0);
    int k_w = kernel.size(1);

    int out_h = (in_h * up_y + pad_y0 + pad_y1 - k_h + down_y) / down_y;
    int out_w = (in_w * up_x + pad_x0 + pad_x1 - k_w + down_x) / down_x;

    auto out = torch::empty({n, channel, out_h, out_w}, input.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upfirdn2d_kernel", [&] {
        dim3 blocks((out_w + 31) / 32, (out_h + 31) / 32);
        dim3 threads(32, 32);

        for (int i = 0; i < n * channel; ++i) {
            upfirdn2d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>() + i * out_h * out_w,
                input.data_ptr<scalar_t>() + i * in_h * in_w,
                kernel.data_ptr<scalar_t>(),
                n,
                in_h,
                in_w,
                k_h,
                k_w,
                up_x,
                up_y,
                down_x,
                down_y,
                pad_x0,
                pad_x1,
                pad_y0,
                pad_y1,
                out_h,
                out_w
            );
        }
    });

    return out;
}