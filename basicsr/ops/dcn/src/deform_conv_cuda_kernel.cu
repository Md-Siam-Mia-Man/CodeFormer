#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

using namespace at;

template <typename scalar_t>
__device__ scalar_t deformable_im2col_bilinear(const scalar_t *input,
                                               const int data_width,
                                               const int height,
                                               const int width, scalar_t h,
                                               scalar_t w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh;
  scalar_t hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = input[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = input[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = input[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = input[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t arg, scalar_t h, scalar_t w,
                                        const int height, const int width) {
  if (arg <= -1 || arg >= height || w <= -1 || w >= width) {
    // empty
    return 0;
  }

  int arg_low = floor(arg);
  int arg_high = arg_low + 1;

  scalar_t weight = 0;
  if (arg_low >= 0 && arg_low <= height - 1)
    weight += (1 - (arg - arg_low));
  if (arg_high >= 0 && arg_high <= height - 1)
    weight += (1 - (arg_high - arg));

  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(scalar_t arg, scalar_t h, scalar_t w,
                                          const int height, const int width) {
  if (arg <= -1 || arg >= height || w <= -1 || w >= width) {
    // empty
    return 0;
  }

  int arg_low = floor(arg);
  int arg_high = arg_low + 1;

  scalar_t weight = 0;
  if (arg_low >= 0 && arg_low <= height - 1)
    weight += (-1 * (1 - (arg - arg_low)));
  if (arg_high >= 0 && arg_high <= height - 1)
    weight += (1 * (1 - (arg_high - arg)));

  return weight;
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
    const int n, const scalar_t *data_im, const scalar_t *data_offset,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col,
    const int width_col, scalar_t *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    scalar_t *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    // const scalar_t* data_im_ptr = data_im + (b_col * num_channels + c_im) *
    // height * width;
    const scalar_t *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t *data_offset_ptr =
        data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          // const scalar_t map_h = i * dilation_h + offset_h;
          // const scalar_t map_w = j * dilation_w + offset_w;
          // const int cur_height = height - h_in;
          // const int cur_width = width - w_in;
          // val = deformable_im2col_bilinear(data_im_ptr, width, cur_height,
          // cur_width, map_h, map_w);
          val = deformable_im2col_bilinear(data_im_ptr, width, height, width,
                                           h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_offset,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    scalar_t *grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t *data_offset_ptr =
        data_offset +
        (b * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight =
              get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy,
                                  cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    scalar_t *grad_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % (2 * kernel_h * kernel_w);
    int b = (index / width_col / height_col) / (2 * kernel_h * kernel_w);
    // compute the start and end of the output

    const int deformable_group_index = b % deformable_group;
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr =
        data_col +
        deformable_group_index * channel_per_deformable_group * batch_size *
            width_col * height_col;
    const scalar_t *data_im_ptr =
        data_im +
        (b * deformable_group + deformable_group_index) *
            channel_per_deformable_group / deformable_group * height * width;
    const scalar_t *data_offset_ptr =
        data_offset +
        (b * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;

    const int offset_c = c - cnt * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      }
      const scalar_t weight = get_coordinate_weight(
          inv_h, inv_w, height, width, data_im_ptr, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
    }

    grad_offset[index] = val;
  }
}

void deformable_im2col(const at::Tensor data_im, const at::Tensor data_offset,
                       const int channels, const int height, const int width,
                       const int ksize_h, const int ksize_w, const int pad_h,
                       const int pad_w, const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       at::Tensor data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * data_col.size(2) * data_col.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       CUDA_NUM_THREADS, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, data_offset_, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, parallel_imgs, channels,
            deformable_group, data_col.size(2), data_col.size(3), data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
  }
}

void deformable_col2im(const at::Tensor data_col, const at::Tensor data_offset,
                       const int channels, const int height, const int width,
                       const int ksize_h, const int ksize_w, const int pad_h,
                       const int pad_w, const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       at::Tensor grad_im) {
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels =
      channels * ksize_h * ksize_w * data_col.size(2) * data_col.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       CUDA_NUM_THREADS, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_offset_, channels, height, width,
            ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
            dilation_w, channel_per_deformable_group, parallel_imgs,
            deformable_group, data_col.size(2), data_col.size(3), grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
  }
}

void deformable_col2im_coord(
    const at::Tensor data_col, const at::Tensor data_im,
    const at::Tensor data_offset, const int channels, const int height,
    const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, at::Tensor grad_offset) {
  const int num_kernels = data_offset.size(1) * data_offset.size(2) *
                          data_offset.size(3) * parallel_imgs;
  const int channel_per_deformable_group =
      channels * parallel_imgs / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();

        deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                             CUDA_NUM_THREADS, 0,
                                             at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_im_, data_offset_, channels, height,
            width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group, parallel_imgs,
            deformable_group, data_col.size(2), data_col.size(3), grad_offset_);
      }));
}

template <typename scalar_t>
__device__ scalar_t dmcn_im2col_bilinear(const scalar_t *bottom_data,
                                         const int data_width, const int height,
                                         const int width, scalar_t h,
                                         scalar_t w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh;
  scalar_t hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_gradient_weight(scalar_t arg, scalar_t h,
                                             scalar_t w, const int height,
                                             const int width) {
  if (arg <= -1 || arg >= height || w <= -1 || w >= width) {
    // empty
    return 0;
  }

  int arg_low = floor(arg);
  int arg_high = arg_low + 1;

  scalar_t weight = 0;
  if (arg_low >= 0 && arg_low <= height - 1)
    weight += (1 - (arg - arg_low));
  if (arg_high >= 0 && arg_high <= height - 1)
    weight += (1 - (arg_high - arg));

  return weight;
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_coordinate_weight(scalar_t arg, scalar_t h,
                                               scalar_t w, const int height,
                                               const int width,
                                               const scalar_t *im_data,
                                               const int data_width,
                                               const int bp_dir) {
  if (arg <= -1 || arg >= height || w <= -1 || w >= width) {
    // empty
    return 0;
  }

  int arg_low = floor(arg);
  int arg_high = arg_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0) {
    if (arg_low >= 0 && arg_low <= height - 1)
      weight += (-1 * (1 - (arg - arg_low)) *
                 im_data[arg_low * data_width + (int)w]);
    if (arg_high >= 0 && arg_high <= height - 1)
      weight +=
          (1 * (1 - (arg_high - arg)) * im_data[arg_high * data_width + (int)w]);
  } else {
    if (arg_low >= 0 && arg_low <= height - 1)
      weight += (-1 * (1 - (arg - arg_low)) *
                 im_data[(int)h * data_width + arg_low]);
    if (arg_high >= 0 && arg_high <= height - 1)
      weight +=
          (1 * (1 - (arg_high - arg)) * im_data[(int)h * data_width + arg_high]);
  }

  return weight;
}

template <typename scalar_t>
__global__ void modulated_deformable_im2col_gpu_kernel(
    const int n, const scalar_t *data_im, const scalar_t *data_offset,
    const scalar_t *data_mask, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int height_col, const int width_col, scalar_t *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    scalar_t *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    // const scalar_t* data_im_ptr = data_im + (b_col * num_channels + c_im) *
    // height * width;
    const scalar_t *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t *data_offset_ptr =
        data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const scalar_t *data_mask_ptr =
        data_mask +
        (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;

        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        const scalar_t mask = data_mask_ptr[data_mask_hw_ptr];

        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          // const scalar_t map_h = i * dilation_h + offset_h;
          // const scalar_t map_w = j * dilation_w + offset_w;
          // const int cur_height = height - h_in;
          // const int cur_width = width - w_in;
          // val = deformable_im2col_bilinear(data_im_ptr, width, cur_height,
          // cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im,
                                     w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename scalar_t>
__global__ void modulated_deformable_col2im_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_offset,
    const scalar_t *data_mask, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    scalar_t *grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t *data_offset_ptr =
        data_offset +
        (b * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const scalar_t *data_mask_ptr =
        data_mask +
        (b * deformable_group + deformable_group_index) * kernel_h * kernel_w *
            height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;

    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t mask = data_mask_ptr[data_mask_hw_ptr];
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight =
              dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data,
                                       cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void modulated_deformable_col2im_coord_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_im,
    const scalar_t *data_offset, const scalar_t *data_mask,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    scalar_t *grad_offset, scalar_t *grad_mask) {
  CUDA_KERNEL_LOOP(index, n) {
    scalar_t val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % (2 * kernel_h * kernel_w);
    int b = (index / width_col / height_col) / (2 * kernel_h * kernel_w);
    // compute the start and end of the output

    const int deformable_group_index = b % deformable_group;
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr =
        data_col +
        deformable_group_index * channel_per_deformable_group * batch_size *
            width_col * height_col;
    const scalar_t *data_im_ptr =
        data_im +
        (b * deformable_group + deformable_group_index) *
            channel_per_deformable_group / deformable_group * height * width;
    const scalar_t *data_offset_ptr =
        data_offset +
        (b * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const scalar_t *data_mask_ptr =
        data_mask +
        (b * deformable_group + deformable_group_index) * kernel_h * kernel_w *
            height_col * width_col;

    const int offset_c = c - cnt * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const int data_mask_hw_ptr =
          (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      const scalar_t mask = data_mask_ptr[data_mask_hw_ptr];
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      }
      const scalar_t weight = dmcn_get_coordinate_weight(
          inv_h, inv_w, height, width, data_im_ptr, width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      const scalar_t val_ = dmcn_im2col_bilinear(data_im_ptr, width, height,
                                                 width, inv_h, inv_w);
      mval += val_ * data_col_ptr[col_pos];
    }

    grad_offset[index] = val;
    if (offset_c % 2 == 1)
      // k = (2 * (i * kernel_w + j) + 1)
      // k / 2 = i * kernel_w + j
      grad_mask[((b * deformable_group + deformable_group_index) * kernel_h *
                     kernel_w +
                 offset_c / 2) *
                    height_col * width_col +
                h * width_col + w] = mval;
  }
}

void modulated_deformable_im2col_cuda(
    const at::Tensor data_im, const at::Tensor data_offset,
    const at::Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    at::Tensor data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * data_col.size(2) * data_col.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "modulated_deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        modulated_deformable_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, data_offset_, data_mask_, height_im,
            width_im, kernel_h, kenerl_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group, batch_size,
            channels, deformable_group, height_col, width_col, data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in modulated_deformable_im2col: %s\n",
           cudaGetErrorString(err));
  }
}

void modulated_deformable_col2im_cuda(
    const at::Tensor data_col, const at::Tensor data_offset,
    const at::Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    at::Tensor grad_im) {
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels =
      channels * kernel_h * kenerl_w * data_col.size(2) * data_col.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        modulated_deformable_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_offset_, data_mask_, channels,
            height_im, width_im, kernel_h, kenerl_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, deformable_group, height_col, width_col, grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in modulated_deformable_col2im: %s\n",
           cudaGetErrorString(err));
  }
}

void modulated_deformable_col2im_coord_cuda(
    const at::Tensor data_col, const at::Tensor data_im,
    const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kenerl_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, at::Tensor grad_offset,
    at::Tensor grad_mask) {
  const int num_kernels = data_offset.size(1) * data_offset.size(2) *
                          data_offset.size(3) * batch_size;
  const int channel_per_deformable_group =
      channels * batch_size / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();
        scalar_t *grad_mask_ = grad_mask.data_ptr<scalar_t>();

        modulated_deformable_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_im_, data_offset_, data_mask_,
            channels, height_im, width_im, kernel_h, kenerl_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, batch_size, deformable_group,
            height_col, width_col, grad_offset_, grad_mask_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in modulated_deformable_col2im_coord: %s\n",
           cudaGetErrorString(err));
  }
}