// from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_bias_act_kernel.cu

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
static __global__ void fused_bias_act_kernel(
    scalar_t* out,
    const scalar_t* p_input,
    const scalar_t* p_bias,
    const scalar_t* p_refer,
    int n_channel,
    int n_flow,
    int act,
    int grad,
    scalar_t alpha,
    scalar_t scale
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n_channel * n_flow)
        return;

    const int c_index = index % n_channel;

    scalar_t bias = p_bias[c_index];
    scalar_t input = p_input[index];
    scalar_t refer = (p_refer) ? p_refer[index] : 0;

    scalar_t output;

    if (grad == 0) {
        output = input + bias;
    } else {
        output = input;
    }

    if (act == 1) {
        if (grad == 0) {
            output = (output > 0) ? output : output * alpha;
        } else {
            output = (refer > 0) ? output : output * alpha;
        }
    } else if (act == 3) {
        if (grad == 0) {
            output = (output > 0) ? output : output * alpha;
            output = output * scale;
        } else {
            output = (refer > 0) ? output * scale : output * alpha * scale;
        }
    }

    out[index] = output;
}

torch::Tensor fused_bias_act_op(const torch::Tensor& input,
                                const torch::Tensor& bias,
                                const torch::Tensor& refer,
                                int act, int grad, float alpha, float scale) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    auto out = torch::empty_like(input);

    int n_channel = input.size(1);
    int n_flow = input.size(0) * input.size(2) * input.size(3);
    int n_block = (n_channel * n_flow + 512 - 1) / 512;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_bias_act_kernel", [&] {
        fused_bias_act_kernel<scalar_t><<<n_block, 512, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            refer.data_ptr<scalar_t>(),
            n_channel,
            n_flow,
            act,
            grad,
            static_cast<scalar_t>(alpha),
            static_cast<scalar_t>(scale)
        );
    });

    return out;
}