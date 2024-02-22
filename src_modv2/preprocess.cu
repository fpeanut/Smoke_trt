#include "preprocess.h"

#include <cuda_runtime_api.h>

namespace apollo {
namespace perception {
namespace camera {

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess)                                                         \
        {                                                                                      \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif // CUDA_CHECK

static uint8_t *img_buffer_device = nullptr;

struct AffineMatrix
{
    float value[6];
};

void cuda_preprocess_init(int max_image_size)
{
    // prepare input data in device memory
    CUDA_CHECK(cudaMalloc((void **)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy()
{
    CUDA_CHECK(cudaFree(img_buffer_device));
}

// 一个线程处理一个像素点
__global__ void preprocess_kernel(
    uint8_t *src, float *dst, int dst_width,
    int dst_height, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge)
        return;

    int dx = position % dst_width; // 计算当前线程对应的目标图像的x坐标
    int dy = position / dst_width; // 计算当前线程对应的目标图像的y坐标

    // normalization（对原图中(x,y)坐标的像素点3个通道进行归一化）
    float c0 = src[dy * dst_width * 3 + dx * 3 + 0];
    float c1 = src[dy * dst_width * 3 + dx * 3 + 1];
    float c2 = src[dy * dst_width * 3 + dx * 3 + 2];

    // bgr to rgb
    float t = c2;
    c2 = c0;
    c0 = t;

    // rgbrgbrgb to rrrgggbbb
    // NHWC to NCHW
    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dy * dst_width + dx;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;
    // *pdst_c0 = c0;
    // *pdst_c1 = c1;
    // *pdst_c2 = c2;
    *pdst_c0 = (c0-123.675)/58.395;
    *pdst_c1 = (c1-116.28)/57.12;
    *pdst_c2 = (c2-103.53)/57.375;
}

// GPU做归一化、BGR2RGB、NHWC to NCHW
void cuda_preprocess(
    uint8_t *src, float *dst, int dst_width, int dst_height)
{

    int img_size = dst_width * dst_height * 3;
    CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));

    // 一个线程处理一个像素点，一共需要 dst_height * dst_width 个线程
    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);


    preprocess_kernel<<<blocks, threads>>>(
        img_buffer_device, dst, dst_width, dst_height, jobs);
}

void SMOKEPreprocess(cv::Mat &img, int inputW, int inputH, float *buffer)
{
    cuda_preprocess(img.ptr(), buffer, inputW, inputH);
}

}  // namespace camera
}  // namespace perception
}  // namespace apollo
