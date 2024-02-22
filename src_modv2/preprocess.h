#pragma once
#include <opencv2/opencv.hpp>

namespace apollo {
namespace perception {
namespace camera {


void SMOKEPreprocess(cv::Mat &img, int inputW, int inputH, float *buffer);

void cuda_preprocess(uint8_t *src, float *dst, int dst_width, int dst_height);
void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();

}  // namespace camera
}  // namespace perception
}  // namespace apollo
