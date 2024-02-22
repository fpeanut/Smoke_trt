#pragma once

#include <opencv2/opencv.hpp>

void draw_box3d(cv::Mat& img, std::vector<std::vector<cv::Point2f>>& img_corners);