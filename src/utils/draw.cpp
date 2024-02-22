#include "draw.h"

void draw_box3d(cv::Mat &img, std::vector<std::vector<cv::Point2f>> &corners_vec)
{
    for (auto img_corners : corners_vec)
    {
        for (int i = 0; i < 4; ++i)
        {
            const auto &p1 = img_corners[i];
            const auto &p2 = img_corners[(i + 1) % 4];
            const auto &p3 = img_corners[i + 4];
            const auto &p4 = img_corners[(i + 1) % 4 + 4];
            cv::line(img, p1, p2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::line(img, p3, p4, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::line(img, p1, p3, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
    }
}