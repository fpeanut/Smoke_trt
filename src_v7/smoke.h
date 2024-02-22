#pragma once
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "engine.h"
#include "utils/config.h"


namespace smoke
{

    const int kInputH = 384;
    const int kInputW = 1280;
    const int kInputC = 3;
    const int kOutputH = kInputH / 4;
    const int kOutputW = kInputW / 4;
    const int kTopK = 100;
    const float kScoreThresh = 0.25f;
    const char *kInputTensorName = "input.1"; // 输入层名称
    const char *kOutputBBoxPreds = "1123";    // 输出，预测的边界框[100,8]
    const char *kOutputScores = "1125";       // 输出，预测的分数[100]
    const char *kOutputIndices = "1126";      // 输出，预测的类别[1,100]

    class SMOKE
    {
    public:
        SMOKE(const std::string &model_path,const cv::Mat trans_mat);
        ~SMOKE() = default;
        // ~SMOKE();
        std::vector<std::vector<cv::Point2f>> run(cv::Mat &img);
        void doInference();
        void preprocess(cv::Mat &img);
        std::vector<std::vector<cv::Point2f>> postprocess(const cv::Mat &img);

    private:
        std::shared_ptr<TrtEngine> engine_;
        cv::Mat intrinsic_;
        cv::Mat trans_mat_;
        std::vector<float> base_depth_;
        std::vector<Box3dDim> base_dims_;
        bool started_ = false;
    };

}
