#include"smoke.h"
#include<fstream>
#include<memory>
#include<NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include"preprocess.h"
// #include"trt_modulated_deform_conv.h"
#include"plugin/trt_modulated_deform_conv.h"

namespace apollo {
namespace perception {
namespace camera {

const int kInputH = 384;
const int kInputW = 1280;
const int kInputC = 3;
const int kOutputH = kInputH / 4;
const int kOutputW = kInputW / 4;
const int kTopK = 100;
const float kScoreThresh = 0.25f;
// const char *kInputTensorName = "input.1"; // 输入层名称
// const char *kOutputBBoxPreds = "1123";    // 输出，预测的边界框[100,8]
// const char *kOutputScores = "1125";       // 输出，预测的分数[100]
// const char *kOutputIndices = "1126";      // 输出，预测的类别[1,100]

float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

SMOKE::SMOKE(const std::string& engine_path, 
        const cv::Mat& trans_mat){

    base_depth_ = {28.01f, 16.32f};
    base_dims_.resize(3); // pedestrian, cyclist, car
    base_dims_[0].x = 0.88f;
    base_dims_[0].y = 1.73f;
    base_dims_[0].z = 0.67f;
    base_dims_[1].x = 1.78f;
    base_dims_[1].y = 1.70f;
    base_dims_[1].z = 0.58f;
    base_dims_[2].x = 3.88f;
    base_dims_[2].y = 1.63f;
    base_dims_[2].z = 1.53f;        

    intrinsic_ = (cv::Mat_<float>(3, 3) << 1068.27, 0.0, 973.472, 0.0, 1066.57, 545.227, 0.0, 0.0, 1.0);
    trans_mat_=(cv::Mat_<float>(3,3)<<trans_mat.at<double>(0,0),trans_mat.at<double>(0,1),trans_mat.at<double>(0,2),
                                    trans_mat.at<double>(1,0),trans_mat.at<double>(1,1),trans_mat.at<double>(1,2),
                                    0,0,1); 
    buffer_size_[0] = kInputC * kInputH * kInputW;
    buffer_size_[1] = kTopK * 8;
    buffer_size_[2] = kTopK;
    buffer_size_[3] = kTopK;

    cudaMalloc(&buffers_[0], buffer_size_[0] * sizeof(float));
    cudaMalloc(&buffers_[1], buffer_size_[1] * sizeof(float));
    cudaMalloc(&buffers_[2], buffer_size_[2] * sizeof(float));
    cudaMalloc(&buffers_[3], buffer_size_[3] * sizeof(float));

    image_data_.resize(buffer_size_[0]);
    bbox_preds_.resize(buffer_size_[1]);
    topk_scores_.resize(buffer_size_[2]);
    topk_indices_.resize(buffer_size_[3]);

    cudaStreamCreate(&stream_);
    LoadEngine(engine_path);
    
}

SMOKE::~SMOKE() {
    cudaStreamDestroy(stream_);
    for (auto& buffer : buffers_) {
        cudaFree(buffer);
    }
    if (context_ != nullptr) {
        context_->destroy();
        engine_->destroy();
    }
    cuda_preprocess_destroy(); 
}

void SMOKE::LoadEngine(const std::string& engine_path) {
    std::ifstream in_file(engine_path, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    in_file.seekg(0, in_file.end);
    int length = in_file.tellg();
    in_file.seekg(0, in_file.beg);
    std::unique_ptr<char[]> trt_model_stream(new char[length]);
    in_file.read(trt_model_stream.get(), length);
    in_file.close();

    // getPluginCreator could not find plugin: MMCVModulatedDeformConv2d version: 1
    initLibNvInferPlugins(&g_logger_, "");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_);
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trt_model_stream.get(), length, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);

    runtime->destroy();
}

void SMOKE::Detect(cv::Mat& raw_img,cv::Mat &dst) {

    // // // Preprocessing
    // float mean[3] {123.675f, 116.280f, 103.530f};
    // float std[3] = {58.395f, 57.120f, 57.375f};
    // uint8_t* data_hwc = reinterpret_cast<uint8_t*>(dst.data);
    // float* data_chw = image_data_.data();
    // for (int c = 0; c < 3; ++c) {
    //     for (unsigned j = 0, img_size = dst.rows * dst.cols; j < img_size; ++j) {
    //         data_chw[c * img_size + j] = (data_hwc[j * 3 + 2 - c] - mean[c]) / std[c];  //bgr2rgb
    //     }
    // }
    cuda_preprocess_init(dst.rows*dst.cols);
    SMOKEPreprocess(dst,dst.cols,dst.rows,static_cast<float*>(buffers_[0]));

    // Do inference
    // cudaMemcpyAsync(buffers_[0], image_data_.data(), buffer_size_[0] * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->executeV2(&buffers_[0]);
    cudaMemcpyAsync(bbox_preds_.data(), buffers_[1], buffer_size_[1] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(topk_scores_.data(), buffers_[2], buffer_size_[2] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(topk_indices_.data(), buffers_[3], buffer_size_[3] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    // Decoding and visualization
    PostProcess(raw_img);
}

void SMOKE::PostProcess(cv::Mat& input_img) {
    for (int i = 0; i < kTopK; ++i) {
        if (topk_scores_[i] < kScoreThresh) {
            continue;
        }
        // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/coders/smoke_bbox_coder.py#L52
        int class_id = static_cast<int>(topk_indices_[i] / kOutputH / kOutputW);
        int location = static_cast<int>(topk_indices_[i]) % (kOutputH * kOutputW);
        int img_x = location % kOutputW;
        int img_y = location / kOutputW;
        // Depth
        float z = base_depth_[0] + bbox_preds_[8*i] * base_depth_[1];
        // std::cout<<z<<";";
        // Location
        cv::Mat img_point(3, 1, CV_32FC1);
        img_point.at<float>(0) =(static_cast<float>(img_x) + bbox_preds_[8*i + 1]);
        img_point.at<float>(1) =(static_cast<float>(img_y) + bbox_preds_[8*i + 2]);
        img_point.at<float>(2) = 1.0f;
        cv::Mat trams_point=trans_mat_.inv()*img_point;
        cv::Mat cam_point = intrinsic_.inv() * trams_point * z;
        float x = cam_point.at<float>(0);
        float y = cam_point.at<float>(1);
        // Dimension
        float w = base_dims_[class_id].x * expf(Sigmoid(bbox_preds_[8*i + 3]) - 0.5f);
        float l = base_dims_[class_id].y * expf(Sigmoid(bbox_preds_[8*i + 4]) - 0.5f);
        float h = base_dims_[class_id].z * expf(Sigmoid(bbox_preds_[8*i + 5]) - 0.5f);
        // Orientation
        float ori_norm = sqrtf(powf(bbox_preds_[8*i + 6], 2.0f) + powf(bbox_preds_[8*i + 7], 2.0f));
        bbox_preds_[8*i + 6] /= ori_norm;  //sin(alpha)
        bbox_preds_[8*i + 7] /= ori_norm;  //cos(alpha)
        float ray = atan(x / (z + 1e-7f));
        float alpha = atan(bbox_preds_[8*i + 6] / (bbox_preds_[8*i + 7] + 1e-7f));
        if (bbox_preds_[8*i + 7] > 0.0f) {
            alpha -= M_PI / 2.0f;
        } else {
            alpha += M_PI / 2.0f;
        }
        float angle = alpha + ray;
        if (angle > M_PI) {
            angle -= 2.0f * M_PI;
        } else if (angle < -M_PI) {
            angle += 2.0f * M_PI;
        }

        // std::cout<<"result:"<<x<<";"<<y<<";"<<z<<";"<<w<<";"<<l<<";"<<h<<";"<<angle<<std::endl;

        // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py#L117
        //              front z
        //                   /
        //                  /
        //    (x0, y0, z1) + -----------  + (x1, y0, z1)
        //                /|            / |
        //               / |           /  |
        // (x0, y0, z0) + ----------- +   + (x1, y1, z1)
        //              |  /      .   |  /
        //              | / origin    | /
        // (x0, y1, z0) + ----------- + -------> x right
        //              |             (x1, y1, z0)
        //              |
        //              v
        //         down y
        cv::Mat cam_corners = (cv::Mat_<float>(8, 3) << 
            -w, -l, -h,     // (x0, y0, z0)
            -w, -l,  h,     // (x0, y0, z1)
            -w,  l,  h,     // (x0, y1, z1)
            -w,  l, -h,     // (x0, y1, z0)
             w, -l, -h,     // (x1, y0, z0)
             w, -l,  h,     // (x1, y0, z1)
             w,  l,  h,     // (x1, y1, z1)
             w,  l, -h);    // (x1, y1, z0)
        cam_corners = 0.5f * cam_corners;
        cv::Mat rotation_y = cv::Mat::eye(3, 3, CV_32FC1);
        rotation_y.at<float>(0, 0) = cosf(angle);
        rotation_y.at<float>(0, 2) = sinf(angle);
        rotation_y.at<float>(2, 0) = -sinf(angle);
        rotation_y.at<float>(2, 2) = cosf(angle);
        // cos, 0, sin
        //   0, 1,   0
        //-sin, 0, cos
        cam_corners = cam_corners * rotation_y.t();
        for (int i = 0; i < 8; ++i) {
            cam_corners.at<float>(i, 0) += x;
            cam_corners.at<float>(i, 1) += y;
            cam_corners.at<float>(i, 2) += z;
        }
        cam_corners = cam_corners * intrinsic_.t();
        std::vector<cv::Point2f> img_corners(8);
        for (int i = 0; i < 8; ++i) {
            img_corners[i].x = cam_corners.at<float>(i, 0) / cam_corners.at<float>(i, 2);
            img_corners[i].y = cam_corners.at<float>(i, 1) / cam_corners.at<float>(i, 2);
        }
        // std::cout<<img_corners[0].x<<";";
        for (int i = 0; i < 4; ++i) {
            const auto& p1 = img_corners[i];
            const auto& p2 = img_corners[(i + 1) % 4];
            const auto& p3 = img_corners[i + 4];
            const auto& p4 = img_corners[(i + 1) % 4 + 4];
            cv::line(input_img, p1, p2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::line(input_img, p3, p4, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::line(input_img, p1, p3, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
    }
    // cv::imwrite("./test.jpg",input_img);
    // cv::waitKey(0);
}

}  // namespace camera
}  // namespace perception
}  // namespace apollo