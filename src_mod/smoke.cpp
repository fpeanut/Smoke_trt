#include "smoke.h"

#include "utils/preprocess.h"
#include "utils/config.h"
namespace apollo {
namespace perception {
namespace camera {



namespace smoke
{

    float Sigmoid(float x)
    {
        return 1.0f / (1.0f + expf(-x));
    }

    SMOKE::SMOKE(const std::string &model_path,const cv::Mat trans_mat)
    {
        engine_.reset(new TrtEngine(model_path));

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
        // 相机内参，分别是fx：721.5377，fy：721.5377，cx：609.5593，cy：172.854
        // intrinsic_ = (cv::Mat_<float>(3, 3) << 721.5377, 0.0, 609.5593, 0.0, 721.5377, 172.854, 0.0, 0.0, 1.0); // kitti相机内参
        intrinsic_ = (cv::Mat_<float>(3, 3) << 1068.27, 0.0, 973.472, 0.0, 1066.57, 545.227, 0.0, 0.0, 1.0); // 测试相机内参
        // intrinsic_ = (cv::Mat_<float>(3, 3) << 809.2210, 0.0, 829.2196, 0.0, 809.2210, 481.7784, 0.0, 0.0, 1.0); // nuscense相机内参
        trans_mat_=(cv::Mat_<float>(3,3)<<trans_mat.at<double>(0,0),trans_mat.at<double>(0,1),trans_mat.at<double>(0,2),
                                            trans_mat.at<double>(1,0),trans_mat.at<double>(1,1),trans_mat.at<double>(1,2),
                                            0,0,1);
        // std::cout<<"trans_mat_:"<<trans_mat_<<std::endl;
    }

    std::vector<std::vector<cv::Point2f>> SMOKE::run(cv::Mat &img)
    {
        preprocess(img);
        doInference();
        return postprocess(img);
    }

    void SMOKE::doInference()
    {
        engine_->doInference();
    }

    void SMOKE::preprocess(cv::Mat &img)
    {
        SMOKEPreprocess(img, kInputW, kInputH, (float *)engine_->getDeviceBuffer(kInputTensorName));
    }

    std::vector<std::vector<cv::Point2f>> SMOKE::postprocess(const cv::Mat &img)
    {

        if (!started_)
        {
            // Modify camera intrinsics due to scaling 调整相机内参
            intrinsic_.at<float>(0, 0) *= static_cast<float>(kInputW) / img.cols;
            intrinsic_.at<float>(0, 2) *= static_cast<float>(kInputW) / img.cols;
            intrinsic_.at<float>(1, 1) *= static_cast<float>(kInputH) / img.rows;
            intrinsic_.at<float>(1, 2) *= static_cast<float>(kInputH) / img.rows;
            started_ = true;
        }

        static std::vector<float> bbox_preds(kTopK * 8); // 预测的边界框，[100,8]
        static std::vector<float> topk_scores(kTopK);    // 预测的分数，[100]
        static std::vector<float> topk_indices(kTopK);   // 预测的类别，[1,100]

        memcpy(bbox_preds.data(), engine_->getHostBuffer(kOutputBBoxPreds), kTopK * 8 * sizeof(float));
        memcpy(topk_scores.data(), engine_->getHostBuffer(kOutputScores), kTopK * sizeof(float));
        memcpy(topk_indices.data(), engine_->getHostBuffer(kOutputIndices), kTopK * sizeof(float));

        std::vector<std::vector<cv::Point2f>> corners_vec; // 用来存储边界框的8个顶点坐标

        for (int i = 0; i < kTopK; ++i) // 遍历每个预测的边界框
        {

            if (topk_scores[i] < kScoreThresh) // 如果分数小于阈值，则跳过
            {
                continue;
            }
            // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/coders/smoke_bbox_coder.py#L52
            int class_id = static_cast<int>(topk_indices[i] / kOutputH / kOutputW); //
            int location = static_cast<int>(topk_indices[i]) % (kOutputH * kOutputW);
            int img_x = location % kOutputW;
            int img_y = location / kOutputW;
            // Depth
            float z = base_depth_[0] + bbox_preds[8 * i] * base_depth_[1];
            // std::cout<<z<<";";
            // Location
            cv::Mat img_point(3, 1, CV_32FC1); //
            img_point.at<float>(0) = (static_cast<float>(img_x) + bbox_preds[8 * i + 1]);
            img_point.at<float>(1) = (static_cast<float>(img_y) + bbox_preds[8 * i + 2]);
            img_point.at<float>(2) = 1.0f;
            cv::Mat trams_point=trans_mat_.inv()*img_point;
            cv::Mat cam_point = intrinsic_.inv() * trams_point * z;
            float x = cam_point.at<float>(0);
            float y = cam_point.at<float>(1);
            // Dimension
            float w = base_dims_[class_id].x * expf(Sigmoid(bbox_preds[8 * i + 3]) - 0.5f);
            float l = base_dims_[class_id].y * expf(Sigmoid(bbox_preds[8 * i + 4]) - 0.5f);
            float h = base_dims_[class_id].z * expf(Sigmoid(bbox_preds[8 * i + 5]) - 0.5f);
            // Orientation
            float ori_norm = sqrtf(powf(bbox_preds[8 * i + 6], 2.0f) + powf(bbox_preds[8 * i + 7], 2.0f));
            bbox_preds[8 * i + 6] /= ori_norm; // sin(alpha)
            bbox_preds[8 * i + 7] /= ori_norm; // cos(alpha)
            float ray = atan(x / (z + 1e-7f));
            float alpha = atan(bbox_preds[8 * i + 6] / (bbox_preds[8 * i + 7] + 1e-7f));
            if (bbox_preds[8 * i + 7] > 0.0f)
            {
                alpha -= M_PI / 2.0f;
            }
            else
            {
                alpha += M_PI / 2.0f;
            }
            float angle = alpha + ray;
            if (angle > M_PI)
            {
                angle -= 2.0f * M_PI;
            }
            else if (angle < -M_PI)
            {
                angle += 2.0f * M_PI;
            }
            std::cout<<"result:"<<x<<";"<<y<<";"<<z<<";"<<w<<";"<<l<<";"<<h<<";"<<angle<<std::endl;

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
            cv::Mat cam_corners = (cv::Mat_<float>(8, 3) << -w, -l, -h, // (x0, y0, z0)
                                   -w, -l, h,                           // (x0, y0, z1)
                                   -w, l, h,                            // (x0, y1, z1)
                                   -w, l, -h,                           // (x0, y1, z0)
                                   w, -l, -h,                           // (x1, y0, z0)
                                   w, -l, h,                            // (x1, y0, z1)
                                   w, l, h,                             // (x1, y1, z1)
                                   w, l, -h);                           // (x1, y1, z0)
            cam_corners = 0.5f * cam_corners;
            cv::Mat rotation_y = cv::Mat::eye(3, 3, CV_32FC1);
            rotation_y.at<float>(0, 0) = cosf(angle);
            rotation_y.at<float>(0, 2) = sinf(angle);
            rotation_y.at<float>(2, 0) = -sinf(angle);
            rotation_y.at<float>(2, 2) = cosf(angle);
            // cos, 0, sin
            //   0, 1,   0
            //-sin, 0, cos
            cam_corners = cam_corners * rotation_y.t(); // 得到旋转后的坐标
            for (int i = 0; i < 8; ++i)
            {
                cam_corners.at<float>(i, 0) += x;
                cam_corners.at<float>(i, 1) += y;
                cam_corners.at<float>(i, 2) += z;
            }
            cam_corners = cam_corners * intrinsic_.t(); // 得到在图像上的坐标
            std::vector<cv::Point2f> img_corners(8);
            for (int i = 0; i < 8; ++i)
            {
                img_corners[i].x = cam_corners.at<float>(i, 0) / cam_corners.at<float>(i, 2);
                img_corners[i].y = cam_corners.at<float>(i, 1) / cam_corners.at<float>(i, 2);
            }
            // std::cout<<img_corners[0].x<<";";
            corners_vec.push_back(img_corners);
        }
        return corners_vec;
    }


} // namespace

}
}
}    
