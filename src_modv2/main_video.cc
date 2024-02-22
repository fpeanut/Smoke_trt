#include <iostream>
#include "gflags/gflags.h"
#include "smoke.h"
#include<chrono>


DEFINE_string(smoke, "smoke-sim.engine", "facenet model path");
DEFINE_string(vid, "test.mp4", "video path");
DEFINE_int32(raw_w,1920,"rational image width");
DEFINE_int32(raw_h,1080,"rational image height");
DEFINE_int32(down_scale,4,"down sample's scale");

cv::Point2f _get_ref_point(cv::Point2f p1,cv::Point2f p2){
    cv::Point2f d=p1-p2;
    cv::Point2f p3=p2+cv::Point2f(-1*d.y+d.x);
    return p3;
}

cv::Mat _get_transform_matrix(cv::Point2i center,cv::Size scale,cv::Size out_scale){
    int src_w=scale.width;
    int dst_w=out_scale.width;
    int dst_h=out_scale.height;

    cv::Point2f src_dir=cv::Point2f(0,src_w*(-0.5));
    cv::Point2f dst_dir=cv::Point2f(0,dst_w*(-0.5));

    cv::Mat src= cv::Mat::zeros(3, 2, CV_32F);
    cv::Mat dst= cv::Mat::zeros(3, 2, CV_32F);

    src.at<float>(0,0)=center.x;
    src.at<float>(0,1)=center.y;
    src.at<float>(1,0)=center.x+src_dir.x;
    src.at<float>(1,1)=center.y+src_dir.y;

    dst.at<float>(0,0)=dst_w*0.5;
    dst.at<float>(0,1)=dst_h*0.5;
    dst.at<float>(1,0)=dst_w*0.5+dst_dir.x;
    dst.at<float>(1,1)=dst_h*0.5+dst_dir.y;

    cv::Point2f src_p=_get_ref_point(
            cv::Point2f(src.at<float>(0,0),src.at<float>(0,1)),
            cv::Point2f(src.at<float>(1,0),src.at<float>(1,1)));
    cv::Point2f dst_p=_get_ref_point(
        cv::Point2f(dst.at<float>(0,0),dst.at<float>(0,1)),
        cv::Point2f(dst.at<float>(1,0),dst.at<float>(1,1)));
    
    src.at<float>(2,0)=src_p.x;
    src.at<float>(2,1)=src_p.y;
    dst.at<float>(2,0)=dst_p.x;
    dst.at<float>(2,1)=dst_p.y;
    cv::Mat trans_affine=cv::Mat::zeros(3,2,CV_64F);
    trans_affine=cv::getAffineTransform(src,dst);
    return trans_affine;

} 

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        // gflags print help message
        std::cout << "Usage: ./smoke_test --smoke smoke-sim.engine --vid test.mp4" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string smoke_path = FLAGS_smoke;
    std::string vid = FLAGS_vid;
    cv::Mat p;

    cv::VideoCapture cap(vid); // 打开视频

    // get height with from cap
    int fps = cap.get(cv::CAP_PROP_FPS);
    // image affine 
    cv::Size scale=cv::Size(FLAGS_raw_w,FLAGS_raw_h);
    cv::Size out_scale=cv::Size(1280,384);
    cv::Point2i center=cv::Point2i(FLAGS_raw_w/2,FLAGS_raw_h/2);
    int width=out_scale.width;
    int height=out_scale.height;
    
    cv::Mat trans_affine;
    trans_affine=_get_transform_matrix(center,scale,out_scale);//仿射变换矩阵

    cv::Mat trans_mat;
    trans_mat=_get_transform_matrix(center,scale,out_scale/FLAGS_down_scale);//下采样仿射变换矩阵
    

    apollo::perception::camera::SMOKE smoke(smoke_path,trans_mat); // 实例化engine

    // h264 save mp4
    cv::VideoWriter writer;

    writer.open("output.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, scale);

    while (!cap.isOpened())
    {
        std::cout << "open video failed, retry..." << std::endl;
        cap.open(vid);
    }
    while (true)
    {
        cv::Mat frame,dst;
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "frame empty" << std::endl;
            break;
        }
        cv::warpAffine(frame,dst,trans_affine,out_scale);

        auto start = std::chrono::high_resolution_clock::now();

        smoke.Detect(frame,dst);
        writer.write(frame); // write frame

        cv::imwrite("output.jpg", frame);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "duration: " << duration << "ms" << std::endl;
        std::cout << "fps: " << 1000.0 / duration << std::endl;
    }


    return 0;
}