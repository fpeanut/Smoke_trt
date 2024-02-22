#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include<memory>
// #include "buffers.h"
// #include "utils/preprocess.h"
#include "gflags/gflags.h"
#include <NvInferPlugin.h>

#include <opencv2/opencv.hpp>

using namespace nvinfer1;

// define input flags

DEFINE_string(onnx_file, "smoke_dla34.onnx", "onnx file path");
DEFINE_string(calib_dir, "", "calibration data dir");
DEFINE_string(calib_list_file, "", "calibration data list file");
DEFINE_string(input_name, "input", "network input name");
DEFINE_int32(input_h, 384, "network input height");
DEFINE_int32(input_w, 1280, "network input width");
DEFINE_int32(input_c, 3, "network input channel");
DEFINE_bool(int8, false, "use int8 mode");
DEFINE_string(model_name, "smoke", "model name");
DEFINE_string(format, "nchw", "input format");


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        // std::cout << "Usage: ./build [onnx_file] [calib_dir] [calib_list_file]" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const char *onnx_file_path = FLAGS_onnx_file.c_str();
    const char *calib_dir = FLAGS_calib_dir.c_str();
    const char *calib_list_file = FLAGS_calib_list_file.c_str();

    int input_h = FLAGS_input_h;
    int input_w = FLAGS_input_w;
    const char *input_name = FLAGS_input_name.c_str();

    bool useInt8 = FLAGS_int8;

    // remove extension of onnx_file_path
    std::string output_file_name = FLAGS_onnx_file.substr(0, FLAGS_onnx_file.find_last_of(".")) + ".engine";

    sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);

    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    // 1. Create builder
    // auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    if (!builder)
    {
        return -1;
    }


    // 2. Create network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    if (!network)
    {
        return -1;
    }

    // 3. Create builder config
    nvinfer1::IBuilderConfig *config=builder->createBuilderConfig();
    if (!config)
    {
        return -1;
    }

    // 4. Create ONNX Parser
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());

    // 5. Parse ONNX model
    auto parsed = parser->parseFromFile(onnx_file_path, static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return -1;
    }

    auto input = network->getInput(0);
    auto profile = builder->createOptimizationProfile();
    if (FLAGS_format == "nchw")
    {
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, FLAGS_input_c, input_h, input_w});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, FLAGS_input_c, input_h, input_w});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, FLAGS_input_c, input_h, input_w});
    }
    else
    {
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, input_h, input_w, FLAGS_input_c});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, input_h, input_w, FLAGS_input_c});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, input_h, input_w, FLAGS_input_c});
    }
    config->addOptimizationProfile(profile);

    // 6. set calibration configuration
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 30);
    

    // 7. Create CUDA stream for profiling
    cudaStream_t profileStream;
    cudaError_t cudaStatus = cudaStreamCreate(&profileStream);
    if (cudaStatus != cudaSuccess)
    {
        return -1;
    }

    config->setProfileStream(profileStream);

    // 8. Build Serialized Engine
    nvinfer1::ICudaEngine *plan=builder->buildEngineWithConfig(*network, *config);
    if (!plan)
    {
        return -1;
    }

        //序列化engine
    nvinfer1::IHostMemory *engineString =plan->serialize();
    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "Failed building serialized engine!" << std::endl;
        return 1;
    }
    std::cout << "Succeeded building serialized engine!" << std::endl;

    // 9. save engine
    std::ofstream engine_file(output_file_name, std::ios::binary);
    engine_file.write((char *)engineString->data(), engineString->size());
    engine_file.close();

    builder->destroy();
    network->destroy();


    return 0;
}


