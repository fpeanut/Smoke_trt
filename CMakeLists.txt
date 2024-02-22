cmake_minimum_required(VERSION 3.11)
project(TensorRT_MODNet LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
include_directories(${CMAKE_SOURCE_DIR}/src_modv2)
link_directories(/usr/local/cuda/lib64/)

add_library(mmcv_plugin SHARED
    src_modv2/trt_modulated_deform_conv_kernel.cu
    src_modv2/trt_cuda_helper.cu
    src_modv2/trt_modulated_deform_conv.cpp
)
target_include_directories(mmcv_plugin
    PUBLIC
    ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries(mmcv_plugin
    PUBLIC
    -Wl,--no-as-needed,--gc-sections,--unresolved-symbols=ignore-in-shared-libs
    cublas
    ${TensorRT_LIBRARIES}
)
set_target_properties(mmcv_plugin PROPERTIES CUDA_ARCHITECTURES "61")


add_library(smoke SHARED
    src_modv2/preprocess.cu
    src_modv2/smoke.cc
)
target_link_libraries(smoke
    PUBLIC
    -Wl,--no-as-needed,--gc-sections,--unresolved-symbols=ignore-in-shared-libs
    mmcv_plugin
    cuda 
    cudart
    ${OpenCV_LIBS}
)
set_target_properties(smoke PROPERTIES CUDA_ARCHITECTURES "61")


add_executable(main
    src_modv2/main.cc
)
target_link_libraries(main  smoke Threads::Threads gflags)

add_executable(main_video
    src_modv2/main_video.cc
)
target_link_libraries(main_video smoke Threads::Threads gflags)