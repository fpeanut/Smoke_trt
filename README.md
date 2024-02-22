# Smoke_trt
This version of smoke deployment project can adapt to the internal parameters of different cameras, solving the problem that other deployment projects can only use kitti internal parameters.

Optimization summary

- The project smoke can adapt to the resolution and internal parameters of various cameras.
- Detailed integration of tensorrt7, tensorrt8 and apollo8.0 environment code.
- Solve the problem of not recognizing the deconv plugin operator.

## Model && Data

Regarding the model, model conversion code and test data acquisition, please send an email to 250854911@qq.com. I will reply in time when I see it.

## Project Introduction && Run

- tensorrt8 smoke 
Include src/,build.cu,smoke_signal_test.cpp,smoke_test.cpp,CMakeLists_v8.txt

```shell
Modify CMakeLists_v8.txt to CMakeLists.txt
$ mkdir build && cd build
$ cmake .. && make 
$ cd ..
./build/build --onnx_file ./weights/smoke_dla34.onnx     
./build/smoke_test --smoke ./weights/smoke_dla34.engine --vid media/road_wx1.avi
./build/smoke_img_test --smoke ./weights/smoke_dla34.engine --image media/test.jpg
```

- tensorrt7 smoke 
Include src_v7/,build.cu,smoke_signal_test_v7.cpp,smoke_test.cpp,CMakeLists_v7.txt
```shell
Same as above
```

- tensorrt_apollo smoke 
Include src_modv2/,CMakeLists_apollo.txt
```shell
Same as above
```

#### Performance in RTX3070ti of FP16
```
| Function(unit:ms) | NVIDIA RTX 3070ti Laptop GPU |
| ----------------- | --------------------------- |
| Summary           | 25 ms                 |
```

## Note 
Please contact me about smoke adaptive resolution detection theory