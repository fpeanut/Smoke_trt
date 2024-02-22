# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /app/TensorRT/smoke_bushu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /app/TensorRT/smoke_bushu/build

# Include any dependencies generated for this target.
include CMakeFiles/engine.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/engine.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/engine.dir/flags.make

CMakeFiles/engine.dir/src_v7/engine.cu.o: CMakeFiles/engine.dir/flags.make
CMakeFiles/engine.dir/src_v7/engine.cu.o: ../src_v7/engine.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/app/TensorRT/smoke_bushu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/engine.dir/src_v7/engine.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /app/TensorRT/smoke_bushu/src_v7/engine.cu -o CMakeFiles/engine.dir/src_v7/engine.cu.o

CMakeFiles/engine.dir/src_v7/engine.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/engine.dir/src_v7/engine.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/engine.dir/src_v7/engine.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/engine.dir/src_v7/engine.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/engine.dir/src_v7/common/logger.cpp.o: CMakeFiles/engine.dir/flags.make
CMakeFiles/engine.dir/src_v7/common/logger.cpp.o: ../src_v7/common/logger.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/app/TensorRT/smoke_bushu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/engine.dir/src_v7/common/logger.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/engine.dir/src_v7/common/logger.cpp.o -c /app/TensorRT/smoke_bushu/src_v7/common/logger.cpp

CMakeFiles/engine.dir/src_v7/common/logger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/engine.dir/src_v7/common/logger.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /app/TensorRT/smoke_bushu/src_v7/common/logger.cpp > CMakeFiles/engine.dir/src_v7/common/logger.cpp.i

CMakeFiles/engine.dir/src_v7/common/logger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/engine.dir/src_v7/common/logger.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /app/TensorRT/smoke_bushu/src_v7/common/logger.cpp -o CMakeFiles/engine.dir/src_v7/common/logger.cpp.s

# Object files for target engine
engine_OBJECTS = \
"CMakeFiles/engine.dir/src_v7/engine.cu.o" \
"CMakeFiles/engine.dir/src_v7/common/logger.cpp.o"

# External object files for target engine
engine_EXTERNAL_OBJECTS =

CMakeFiles/engine.dir/cmake_device_link.o: CMakeFiles/engine.dir/src_v7/engine.cu.o
CMakeFiles/engine.dir/cmake_device_link.o: CMakeFiles/engine.dir/src_v7/common/logger.cpp.o
CMakeFiles/engine.dir/cmake_device_link.o: CMakeFiles/engine.dir/build.make
CMakeFiles/engine.dir/cmake_device_link.o: libmmcv_plugin.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvinfer.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvparsers.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvinfer.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvparsers.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
CMakeFiles/engine.dir/cmake_device_link.o: CMakeFiles/engine.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/app/TensorRT/smoke_bushu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/engine.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/engine.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/engine.dir/build: CMakeFiles/engine.dir/cmake_device_link.o

.PHONY : CMakeFiles/engine.dir/build

# Object files for target engine
engine_OBJECTS = \
"CMakeFiles/engine.dir/src_v7/engine.cu.o" \
"CMakeFiles/engine.dir/src_v7/common/logger.cpp.o"

# External object files for target engine
engine_EXTERNAL_OBJECTS =

libengine.so: CMakeFiles/engine.dir/src_v7/engine.cu.o
libengine.so: CMakeFiles/engine.dir/src_v7/common/logger.cpp.o
libengine.so: CMakeFiles/engine.dir/build.make
libengine.so: libmmcv_plugin.so
libengine.so: /usr/lib/x86_64-linux-gnu/libnvinfer.so
libengine.so: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
libengine.so: /usr/lib/x86_64-linux-gnu/libnvparsers.so
libengine.so: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libnvinfer.so
libengine.so: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
libengine.so: /usr/lib/x86_64-linux-gnu/libnvparsers.so
libengine.so: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
libengine.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
libengine.so: CMakeFiles/engine.dir/cmake_device_link.o
libengine.so: CMakeFiles/engine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/app/TensorRT/smoke_bushu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libengine.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/engine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/engine.dir/build: libengine.so

.PHONY : CMakeFiles/engine.dir/build

CMakeFiles/engine.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/engine.dir/cmake_clean.cmake
.PHONY : CMakeFiles/engine.dir/clean

CMakeFiles/engine.dir/depend:
	cd /app/TensorRT/smoke_bushu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /app/TensorRT/smoke_bushu /app/TensorRT/smoke_bushu /app/TensorRT/smoke_bushu/build /app/TensorRT/smoke_bushu/build /app/TensorRT/smoke_bushu/build/CMakeFiles/engine.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/engine.dir/depend
