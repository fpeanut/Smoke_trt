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
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src_modv2/main.cc.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src_modv2/main.cc.o: ../src_modv2/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/app/TensorRT/smoke_bushu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src_modv2/main.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src_modv2/main.cc.o -c /app/TensorRT/smoke_bushu/src_modv2/main.cc

CMakeFiles/main.dir/src_modv2/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src_modv2/main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /app/TensorRT/smoke_bushu/src_modv2/main.cc > CMakeFiles/main.dir/src_modv2/main.cc.i

CMakeFiles/main.dir/src_modv2/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src_modv2/main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /app/TensorRT/smoke_bushu/src_modv2/main.cc -o CMakeFiles/main.dir/src_modv2/main.cc.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src_modv2/main.cc.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src_modv2/main.cc.o
main: CMakeFiles/main.dir/build.make
main: libsmoke.so
main: libmmcv_plugin.so
main: /usr/lib/x86_64-linux-gnu/libnvinfer.so
main: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
main: /usr/lib/x86_64-linux-gnu/libnvparsers.so
main: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
main: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
main: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/app/TensorRT/smoke_bushu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /app/TensorRT/smoke_bushu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /app/TensorRT/smoke_bushu /app/TensorRT/smoke_bushu /app/TensorRT/smoke_bushu/build /app/TensorRT/smoke_bushu/build /app/TensorRT/smoke_bushu/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend
