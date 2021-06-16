cmake_minimum_required(VERSION 3.0.0)



# CUDA
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


# OpenCV
# find_package(OpenCV 3.4.8 REQUIRED PATHS ../extra/opencv348)
find_package(OpenCV REQUIRED)


# TensorRT
set(TENSORRT_ROOT /home/ubuntu/PLATFORM/3rdparty/TensorRT-7.0.0.11)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)


#
aux_source_directory(. LIB_SRC)
list(APPEND LIB_SRC 
    extra/include/common.hpp 
    extra/include/yololayer.cu)
message(${LIB_SRC})
cuda_add_library(detector SHARED ${LIB_SRC} )

target_include_directories(detector 
            PUBLIC ${TENSORRT_INCLUDE_DIR}
            PUBLIC ${OpenCV_INCLUDE_DIRS}
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extra/include)

target_link_libraries(detector ${OpenCV_LIBS} 
                      ${TENSORRT_LIB_DIR}/libnvinfer.so 
                      ${CUDA_LIBRARIES})

# cd build 
# make install
install(FILES class_detector.h DESTINATION include)
install(TARGETS detector LIBRARY DESTINATION lib)
install(FILES extra/yolov5s.engine DESTINATION lib)
