cmake_minimum_required(VERSION 3.0.0)

if (WIN32)
	MESSAGE(STATUS "Now is windows")
    # CUDA
    find_package(CUDA REQUIRED)

    # OpenCV
    find_package(OpenCV 3.4.4 REQUIRED PATHS D:/3rd_party/opencv344/build_gpu/install)

    # TensorRT
    set(TENSORRT_ROOT D:/TensorRT-7.0.0.11)
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)

    # windows utils
    set(UTILS_INCLUDE D:/3rd_party/utils)

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
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extra/include
                Public ${UTILS_INCLUDE})

    target_link_libraries(detector ${OpenCV_LIBS} 
                        ${TENSORRT_LIB_DIR}/nvinfer.lib 
                        ${CUDA_LIBRARIES})
    

elseif (UNIX)
    # CUDA
    find_package(CUDA REQUIRED)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)


    # OpenCV
    if(EV_DOCKER)
        find_package(OpenCV REQUIRED)
    else()
        find_package(OpenCV 3.4.15 REQUIRED PATHS /root/PLATFORM/3rdparty/opencv3415)
    endif()


    # TensorRT
    if(EV_DOCKER)
        set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu/)
        set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu/)
    else()
        set(TENSORRT_ROOT /root/PLATFORM/3rdparty/TensorRT-7.0.0.11)
        set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
        set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
    endif()



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
endif()



# cd build 
# make install
install(FILES class_detector.h DESTINATION include)
install(TARGETS detector LIBRARY DESTINATION lib)
install(FILES extra/yolov5s.wts DESTINATION lib)
