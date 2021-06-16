#pragma once

// std
#include <opencv2/opencv.hpp>
#include "class_detector.h"

#include "NvInferRuntime.h"

class _Detector
{
public:
    _Detector();
    ~_Detector();

public:
    bool init(const std::string &engine_path_);

    bool detect(const std::vector<cv::Mat> &imgs_, std::vector<std::vector<Detection> > &vec_detections_);
    
private:
    nvinfer1::IRuntime* _runtime;
    nvinfer1::ICudaEngine* _engine;
    nvinfer1::IExecutionContext* _context;
    cudaStream_t _stream;

    void* _buffers[2];
};
