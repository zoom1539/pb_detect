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
    bool serialize(std::string &wts_path_, const std::string &engine_path_);
    
    bool init(const std::string &engine_path_);

    bool detect(const std::vector<cv::Mat> &imgs_, std::vector<std::vector<Detection> > &vec_detections_);
    
private:
    nvinfer1::IRuntime* _runtime = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;
    cudaStream_t _stream;

    void* _buffers[2] = {nullptr};
};
