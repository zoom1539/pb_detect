#pragma once

// std
#include <opencv2/opencv.hpp>
#include "class_detector.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

using namespace nvinfer1;

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
    int get_width(int x, float gw, int divisor = 8);
    int get_depth(int x, float gd);
    ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name);
    ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name);
    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name);
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);

private:
    nvinfer1::IRuntime* _runtime = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;
    cudaStream_t _stream;

    void* _buffers[2] = {nullptr};
};
