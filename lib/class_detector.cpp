#include "class_detector.h"
#include "class_detector_.h"

class Detector::Impl
{
public:
    _Detector _detector;
};

Detector::Detector() : _impl(new Detector::Impl())
{
}

Detector::~Detector()
{
    delete _impl;
    _impl = NULL;
}

bool Detector::init(const std::string &engine_path_)
{
    return _impl->_detector.init(engine_path_);
}

bool Detector::detect(const std::vector<cv::Mat> &imgs_, 
                      std::vector<std::vector<Detection> > &vec_detections_)
{
    return _impl->_detector.detect(imgs_, vec_detections_);
}


