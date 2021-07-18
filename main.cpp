#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <chrono>
#include "class_detector.h"


int main()
{
    Detector detector;

    //
    std::string wts_path = "../yolov5s.wts";
    std::string engine_path = "../lib/extra/yolov5s_fp16_b1.engine";
    int class_num = 80;
#if 0
    bool is_serialize = detector.serialize(wts_path, engine_path, class_num);
    if(!is_serialize)
    {
        std::cout << "init fail\n";
        return 0;
    }

    return 1;


#else
    bool is_init = detector.init(engine_path);
    if(!is_init)
    {
        std::cout << "init fail\n";
        return 0;
    }

    //
    std::vector<cv::Mat> imgs;
    {
        cv::Mat img = cv::imread("../data/phone.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/zidane.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/bus.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/bus.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/zidane.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/bus.jpg");
        imgs.push_back(img);
    }
    

    auto start = std::chrono::system_clock::now();

    std::vector<std::vector<Detection> > vec_detections;
    bool is_detect = detector.detect(imgs, vec_detections);

    if(!is_detect)
    {
        std::cout << "detect fail\n";
        return 0;
    }

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " total ms" << std::endl;
    
    //
    for (int i = 0; i < imgs.size(); i++)
    {
        std::vector<Detection> detections = vec_detections[i];
        for (int j = 0; j < detections.size(); j++)
        {
            cv::rectangle(imgs[i], detections[j].rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(imgs[i], std::to_string((int)detections[j].class_id), cv::Point(detections[j].rect.x, detections[j].rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        std::stringstream ss;
        ss <<"../data/" << i << ".jpg";
        cv::imwrite(ss.str(), imgs[i]);
    }
#endif
    
    std::cin.get();
    return 0;
}

