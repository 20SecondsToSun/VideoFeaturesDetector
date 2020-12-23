#pragma once

#include <opencv2/videoio.hpp>
#include <vector>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/core/mat.hpp>

class WebCamObjectDetector
{
public:
    WebCamObjectDetector();

    void update();
    virtual ~WebCamObjectDetector();

protected:
    constexpr static float c_confThreshold = 0.5; // Confidence threshold
    constexpr static float c_nmsThreshold = 0.4;  // Non-maximum suppression threshold
    const std::string c_winName = "Deep learning object detection in OpenCV";


    constexpr static int c_inpWidth = 416;
    constexpr static int c_inpHeight = 416;
    
    std::vector<std::string> m_classes;
    cv::dnn::Net m_net;

    cv::Mat m_frame, m_blob;

    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
    std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
    void init();

private:
    cv::VideoCapture m_stream;
};

