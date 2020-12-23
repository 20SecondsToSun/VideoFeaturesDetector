#include "WebCamObjectDetector.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace dnn;
namespace fs = std::filesystem;

WebCamObjectDetector::WebCamObjectDetector()
{
    m_stream.open(0);
    init();
}

void WebCamObjectDetector::init()
{
    const auto fullPath = fs::current_path().append("data").string();

    // Load names of classes
    string classesFile = fullPath + "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
    {
        m_classes.push_back(line);
    }

    // Give the configuration and weight files for the model
    String modelConfiguration = fullPath + "yolov3.cfg";
    String modelWeights = fullPath + "yolov3.weights";

    // Load the network
    m_net = readNetFromDarknet(modelConfiguration, modelWeights);
    m_net.setPreferableBackend(DNN_BACKEND_CUDA);
    m_net.setPreferableTarget(DNN_TARGET_CUDA);

    namedWindow(c_winName, WINDOW_NORMAL);
}

void WebCamObjectDetector::update()
{
    m_stream >> m_frame;
    
    // 4D blob
    blobFromImage(m_frame, m_blob, 1 / 255.0, Size(c_inpWidth, c_inpHeight), Scalar(0.0, 0.0, 0.0), true, false);

    //Sets the input to the network
    m_net.setInput(m_blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    m_net.forward(outs, getOutputsNames(m_net));

    // Remove the bounding boxes with low confidence
    postprocess(m_frame, outs);

    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = m_net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time for a frame : %.2f ms", t);
    putText(m_frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

    // Write the frame with the detection boxes
    Mat detectedFrame;
    m_frame.convertTo(detectedFrame, CV_8U);

    imshow(c_winName, m_frame);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void WebCamObjectDetector::postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > c_confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, c_confThreshold, c_nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void WebCamObjectDetector::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!m_classes.empty())
    {
        CV_Assert(classId < (int)m_classes.size());
        label = m_classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> WebCamObjectDetector::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

WebCamObjectDetector::~WebCamObjectDetector()
{
    m_stream.release();
}
