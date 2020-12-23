#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include "WebCamObjectDetector.h"

int main()
{
    const auto detector = std::make_shared<WebCamObjectDetector>();

    while (cv::waitKey(1) < 0)
    {
        detector->update();
    }

    return 0;
}
