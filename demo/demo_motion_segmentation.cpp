/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cvlib::motion_segmentation mseg; 
    //auto mseg = cv::createBackgroundSubtractorMOG2(); 
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    int threshold = 50;
    double alpha = 0.1;
    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    //threshold is not link to mseg
    cv::createTrackbar("th", demo_wnd, &threshold, 255);

    cv::Mat frame;
    cv::Mat frame_mseg;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        //error: "setVarThreshold": is not an element of "cvslib::motion_segmentation".
        //mseg->setVarThreshold(threshold); // \todo use TackbarCallback
        mseg.apply(frame, frame_mseg, alpha);
        //mseg->apply(frame, frame_mseg);
        if (!frame_mseg.empty())
            cv::imshow(demo_wnd, frame_mseg);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
