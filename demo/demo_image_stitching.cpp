/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    cv::Mat stitch_frame;
    cv::Mat result;

    auto stitcher = cvlib::Stitcher(50);
    bool is_initialized = false;

    int pressed_key = 0;
    const auto ESC_KEY_CODE = 27;
    const auto SPACE_KEY_CODE = 32;
    while (pressed_key != ESC_KEY_CODE)
    {
        cap >> frame;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::imshow(main_wnd, frame);
        pressed_key = cv::waitKey(30);
        if (pressed_key == SPACE_KEY_CODE)
            if (!is_initialized)
            {
                stitcher.initialize(frame);
                is_initialized = true;
            }
            else
            {
                stitcher.stitch(frame, result);
                cv::imshow(demo_wnd, result);
            }
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
    return 0;
}
