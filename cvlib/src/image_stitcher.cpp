#include <cmath>
#include <vector>
#include "cvlib.hpp"

namespace cvlib
{
Stitcher::Stitcher(float ratio)
{
    this->detector = cvlib::corner_detector_fast::create(); 
    this->matcher = cvlib::descriptor_matcher(ratio);
}

void Stitcher::initialize(cv::InputArray input)
{
    input.getMat().copyTo(this->dst);
}

void Stitcher::stitch(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat src;
    input.getMat().copyTo(src);

    this->detector->detectAndCompute(src, cv::noArray(), this->srcCorners, this->srcDescriptors);
    this->detector->detectAndCompute(this->dst, cv::noArray(), this->dstCorners, this->dstDescriptors);

    std::vector<std::vector<cv::DMatch>> matches;
    this->matcher.radiusMatch(this->srcDescriptors, this->dstDescriptors, matches, 100);

    std::vector<cv::Point2f> srcKps;
    std::vector<cv::Point2f> dstKps;
    for (const auto& match : matches)
        if (!match.empty())
        {
            srcKps.push_back(this->srcCorners[match[0].queryIdx].pt);
            dstKps.push_back(this->dstCorners[match[0].queryIdx].pt);
        }
    if ((srcKps.empty() || dstKps.empty()))
        return;

    cv::Mat homography = cv::findHomography(srcKps, dstKps, cv::RANSAC);
    const auto dst_size = cv::Size(dst.cols + src.cols, dst.rows);
    dst = cv::Mat(dst_size, CV_8U);
    cv::warpPerspective(src, dst, homography, dst.size(), cv::INTER_CUBIC);
    cv::Mat roi = cv::Mat(dst, cv::Rect(0, 0, dst.cols, dst.rows));
    dst.copyTo(output);
}
} // namespace cvlib