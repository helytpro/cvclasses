/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{
motion_segmentation::motion_segmentation(){};


void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double weight)
{
    // \todo implement your own algorithm:
    //       * Mean

    // reading
    cv::Mat image = _image.getMat();

    // skip a first frame
    if (bg_model_.empty())
    {
        image.copyTo(bg_model_);
        _fgmask.assign(cv::Mat::zeros(image.size(), image.type()));
    }
    else
    {
        cv::Mat res;
        cv::absdiff(image, bg_model_, res);
        cv::threshold(res, res, 50, 128, cv::THRESH_BINARY);

        _fgmask.assign(res);

        bg_model_ = (1 - weight) * bg_model_ + weight * image;
    }
}
} // namespace cvlib