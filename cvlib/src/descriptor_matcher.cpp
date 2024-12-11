/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
int descriptor_matcher::distance(const cv::Mat& q_desc, const cv::Mat& t_desc)
{
    int dist = 0;
    int length = q_desc.cols;
    uint16_t bytes_q, bytes_t;
    bool bit_q, bit_t;

    for (int i = 0; i < length; ++i)
    {
        bytes_q = q_desc.at<uint16_t>(0, i);
        bytes_t = t_desc.at<uint16_t>(0, i);

        for (int j = 0; j < 16; ++j)
        {
            bit_q = (((bytes_q) << (15 - j)) & 0xFFFF) >> 15;
            bit_t = (((bytes_t) << (15 - j)) & 0xFFFF) >> 15;

            if (bit_q != bit_t)
                dist++;
        }
    }

    return dist;
}

void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    int current_dist;
    int trainIdx;
    int dist = this->ratio_;
    bool flag;

    matches.resize(q_desc.rows);

    cv::RNG rnd;
    for (int i = 0; i < q_desc.rows; ++i)
    {
        trainIdx = 0;
        dist = this->ratio_;
        flag = false;
        for (int j = 0; j < t_desc.rows; j++)
        {
            current_dist = this->distance(q_desc.row(i), t_desc.row(j));

            if (current_dist < dist)
            {
                flag = true;
                dist = current_dist;
                trainIdx = j;
            }
        }

        if (flag)
            matches[i].emplace_back(i, trainIdx, dist);
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float /*maxDistance*/,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib