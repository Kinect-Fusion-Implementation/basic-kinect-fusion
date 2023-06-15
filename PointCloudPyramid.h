#pragma once
#include "Eigen.h"
#include <math.h>
#include "PointCloud.h"
#include <assert.h>

class PointCloudPyramid
{
private:
    std::vector<PointCloud> pointClouds;
    float *rawDepthMap;
    float *smoothedDepthMap;

private:
    PointCloudPyramid() {}

public:
    ~PointCloudPyramid()
    {
        delete[] smoothedDepthMap;
    }

    PointCloudPyramid(float *depthMap, const Matrix3f &depthIntrinsics, const Matrix4f &depthExtrinsics, const unsigned int width, const unsigned int height, const unsigned int levels = 3, const float sigmaS, const float sigmaR)
    {
        // Ensure that this is definitely uneven!
        const unsigned windowSize = 7;
        // Defines how many pixels around a block are considered
        const unsigned blockSize = 7;
        assert(windowSize % 2 == 1);
        assert(blockSize % 2 == 1);
        computeSmoothedDepthMap(width, height, windowSize, sigmaR, sigmaS);
        float *currentDepthMap = this.smoothedDepthMap;
        this.pointClouds.reserve(levels);
        for (size_t i = 0; i < levels; i++)
        {
            this.pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, width >> i, height >> i);
            currentDepthMap = subsampleDepthMap(currentDepthMap, width >> i, height >> i, blockSize, sigmaR);
        }
    }

    const std::vector<PointCloud> &getPointClouds() const
    {
        return pointClouds;
    }

private:
    /**
     * Computes the smoothed depthmap for every pixel based on a windowSize
     */
    void computeSmoothedDepthMap(const unsigned width, const unsigned height, const unsigned windowSize, const float sigmaR, const float sigmaS)
    {
        assert(windowSize % 2 == 1);
        // Create row major representation of depth map
        this.smoothedDepthMap = new float[width * height];
#pragma omp parallel for
        for (int v = 0; v < height; ++v)
        {
            for (int u = 0; u < width; ++u)
            {
                unsigned int idx = v * width + u; // linearized index
                float normalizer = 0.0;
                float sum = 0.0;

                const int lowerLimitHeight = math::max(v - (windowSize / 2), 0);
                const int upperLimitHeight = math::min(v + (windowSize / 2) + 1, height);
                // Compute bilinear filter over the windowSize
                for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
                {
                    const int lowerLimitWidth = math::max(u - (windowSize / 2), 0);
                    const int upperLimitWidth = math::min(u + (windowSize / 2) + 1, width);
                    for (int x = lowerLimitWidth; x < upperLimitWidth; ++x)
                    {
                        unsigned int idxWindow = y * width + x; // linearized index
                        float summand = math::exp(-((u - x) * *2 + (v - y) * *2) * (1 / (sigmaR * *2))) * math::exp(-(math::abs(this.rawDepthMap[idx] - this.rawDepthMap[idxWindow]) * (1 / (sigmaS * *2))));
                        normalizer += summand;
                        sum += summand * this.rawDepthMap[idxWindow];
                    }
                }
                this.smoothedDepthMap[idx] = sum / normalizer;
            }
        }
    }

private:
    float *subsampleDepthMap(float *depthMap, const unsigned width, const unsigned height, const unsigned blockSize, const float sigmaR)
    {
        float threshold = 3 * sigmaR;
        float *blockAverage = new float[(width / 2) * (height / 2)];
#pragma omp parallel for
        for (int v = 0; v < height; v = v + 2)
        {
            for (int u = 0; u < width; u = u + 2)
            {
                unsigned int idx = v * width + u; // linearized index
                float sum = 0.0;
                const unsigned blockEntries = blockSize * blockSize;
                // Compute block average
                for (int y = math::max(v - (blockSize / 2), 0); y < math::min(v + (blockSize / 2) + 1, height); ++y)
                {
                    for (int x = math::max(u - (blockSize / 2), 0); x < math::min(u + (blockSize / 2) + 1, width); ++x)
                    {
                        unsigned int idxBlock = y * width + x; // linearized index
                        // TODO: Check whether pipeline issues due to wrong branch prediction are slower than this version without branching
                        int invalid = (int)(math::abs(this.rawDepthMap[idxBlock] - this.rawDepthMap[idx]) > threshold);
                        blockEntries -= invalid;
                        sum += this.rawDepthMap[idxBlock] * (1 - invalid);
                    }
                }
                blockAverage[(v / 2) * width + (u / 2)] = sum / blockEntries;
            }
        }
        // TODO: Ensure to delete depthMap after computation, except if it is the original smoothed one
        if (depthMap != this.smoothedDepthMap) {
            delete[] depthMap;
        }
        // delete[] depthMap
        return blockAverage;
    }
};
