#pragma once
#include "Eigen.h"
#include <algorithm>
#include <cmath>

#include "PointCloud.h"
#include <assert.h>

class PointCloudPyramid
{
private:
	std::vector<PointCloud> pointClouds;
	float* rawDepthMap;
	float* smoothedDepthMap;

private:
	PointCloudPyramid() {}

public:
	~PointCloudPyramid()
	{
		delete[] smoothedDepthMap;
	}

	PointCloudPyramid(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, const unsigned int width, const unsigned int height, const unsigned int levels = 3, const float sigmaS, const float sigmaR)
	{
		// Ensure that this is definitely uneven!
		const unsigned windowSize = 7;
		// Defines how many pixels around a block are considered
		const unsigned blockSize = 7;
		assert(windowSize % 2 == 1);
		assert(blockSize % 2 == 1);
		computeSmoothedDepthMap(width, height, windowSize, sigmaR, sigmaS);
		float* currentDepthMap = smoothedDepthMap;
		pointClouds.reserve(levels);
		for (size_t i = 0; i < levels; i++)
		{
			pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, width >> i, height >> i);
			currentDepthMap = subsampleDepthMap(currentDepthMap, width >> i, height >> i, blockSize, sigmaR);
		}
	}

	const std::vector<PointCloud>& getPointClouds() const
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
		smoothedDepthMap = new float[width * height];
#pragma omp parallel for
		for (int v = 0; v < height; ++v)
		{
			for (int u = 0; u < width; ++u)
			{
				unsigned int idx = v * width + u; // linearized index
				float normalizer = 0.0;
				float sum = 0.0;

				const int lowerLimitHeight = std::max(v - (windowSize / 2), (unsigned int)0);
				const int upperLimitHeight = std::min(v + (windowSize / 2) + 1, height);
				// Compute bilinear filter over the windowSize
				for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
				{
					const int lowerLimitWidth = std::max(u - (windowSize / 2), (unsigned int)0);
					const int upperLimitWidth = std::min(u + (windowSize / 2) + 1, width);
					for (int x = lowerLimitWidth; x < upperLimitWidth; ++x)
					{
						unsigned int idxWindow = y * width + x; // linearized index
						float summand = std::exp(-((u - x) << 1 + (v - y) << 1) * (1 / std::pow(sigmaR, 2))) * std::exp(-(std::abs(rawDepthMap[idx] - rawDepthMap[idxWindow]) * (1 / std::pow(sigmaS, 2))));
						normalizer += summand;
						sum += summand * rawDepthMap[idxWindow];
					}
				}
				smoothedDepthMap[idx] = sum / normalizer;
			}
		}
	}

private:
	float* subsampleDepthMap(float* depthMap, const unsigned width, const unsigned height, const unsigned blockSize, const float sigmaR)
	{
		float threshold = 3 * sigmaR;
		float* blockAverage = new float[(width / 2) * (height / 2)];
#pragma omp parallel for
		for (int v = 0; v < height; v = v + 2)
		{
			for (int u = 0; u < width; u = u + 2)
			{
				unsigned int idx = v * width + u; // linearized index
				float sum = 0.0;
				unsigned blockEntries = blockSize * blockSize;
				// Compute block average
				for (int y = std::max(v - (blockSize / 2), (unsigned int)0); y < std::min(v + (blockSize / 2) + 1, height); ++y)
				{
					for (int x = std::max(u - (blockSize / 2), (unsigned int)0); x < std::min(u + (blockSize / 2) + 1, width); ++x)
					{
						unsigned int idxBlock = y * width + x; // linearized index
						// TODO: Check whether pipeline issues due to wrong branch prediction are slower than this version without branching
						int invalid = (int)(std::abs(rawDepthMap[idxBlock] - rawDepthMap[idx]) > threshold);
						blockEntries -= invalid;
						sum += rawDepthMap[idxBlock] * (1 - invalid);
					}
				}
				blockAverage[(v / 2) * width + (u / 2)] = sum / blockEntries;
			}
		}
		// TODO: Ensure to delete depthMap after computation, except if it is the original smoothed one
		if (depthMap != smoothedDepthMap) {
			delete[] depthMap;
		}
		// delete[] depthMap
		return blockAverage;
	}
};
