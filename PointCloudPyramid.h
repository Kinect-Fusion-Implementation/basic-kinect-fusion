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

	PointCloudPyramid(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, const unsigned int width, const unsigned int height, const unsigned int levels, const float sigmaS, const float sigmaR)
	{
		// Ensure that this is definitely uneven!
		const unsigned windowSize = 7;
		// Defines how many pixels around a block are considered
		const unsigned blockSize = 7;
		assert(windowSize % 2 == 1);
		assert(blockSize % 2 == 1);

		rawDepthMap = depthMap;
		computeSmoothedDepthMap(width, height, windowSize, sigmaR, sigmaS);
		/*
		std::cout << "Printing smooth map:" << std::endl;
		for (unsigned int j = 0; j < height; j++) {
			for (unsigned int i = 0; i < width; i++) {
				std::cout << " " << smoothedDepthMap[(width*j + i)];
			}
			std::cout << std::endl;
		}
		*/
		FreeImage smoothedDepthImage(width, height, 1);
		smoothedDepthImage.data = smoothedDepthMap;
		std::cout << "Saving smoothed depthmap... " << std::endl;
		std::string fileName("./SmoothedDepthMap");
		smoothedDepthImage.SaveDepthMapToFile(fileName + std::to_string(0) + ".png");

		float* currentDepthMap = smoothedDepthMap;
		pointClouds.reserve(levels);
		pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, width, height, 0);
		for (size_t i = 0; i < levels; i++)
		{
			currentDepthMap = subsampleDepthMap(currentDepthMap, width >> i, height >> i, blockSize, sigmaR, i + 1);

			FreeImage subsampledDepthImage(width >> i, height >> i, 1);
			std::cout << "Saving subsampled depthmap... " << std::endl;
			std::string fileName("./SubsampledDepthMap");
			smoothedDepthImage.SaveDepthMapToFile(fileName + std::to_string(i+1) + ".png");

			pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, width >> i, height >> i, i + 1);	
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
	void computeSmoothedDepthMap(const int width, const int height, const int windowSize, const float sigmaR, const float sigmaS)
	{
		assert(width > 0);
		assert(height > 0);
		assert(windowSize > 0);
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

				const int lowerLimitHeight = std::max(v - (windowSize / 2), 0);
				const int upperLimitHeight = std::min(v + (windowSize / 2) + 1, height);
				// Compute bilinear filter over the windowSize
				for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
				{
					const int lowerLimitWidth = std::max(u - (windowSize / 2), 0);
					const int upperLimitWidth = std::min(u + (windowSize / 2) + 1, width);
					for (int x = lowerLimitWidth; x < upperLimitWidth; ++x)
					{
						unsigned int idxWindow = y * width + x; // linearized index
						if (rawDepthMap[idxWindow] == MINF) {
							continue;
						}
						float summand = std::exp(-(std::pow((u - x), 2) + std::pow((v - y), 2)) * (1 / std::pow(sigmaS, 2))) * std::exp(-(std::pow(rawDepthMap[idx] - rawDepthMap[idxWindow], 2) * (1 / std::pow(sigmaR, 2))));
						normalizer += summand;
						sum += summand * rawDepthMap[idxWindow];
					}
				}
				smoothedDepthMap[idx] = sum / normalizer;
			}
		}
	}

private:
	float* subsampleDepthMap(float* depthMap, const unsigned width, const unsigned height, const unsigned blockSize, const float sigmaR, const int level)
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
						int invalid = (int)(std::abs(depthMap[idxBlock] - depthMap[idx]) > threshold);
						blockEntries -= invalid;
						sum += depthMap[idxBlock] * (1 - invalid);
					}
				}
				blockAverage[(v / 2) * (width/2) + (u / 2)] = sum / blockEntries;
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
