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
	float* m_smoothedDepthMap;
	int m_width;
	int m_height;
	int m_windowSize;
	int m_blockSize;

private:
	PointCloudPyramid() {}

	void printImageArray(float* map, int width, int height, std::string name) {
		std::cout << "Printing " + name + ":" << std::endl;
		for (unsigned int j = 0; j < height; j++) {
			for (unsigned int i = 0; i < width; i++) {
				std::cout << " " << map[(width * j + i)];
			}
			std::cout << std::endl;
		}
	}

public:
	~PointCloudPyramid()
	{
		delete[] m_smoothedDepthMap;
	}

	PointCloudPyramid(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, const unsigned int width, const unsigned int height, const unsigned int levels, const unsigned int windowSize, const unsigned int blockSize, const unsigned int invalidThreshold, const float sigmaS, const float sigmaR) : m_width(width), m_height(height), m_windowSize(windowSize), m_blockSize(blockSize)
	{
		assert(m_width > 0);
		assert(m_height > 0);
		assert(m_windowSize > 0);
		assert(m_windowSize % 2 == 1);
		assert(m_blockSize % 2 == 1);

		rawDepthMap = depthMap;
		computeSmoothedDepthMap(invalidThreshold, sigmaR, sigmaS);

		FreeImage smoothedDepthImage(m_width, m_height, 1);
		smoothedDepthImage.data = m_smoothedDepthMap;
		std::cout << "Saving smoothed depthmap... " << std::endl;
		std::string fileName("./SmoothedDepthMap");
		smoothedDepthImage.SaveDepthMapToFile(fileName + std::to_string(0) + ".png");

		float* currentDepthMap = m_smoothedDepthMap;
		pointClouds.reserve(levels);
		pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, m_width, m_height, 0);
		for (size_t i = 0; i < levels;)
		{
			currentDepthMap = subsampleDepthMap(currentDepthMap, m_width >> i, m_height >> i, sigmaR, i + 1);
			i++;
			FreeImage subsampledDepthImage(m_width >> i, m_height >> i, 1);
			std::cout << "Saving subsampled depthmap... " << std::endl;
			std::string fileName("./SubsampledDepthMap");
			subsampledDepthImage.data = currentDepthMap;
			subsampledDepthImage.SaveDepthMapToFile(fileName + std::to_string(i) + ".png");

			pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, m_width >> i, m_height >> i, i);
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
	void computeSmoothedDepthMap(const int validitySmoothingThreshold, const float sigmaR, const float sigmaS)
	{
		// Create row major representation of depth map
		m_smoothedDepthMap = new float[m_width * m_height];
#pragma omp parallel for
		for (int v = 0; v < m_height; ++v)
		{
			for (int u = 0; u < m_width; ++u)
			{
				unsigned int idx = v * m_width + u; // linearized index
				float normalizer = 0.0;
				float sum = 0.0;
				int invalidCounter = 0;

				const int lowerLimitHeight = std::max(v - (m_windowSize / 2), 0);
				const int upperLimitHeight = std::min(v + (m_windowSize / 2) + 1, m_height);
				// Compute bilinear filter over the windowSize
				for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
				{
					const int lowerLimitWidth = std::max(u - (m_windowSize / 2), 0);
					const int upperLimitWidth = std::min(u + (m_windowSize / 2) + 1, m_width);
					for (int x = lowerLimitWidth; x < upperLimitWidth; ++x)
					{
						unsigned int idxWindow = y * m_width + x; // linearized index
						// TODO: Check whether depth difference too large
						if (rawDepthMap[idxWindow] == MINF) {
							invalidCounter++;
							continue;
						}
						float summand = std::exp(-(std::pow((u - x), 2) + std::pow((v - y), 2)) * (1 / std::pow(sigmaS, 2))) * std::exp(-(std::pow(rawDepthMap[idx] - rawDepthMap[idxWindow], 2) * (1 / std::pow(sigmaR, 2))));
						normalizer += summand;
						sum += summand * rawDepthMap[idxWindow];
					}
				}
				if (rawDepthMap[idx] != MINF || invalidCounter < validitySmoothingThreshold) {
					m_smoothedDepthMap[idx] = sum / normalizer;
				}
				else {
					m_smoothedDepthMap[idx] = MINF;
				}
			}
		}
	}

private:
	float* subsampleDepthMap(float* depthMap, const unsigned width, const unsigned height, const float sigmaR, const int level)
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
				unsigned blockEntries = m_blockSize * m_blockSize;
				// Compute block average
				for (int y = std::max(v - (m_blockSize / 2), 0); y < std::min(v + (m_blockSize / 2) + 1, int(height)); ++y)
				{
					for (int x = std::max(u - (m_blockSize / 2), 0); x < std::min(u + (m_blockSize / 2) + 1, int(width)); ++x)
					{
						unsigned int idxBlock = y * width + x; // linearized index
						// TODO: Check whether pipeline issues due to wrong branch prediction are slower than this version without branching
						int invalid = (int)(std::abs(depthMap[idxBlock] - depthMap[idx]) > threshold);
						blockEntries -= invalid;
						sum += depthMap[idxBlock] * (1 - invalid);
					}
				}
				blockAverage[(v / 2) * (width / 2) + (u / 2)] = sum / blockEntries;
			}
		}

		// TODO: Ensure to delete depthMap after computation, except if it is the original smoothed one
		if (depthMap != m_smoothedDepthMap) {
			delete[] depthMap;
		}
		// delete[] depthMap
		return blockAverage;
	}
};
