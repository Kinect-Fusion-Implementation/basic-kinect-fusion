#pragma once
#include "Eigen.h"
#include <algorithm>
#include <cmath>

#include "PointCloud.h"
#include <assert.h>

class PointCloudPyramid
{
private:
	// Stores layers of the pyramid
	std::vector<PointCloud> pointClouds;
	// Original depth map
	float *rawDepthMap;
	// Smoothed depth map
	float *m_smoothedDepthMap;
	// Width of raw and smoothed depth map
	int m_width;
	// Height of raw and smoothed depth map
	int m_height;
	// Sidelength of square considered for smoothing
	int m_windowSize;
	// Sidelength of square considered for subsampling
	int m_blockSize;

private:
	PointCloudPyramid() = delete;

public:
	~PointCloudPyramid()
	{
		delete[] m_smoothedDepthMap;
	}

	/*
	 * Constructs a pyramid of pointClouds. The first level gets smoothed and the others subsampled.
	 *
	 * depthMap: The original depth map
	 * depthIntrinsics: The matrix which maps from camera space to pixel space
	 * depthExtrinsics: The matrix which maps from world space to camera space
	 * width: The width of the depth map, must be positive
	 * height: The height of the depth map, must be positive
	 * levels: The number of levels of the pyramid
	 * windowSize: The sidelength of the square considered for the smoothing
	 * blockSize: The sidelength of the square considered for the subsampling
	 * sigmaR: Parameter for bilinear smoothing
	 * sigmaS: Parameter for bilinear smoothing
	 */
	PointCloudPyramid(float *depthMap, const Matrix3f &depthIntrinsics, const Matrix4f &depthExtrinsics, const unsigned int width, const unsigned int height, const unsigned int levels, const unsigned int windowSize, const unsigned int blockSize, const float sigmaR, const float sigmaS) : rawDepthMap(depthMap), m_width(width), m_height(height), m_windowSize(windowSize), m_blockSize(blockSize)
	{
		// Input validation
		assert(m_width > 0);
		assert(m_height > 0);
		assert(m_windowSize > 0);
		assert(m_windowSize % 2 == 1);
		assert(m_blockSize % 2 == 1);

// Compute smoothed depth map
#if EVAL_MODE
		auto smoothDepthMapStart = std::chrono::high_resolution_clock::now();
#endif
		computeSmoothedDepthMap(sigmaR, sigmaS);
#if EVAL_MODE
		auto smoothDepthMapEnd = std::chrono::high_resolution_clock::now();
		std::cout << "Computing the smoothed depth map took: " << std::chrono::duration_cast<std::chrono::milliseconds>(smoothDepthMapEnd - smoothDepthMapStart).count() << " ms" << std::endl;
#endif
#if SAVE_IMAGE_MODE 
		ImageUtil::saveDepthMapToImage(m_smoothedDepthMap, m_width, m_height, "SmoothedDepthMap", "Saving smoothed depth map...");
#endif
		// Setup of pyramid
		float *currentDepthMap = m_smoothedDepthMap;
		pointClouds.reserve(levels + 1);

// Construct pyramid of pointClouds
#if EVAL_MODE
		auto computeInitialPointCloudStart = std::chrono::high_resolution_clock::now();
#endif
		pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, m_width, m_height, 0);
#if EVAL_MODE
		auto computeInitialPointCloudEnd = std::chrono::high_resolution_clock::now();
		std::cout << "Computing the point cloud of the original scale took: " << std::chrono::duration_cast<std::chrono::milliseconds>(computeInitialPointCloudEnd - computeInitialPointCloudStart).count() << " ms" << std::endl;

#endif
		for (size_t i = 0; i < levels;)
		{
#if EVAL_MODE
			auto subsampleStart = std::chrono::high_resolution_clock::now();
#endif
			// Compute subsampled depth map
			currentDepthMap = subsampleDepthMap(currentDepthMap, m_width >> i, m_height >> i, sigmaR);
#if EVAL_MODE
			auto subsampleEnd = std::chrono::high_resolution_clock::now();
			std::cout << "Computing subsampled depth map took: " << std::chrono::duration_cast<std::chrono::milliseconds>(subsampleEnd - subsampleStart).count() << " ms" << std::endl;
#endif
			i++;
			// Print subsampled depth map to file
#if SAVE_IMAGE_MODE 
			ImageUtil::saveDepthMapToImage(currentDepthMap, m_width >> i, m_height >> i, std::string("SubsampledDepthMap_") + std::to_string(i), "Saving subsampled depthmap...");
#endif
			// Store subsampled depth map in pyramid
#if EVAL_MODE
			auto constructPointCloudStart = std::chrono::high_resolution_clock::now();
#endif
			pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, m_width >> i, m_height >> i, i);
#if EVAL_MODE
			auto constructPointCloudEnd = std::chrono::high_resolution_clock::now();
			std::cout << "Computing the point cloud of level " << i << " took: " << std::chrono::duration_cast<std::chrono::milliseconds>(constructPointCloudEnd - constructPointCloudStart).count() << " ms" << std::endl;
#endif
		}
	}

	const std::vector<PointCloud> &getPointClouds() const
	{
		return pointClouds;
	}

private:
	/*
	 * Computes the smoothed depth map using bilinear smoothing.
	 */
	void computeSmoothedDepthMap(const float sigmaR, const float sigmaS)
	{
		// Create row major representation of depth map
		m_smoothedDepthMap = new float[m_width * m_height];
#pragma omp parallel for collapse(2)
		for (int v = 0; v < m_height; ++v)
		{
			for (int u = 0; u < m_width; ++u)
			{
				unsigned int idx = v * m_width + u; // linearized index
				if (rawDepthMap[idx] == MINF)
				{
					m_smoothedDepthMap[idx] = MINF;
					continue;
				}

				float normalizer = 0.0;
				float sum = 0.0;

				// Compute bilinear smoothing only on pixels in window of size m_windowSize
				const int lowerLimitHeight = std::max(v - (m_windowSize / 2), 0);
				const int upperLimitHeight = std::min(v + (m_windowSize / 2) + 1, m_height);
				const int lowerLimitWidth = std::max(u - (m_windowSize / 2), 0);
				const int upperLimitWidth = std::min(u + (m_windowSize / 2) + 1, m_width);
// Compute bilinear smoothing
				for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
				{
					for (int x = lowerLimitWidth; x < upperLimitWidth; ++x)
					{
						unsigned int idxWindow = y * m_width + x; // linearized index
						if (rawDepthMap[idxWindow] == MINF)
						{
							continue;
						}
						float summand = std::exp(-(std::pow((u - x), 2) + std::pow((v - y), 2)) * (1 / std::pow(sigmaS, 2))) * std::exp(-(std::pow(rawDepthMap[idx] - rawDepthMap[idxWindow], 2) * (1 / std::pow(sigmaR, 2))));
						normalizer += summand;
						sum += summand * rawDepthMap[idxWindow];
					}
				}
				m_smoothedDepthMap[idx] = sum / normalizer;
			}
		}
	}

private:
	/*
	 * Computes a subsampled depth map using blockSize. The new depth map has half the width and height of the old one.
	 *
	 * depthMap: The depth map to be subsampled
	 * width: The width of the old depth map
	 * height: The height of the old depth map
	 * sigmaR: Defines depth threshold for block averaging
	 *
	 * Returns pointer to the subsampled depth map.
	 */
	float *subsampleDepthMap(float *depthMap, const unsigned width, const unsigned height, const float sigmaR)
	{
		float threshold = 3 * sigmaR;
		float *blockAverage = new float[(width / 2) * (height / 2)];
#pragma omp parallel for collapse(2)
		for (int v = 0; v < height; v = v + 2)
		{
			for (int u = 0; u < width; u = u + 2)
			{
				unsigned int idx = v * width + u; // linearized index
				float sum = 0.0;

				const int lowerLimitHeight = std::max(v - (m_blockSize / 2), 0);
				const int upperLimitHeight = std::min(v + (m_blockSize / 2) + 1, int(height));
				const int lowerLimitWidth = std::max(u - (m_blockSize / 2), 0);
				const int upperLimitWidth = std::min(u + (m_blockSize / 2) + 1, int(width));
				unsigned blockEntries = (upperLimitHeight - lowerLimitHeight) * (upperLimitWidth - lowerLimitWidth);

				// Compute block average over the blockSize
				for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
				{
					for (int x = lowerLimitWidth; x < upperLimitWidth; ++x)
					{
						unsigned int idxBlock = y * width + x; // linearized index
						if (depthMap[idxBlock] == MINF)
						{
							blockEntries--;
						}
						// If the depth at idx is not defined, we don't care about the threshold.
						else if (depthMap[idx] == MINF || std::abs(depthMap[idxBlock] - depthMap[idx]) <= threshold)
						{
							sum += depthMap[idxBlock];
						}
						else
						{
							blockEntries--;
						}
					}
				}
				blockAverage[(v / 2) * (width / 2) + (u / 2)] = sum / blockEntries;
			}
		}

		if (depthMap != m_smoothedDepthMap)
		{
			delete[] depthMap;
		}
		return blockAverage;
	}
};
