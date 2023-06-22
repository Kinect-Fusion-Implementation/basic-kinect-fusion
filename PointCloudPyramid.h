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
	float* rawDepthMap;
	// Smoothed depth map
	float* m_smoothedDepthMap;
	// Width of raw and smoothed depth map
	int m_width;
	// Height of raw and smoothed depth map
	int m_height;
	// Sidelength of square considered for smoothing
	int m_windowSize;
	// Sidelength of square considered for subsampling
	int m_blockSize;

private:
	PointCloudPyramid() {}

	/*
	* Prints a 2D float array on the console.
	*/
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
	PointCloudPyramid(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, const unsigned int width, const unsigned int height, const unsigned int levels, const unsigned int windowSize, const unsigned int blockSize, const float sigmaR, const float sigmaS) : rawDepthMap(depthMap), m_width(width), m_height(height), m_windowSize(windowSize), m_blockSize(blockSize)
	{
		// Input validation
		assert(m_width > 0);
		assert(m_height > 0);
		assert(m_windowSize > 0);
		assert(m_windowSize % 2 == 1);
		assert(m_blockSize % 2 == 1);

		// Compute smoothed depth map
		computeSmoothedDepthMap(sigmaR, sigmaS);

		// Print smoothed depth map to file
		FreeImage smoothedDepthImage(m_width, m_height, 1);
		smoothedDepthImage.data = m_smoothedDepthMap;
		std::cout << "Saving smoothed depthmap... " << std::endl;
		// Dominik: std::string fileName("./Output/SmoothedDepthMap");
		std::string fileName("../Output/SmoothedDepthMap");
		smoothedDepthImage.SaveDepthMapToFile(fileName + std::to_string(0) + ".png");

		// Setup of pyramid
		float* currentDepthMap = m_smoothedDepthMap;
		pointClouds.reserve(levels + 1);

		// Construct pyramid of pointClouds
		pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, m_width, m_height, 0);
		for (size_t i = 0; i < levels;)
		{
			// Compute subsampled depth map
			currentDepthMap = subsampleDepthMap(currentDepthMap, m_width >> i, m_height >> i, sigmaR);
			i++;

			// Print subsampled depth map to file
			FreeImage subsampledDepthImage(m_width >> i, m_height >> i, 1);
			std::cout << "Saving subsampled depthmap... " << std::endl;
			// Dominik: std::string fileName("../Output/SubsampledDepthMap");
			std::string fileName("./Output/SubsampledDepthMap");
			subsampledDepthImage.data = currentDepthMap;
			subsampledDepthImage.SaveDepthMapToFile(fileName + std::to_string(i) + ".png");

			// Store subsampled depth map in pyramid
			pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, m_width >> i, m_height >> i, i);
		}
	}

	const std::vector<PointCloud>& getPointClouds() const
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
#pragma omp parallel for
		for (int v = 0; v < m_height; ++v)
		{
			for (int u = 0; u < m_width; ++u)
			{
				unsigned int idx = v * m_width + u; // linearized index
				if (rawDepthMap[idx] == MINF) {
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
						if (rawDepthMap[idxWindow] == MINF) {
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
	float* subsampleDepthMap(float* depthMap, const unsigned width, const unsigned height, const float sigmaR)
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

				const int lowerLimitHeight = std::max(v - (m_blockSize / 2), 0);
				const int upperLimitHeight = std::min(v + (m_blockSize / 2) + 1, int(height));
				const int lowerLimitWidth = std::max(u - (m_blockSize / 2), 0);
				const int upperLimitWidth = std::min(u + (m_blockSize / 2) + 1, int(width));
				unsigned blockEntries = (upperLimitHeight - lowerLimitHeight) * (upperLimitWidth - lowerLimitWidth);

				// Compute block average over the blockSize
				for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
				{
					for (int x = lowerLimitWidth; x < upperLimitWidth; ++x) {
						unsigned int idxBlock = y * width + x; // linearized index
						if (depthMap[idxBlock] == MINF) {
							blockEntries--;
						}
						// If the depth at idx is not defined, we don't care about the threshold.
						else if (depthMap[idx] == MINF || std::abs(depthMap[idxBlock] - depthMap[idx]) <= threshold) {
							sum += depthMap[idxBlock];
						}
						else {
							blockEntries--;
						}
					}
				}
				blockAverage[(v / 2) * (width / 2) + (u / 2)] = sum / blockEntries;
			}
		}

		if (depthMap != m_smoothedDepthMap) {
			delete[] depthMap;
		}
		return blockAverage;
	}
};
