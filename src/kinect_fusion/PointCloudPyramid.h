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
	// Smoothed depth map stored in DEVICE memory, note that the ownership of this data is handed down to the depth map
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
	/*
	 * Constructs a pyramid of pointClouds. The first level gets smoothed and the others subsampled.
	 *
	 * depthMap: The original depth map, stored in device memory, owned by the sensor! Do not free this!
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
	PointCloudPyramid(float *depthMap, const Matrix3f &depthIntrinsics, const Matrix4f &depthExtrinsics,
					  const unsigned int width, const unsigned int height, const unsigned int levels,
					  const unsigned int windowSize, const unsigned int blockSize, const float sigmaR, const float sigmaS);

	std::vector<PointCloud> &getPointClouds()
	{
		return pointClouds;
	}

private:
	/*
	 * Computes the smoothed depth map using bilinear smoothing.
	 */
	void computeSmoothedDepthMap(const float sigmaR, const float sigmaS);

	/*
	 * Computes a subsampled depth map using blockSize. The new depth map has half the width and height of the old one.
	 *
	 * depthMap: The depth map to be subsampled, points to GPU memory already
	 * width: The width of the old depth map
	 * height: The height of the old depth map
	 * sigmaR: Defines depth threshold for block averaging
	 *
	 * Returns pointer to the subsampled depth map.
	 */
	float *subsampleDepthMap(float *depthMap, const unsigned width, const unsigned height, const float sigmaR);
};
