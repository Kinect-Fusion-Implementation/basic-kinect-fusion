#include "PointCloudPyramid.h"
#include <assert.h>

PointCloudPyramid::PointCloudPyramid(float *depthMap, const Matrix3f &depthIntrinsics,
									 const unsigned int width, const unsigned int height, const unsigned int levels,
									 const unsigned int windowSize, const unsigned int blockSize, const float sigmaR, const float sigmaS)
	: rawDepthMap(depthMap), m_width(width), m_height(height), m_windowSize(windowSize), m_blockSize(blockSize)
{
	// Input validation
	assert(m_width > 0);
	assert(m_height > 0);
	assert(m_windowSize > 0);
	assert(m_windowSize % 2 == 1);
	assert(m_blockSize % 2 == 1);

	// Compute smoothed depth
	computeSmoothedDepthMap(sigmaS, sigmaR);
	// Setup of pyramid
	float *currentDepthMap = m_smoothedDepthMap;

	pointClouds.reserve(levels + 1);

	// Construct pyramid of pointClouds
	// point cloud takes ownership of depth map after construction, no deletion required!

	pointClouds.emplace_back(currentDepthMap, depthIntrinsics, m_width, m_height, 0);
	for (size_t i = 0; i < levels;)
	{
		// Compute subsampled depth map
		currentDepthMap = subsampleDepthMap(currentDepthMap, m_width >> i, m_height >> i, sigmaR);
		i++;

		// Store subsampled depth map in pyramid -> pyramid handles deletion of depth map!
		pointClouds.emplace_back(currentDepthMap, depthIntrinsics, m_width >> i, m_height >> i, i);
	}
}

__host__ PointCloudPyramid::~PointCloudPyramid()
{
	cudaFree(m_rawDepthMapGPU);
}

__global__ void computeSmoothedDepthMapKernel(float *rawDepthMap, float *smoothedDepthMap,
											  const unsigned int width, const unsigned int height,
											  const float sigmaR, const float sigmaS,
											  unsigned int windowSize, const float minf)
{
	unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned h = blockIdx.y * blockDim.y + threadIdx.y;
	if (w > width || h > height)
	{
		printf("returning for: w= %i , h = %i", w, h);
		return;
	}
	unsigned int idx = h * width + w; // linearized index
	if (rawDepthMap[idx] == minf)
	{
		smoothedDepthMap[idx] = minf;
		return;
	}

	float normalizer = 0.0;
	float sum = 0.0;

	// Compute bilinear smoothing only on pixels in window of size m_windowSize
	const int lowerLimitHeight = max(h - (windowSize / 2), 0);
	const int lowerLimitWidth = max(w - (windowSize / 2), 0);
	// Compute bilinear smoothing
	for (int j = 0; j < windowSize * windowSize; j++)
	{
		int x = lowerLimitWidth + (j % windowSize);
		int y = lowerLimitHeight + int(j / windowSize);
		unsigned int idxWindow = x + width * y;
		if (x > width || y > height || rawDepthMap[idxWindow] == minf)
		{
			continue;
		}
		// float summand = std::exp(-(std::pow((w - x), 2) + std::pow((h - y), 2)) * (1 / std::pow(sigmaS, 2))) * std::exp(-(std::pow(rawDepthMap[idx] - rawDepthMap[idxWindow], 2) * (1 / std::pow(sigmaR, 2))));
		float wxs = (w - x) * (w - x);
		float hys = (h - y) * (h - y);
		float sigmaSs = (sigmaS * sigmaS);
		float sigmaRs = (sigmaR * sigmaR);
		float dds = (rawDepthMap[idx] - rawDepthMap[idxWindow]) * (rawDepthMap[idx] - rawDepthMap[idxWindow]);
		float summand = exp(-(wxs + hys) * (1 / sigmaSs)) * exp(-(dds * (1 / sigmaRs)));
		normalizer += summand;
		sum += summand * rawDepthMap[idxWindow];
	}

	smoothedDepthMap[idx] = sum / normalizer;
}

void PointCloudPyramid::computeSmoothedDepthMap(const float sigmaR, const float sigmaS)
{
	dim3 threadBlocks(20, 20);
	dim3 blocks(m_width / 20, m_height / 20);
	unsigned int numberDepthValues = m_width * m_height;
	cudaMalloc(&m_rawDepthMapGPU, numberDepthValues * sizeof(float));
	cudaMalloc(&m_smoothedDepthMap, numberDepthValues * sizeof(float));
	cudaMemcpy(m_rawDepthMapGPU, rawDepthMap, numberDepthValues * sizeof(float), cudaMemcpyHostToDevice);
	computeSmoothedDepthMapKernel<<<blocks, threadBlocks>>>(m_rawDepthMapGPU, m_smoothedDepthMap, m_width, m_height, sigmaR, sigmaS, m_windowSize, MINF);
}

__global__ void subsampleDepthMapKernel(float *depthMap, float *subsampledDepthMap, const unsigned width, const unsigned height, const unsigned blockSize, const float sigmaR, const float minf)
{
	float threshold = 3 * sigmaR;
	// Initially w,h are here values of the output width and height (half of the original)
	unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;

	if (w > (width / 2) || h > (height / 2))
	{
		printf("returning for: w= %i , h = %i", w, h);
		return;
	}
	// Since we use w,h here as the width height on the input, we have to double it
	w = w * 2;
	h = h * 2;

	unsigned int idx = h * width + w; // linearized index
	float sum = 0.0;

	const int lowerLimitHeight = max(h - (blockSize / 2), 0);
	const int upperLimitHeight = min(h + (blockSize / 2) + 1, int(height));
	const int lowerLimitWidth = max(w - (blockSize / 2), 0);
	const int upperLimitWidth = min(w + (blockSize / 2) + 1, int(width));
	unsigned int entries = 0;

	// Compute block average over the blockSize
	for (int j = 0; j < blockSize * blockSize; j++)
	{
		int x = lowerLimitWidth + (j % blockSize);
		int y = lowerLimitHeight + int(j / blockSize);
		unsigned int blockIdx = x + width * y;
		if (x >= width || y >= height || depthMap[blockIdx] == minf)
		{
			continue;
		}
		// If the depth at idx is not defined, we don't care about the threshold.
		else if (depthMap[idx] == minf || abs(depthMap[blockIdx] - depthMap[idx]) <= threshold)
		{
			sum += depthMap[blockIdx];
			entries++;
		}
		else
		{
			continue;
		}
	}
	if (entries == 0)
	{
		subsampledDepthMap[(h / 2) * (width / 2) + (w / 2)] = minf;
	}
	else
	{
		subsampledDepthMap[(h / 2) * (width / 2) + (w / 2)] = sum / entries;
	}
}

float *PointCloudPyramid::subsampleDepthMap(float *depthMap, const unsigned width, const unsigned height, const float sigmaR)
{
	dim3 threadBlocks(20, 20);
	dim3 blocks((width / 2) / 20, (height / 2) / 20);
	float *subsampledDepthMap;
	cudaMalloc(&subsampledDepthMap, (width / 2) * (height / 2) * sizeof(float));

	subsampleDepthMapKernel<<<blocks, threadBlocks>>>(depthMap, subsampledDepthMap, width, height, m_blockSize, sigmaR, MINF);
	return subsampledDepthMap;
}