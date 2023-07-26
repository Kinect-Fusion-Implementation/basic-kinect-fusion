#include "PointCloudPyramid.h"

PointCloudPyramid::PointCloudPyramid(float *depthMap, const Matrix3f &depthIntrinsics, const Matrix4f &depthExtrinsics,
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

	// Compute smoothed depth map
	computeSmoothedDepthMap(sigmaS, sigmaR);

	// Setup of pyramid
	//float *currentDepthMap = m_smoothedDepthMap;
	float *depthMapGPU;
	cudaMalloc(&depthMapGPU, width * height * sizeof(float));
	cudaMemcpy(depthMapGPU, depthMap, width * height * sizeof(float), cudaMemcpyHostToDevice);
	float *currentDepthMap = m_smoothedDepthMap;

	pointClouds.reserve(levels + 1);

	// Construct pyramid of pointClouds
	// point cloud takes ownership of depth map after construction, no deletion required!
	pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, m_width, m_height, 0);

	for (size_t i = 0; i < levels;)
	{
		// Compute subsampled depth map
		currentDepthMap = subsampleDepthMap(currentDepthMap, m_width >> i, m_height >> i, sigmaR);
		i++;

		// Store subsampled depth map in pyramid -> pyramid handles deletion of depth map!
		pointClouds.emplace_back(currentDepthMap, depthIntrinsics, depthExtrinsics, m_width >> i, m_height >> i, i);
	}
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
	const int upperLimitHeight = min(h + (windowSize / 2) + 1, height);
	const int lowerLimitWidth = max(w - (windowSize / 2), 0);
	const int upperLimitWidth = min(w + (windowSize / 2) + 1, width);
	// Compute bilinear smoothing
	for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
	{
		for (int x = lowerLimitWidth; x < upperLimitWidth; ++x)
		{
			unsigned int idxWindow = y * width + x; // linearized index
			if (rawDepthMap[idxWindow] == minf)
			{
				continue;
			}
			float summand = std::exp(-(std::pow((w - x), 2) + std::pow((h - y), 2)) * (1 / std::pow(sigmaS, 2))) * std::exp(-(std::pow(rawDepthMap[idx] - rawDepthMap[idxWindow], 2) * (1 / std::pow(sigmaR, 2))));
			normalizer += summand;
			sum += summand * rawDepthMap[idxWindow];
		}
	}
	smoothedDepthMap[idx] = sum / normalizer;
}

void PointCloudPyramid::computeSmoothedDepthMap(const float sigmaR, const float sigmaS)
{
	dim3 threadBlocks(20, 20);
	dim3 blocks(m_width / 20, m_height / 20);
	float *rawDepthMapGPU;
	unsigned int numberDepthValues = m_width * m_height;
	cudaMalloc(&rawDepthMapGPU, numberDepthValues * sizeof(float));
	cudaMalloc(&m_smoothedDepthMap, numberDepthValues * sizeof(float));
	cudaMemcpy(rawDepthMapGPU, rawDepthMap, numberDepthValues * sizeof(float), cudaMemcpyHostToDevice);

	computeSmoothedDepthMapKernel<<<blocks, threadBlocks>>>(rawDepthMapGPU, m_smoothedDepthMap, m_width, m_height, sigmaR, sigmaS, m_windowSize, MINF);
	cudaFree(rawDepthMapGPU);
}

__global__ void subsampleDepthMapKernel(float *depthMap, float *subsampledDepthMap, const unsigned width, const unsigned height, const unsigned blockSize, const float sigmaR, const float minf)
{
	float threshold = 3 * sigmaR;
	// Initially w,h are here values of the output width and height (half of the original)
	unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned h = blockIdx.y * blockDim.y + threadIdx.y;

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
	unsigned blockEntries = (upperLimitHeight - lowerLimitHeight) * (upperLimitWidth - lowerLimitWidth);

	// Compute block average over the blockSize
	for (int y = lowerLimitHeight; y < upperLimitHeight; ++y)
	{
		for (int x = lowerLimitWidth; x < upperLimitWidth; ++x)
		{
			unsigned int blockIdx = y * width + x; // linearized index
			if (depthMap[blockIdx] == minf)
			{
				blockEntries--;
			}
			// If the depth at idx is not defined, we don't care about the threshold.
			else if (depthMap[idx] == minf || std::abs(depthMap[blockIdx] - depthMap[idx]) <= threshold)
			{
				sum += depthMap[blockIdx];
			}
			else
			{
				blockEntries--;
			}
		}
	}
	subsampledDepthMap[(h / 2) * (width / 2) + (w / 2)] = sum / blockEntries;
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