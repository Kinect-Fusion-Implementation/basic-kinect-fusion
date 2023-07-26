#include "PointCloud.h"
#include <iostream>

__global__ void computeVerticesKernel(float *depthMap, Vector3f *vertexMap, const Matrix3f depthIntrinsics, const Matrix4f depthExtrinsics, const unsigned int width, const unsigned int height, int level, float minf)
{
	unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned h = blockIdx.y * blockDim.y + threadIdx.y;

	if (w > width || h > height)
	{
		printf("returning for: w= %i , h = %i", w, h);
		return;
	}

	// Get depth intrinsics.
	float fovX = depthIntrinsics(0, 0) / pow(2, level);
	float fovY = depthIntrinsics(1, 1) / pow(2, level);
	float cX = depthIntrinsics(0, 2) / pow(2, level);
	float cY = depthIntrinsics(1, 2) / pow(2, level);

	// Compute inverse depth extrinsics.
	Matrix4f depthExtrinsicsInv = depthExtrinsics.inverse();
	Matrix3f rotationInv = depthExtrinsicsInv.block(0, 0, 3, 3);
	Vector3f translationInv = depthExtrinsicsInv.block(0, 3, 3, 1);

	// For every pixel row.
	unsigned int idx = h * width + w; // linearized index
	float depth = depthMap[idx];
	if (depth == minf)
	{
		vertexMap[idx] = Vector3f(minf, minf, minf);
	}
	else
	{
		// Back-projection to camera space.
		vertexMap[idx] = rotationInv * Vector3f((w - cX) / fovX * depth, (h - cY) / fovY * depth, depth) + translationInv;
	}
}

__global__ void computeNormalsKernel(float *depthMap, Vector3f *vertexMap, Vector3f *normalMap, const unsigned int width, const unsigned int height, float minf, const unsigned int maxDistance = 10)
{
	unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned h = blockIdx.y * blockDim.y + threadIdx.y;

	if (w > width || h > height)
	{
		printf("returning for: w= %i , h = %i", w, h);
		return;
	}

	const float maxDistanceHalved = maxDistance / 2.f;
	unsigned int idx = h * width + w; // linearized index

	// We need to compute derivatives and then the normalized normal vector (for valid pixels).
	if (h == 0 || w == 0 || w == (height - 1) || h == (height - 1))
	{
		normalMap[idx] = Vector3f(minf, minf, minf);
		return;
	}

	const float du = 0.5f * (depthMap[idx + 1] - depthMap[idx - 1]);
	const float dv = 0.5f * (depthMap[idx + width] - depthMap[idx - width]);

	if (!isfinite(du) || !isfinite(dv) || abs(du) > maxDistanceHalved || abs(dv) > maxDistanceHalved)
	{
		normalMap[idx] = Vector3f(minf, minf, minf);
	}
	else
	{
		// TODO: Compute the normals using central differences.
		normalMap[idx] = (vertexMap[idx + width] - vertexMap[idx - width]).cross(vertexMap[idx + 1] - vertexMap[idx - 1]);
		normalMap[idx].normalize();
	}
}

__host__ PointCloud::PointCloud(float *depthMap, const Matrix3f &depthIntrinsics, const Matrix4f &depthExtrinsics, const unsigned int width, const unsigned int height, int level, const unsigned int maxDistance) : m_width(width), m_height(height)
{
	// The provided depthmap should already be located on the DEVICE
	dim3 threadBlocks(20, 20);
	dim3 blocks(width / 20, height / 20);
	size_t m_memorySize = sizeof(Vector3f) * width * height;
	m_depthMap = depthMap;
	
	//float *depth;
	//cudaMalloc(&depth, width * height * sizeof(float));
	//cudaMemcpy(depth, depthMap, width * height * sizeof(float), cudaMemcpyHostToDevice);
	//m_depthMap = depth;

	cudaMalloc(&m_points, m_memorySize);
	cudaMalloc(&m_normals, m_memorySize);

	computeVerticesKernel<<<blocks, threadBlocks>>>(m_depthMap, m_points, depthIntrinsics, depthExtrinsics, width, height, level, MINF);

	// Set flag that the device memory still has to be copied
	m_pointsOnCPU = false;
	m_normalsOnCPU = false;
}

__host__ PointCloud::~PointCloud()
{
	cudaFree(m_points);
	cudaFree(m_normals);
	cudaFree(m_depthMap);
	free(m_points_cpu);
	free(m_normals_cpu);
}

__host__ Vector3f *PointCloud::getPointsCPU()
{
	std::cout << "Copy pointcloud to cpu." << std::endl;
	if (!m_pointsOnCPU)
	{
		// Allocate memory for copy
		m_points_cpu = new Vector3f[m_width * m_height];
		cudaMemcpy(m_points_cpu, m_points, m_width * m_height * sizeof(Vector3f), cudaMemcpyDeviceToHost);
		m_pointsOnCPU = true;
	}
	return m_points_cpu;
}

__host__ Vector3f *PointCloud::getNormalsCPU()
{
	if (!m_normalsOnCPU)
	{
		// Allocate memory for copy
		m_normals_cpu = new Vector3f[m_width * m_height];
		cudaMemcpy(m_normals_cpu, m_normals, m_width * m_height * sizeof(Vector3f), cudaMemcpyDeviceToHost);
		m_normalsOnCPU = true;
	}

	return m_normals_cpu;
}