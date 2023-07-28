#include "PointCloud.h"
#include <iostream>

__global__ void computeVerticesKernel(float *depthMap, Vector3f *vertexMap, const Matrix3f depthIntrinsics, const unsigned int width, const unsigned int height, int level, float minf)
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
		vertexMap[idx] = Vector3f((w - cX) / fovX * depth, (h - cY) / fovY * depth, depth);
		/*
		if (w == 353 && h == 206)
		{
			printf("Vertex:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx].x(), vertexMap[idx].y(), vertexMap[idx].z());
		}
		if (w == 352 && h == 206)
		{
			printf("Left Vertex:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx].x(), vertexMap[idx].y(), vertexMap[idx].z());
		}
		if (w == 354 && h == 206)
		{
			printf("Right Vertex:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx].x(), vertexMap[idx].y(), vertexMap[idx].z());
		}
		if (w == 353 && h == 205)
		{
			printf("Upper Vertex:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx].x(), vertexMap[idx].y(), vertexMap[idx].z());
		}
		if (w == 353 && h == 207)
		{
			printf("Lower Vertex:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx].x(), vertexMap[idx].y(), vertexMap[idx].z());
		}
		*/
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
		// Compute the normals that point into the surface
		/*If we use the cross product left-right x lower pixel * upper pixel, the normal vector will point into the plane
		  To see this we can look into the fact that the camera is a RHS and if we use the RH rule we can see this clearly.
		  For proof: Look into the outcommented outputs (points to a pixel on the monitor in the first frame)
		if (w == 353 && h == 207)
		{
			printf("------------------------- In Normal computation ----------------------------------\n");
			printf("Vertex:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx].x(), vertexMap[idx].y(), vertexMap[idx].z());
			printf("Left Vertex:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx - 1].x(), vertexMap[idx - 1].y(), vertexMap[idx - 1].z());
			printf("Right Vertex:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx + 1].x(), vertexMap[idx + 1].y(), vertexMap[idx + 1].z());
			printf("Lower Vertex (larger y):\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx + width].x(), vertexMap[idx + width].y(), vertexMap[idx + width].z());
			printf("Higher Vertex (lower y):\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   vertexMap[idx - width].x(), vertexMap[idx - width].y(), vertexMap[idx - width].z());

			printf("diff left-right ():\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   (vertexMap[idx + 1]-vertexMap[idx - 1]).x(), (vertexMap[idx + 1]-vertexMap[idx - 1]).y(), (vertexMap[idx + 1]-vertexMap[idx - 1]).z());
			printf("diff up-down (in image):\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   (vertexMap[idx + width]-vertexMap[idx - width]).x(), (vertexMap[idx + width]-vertexMap[idx - width]).y(), (vertexMap[idx + width]-vertexMap[idx - width]).z());
			
			printf("Cross vector:\n"
				   "%f\n"
				   "%f\n"
				   "%f\n",
				   ((vertexMap[idx + 1] - vertexMap[idx - 1]).cross(vertexMap[idx + width] - vertexMap[idx - width])).x(), 
				   ((vertexMap[idx + 1] - vertexMap[idx - 1]).cross(vertexMap[idx + width] - vertexMap[idx - width])).y(), 
				   ((vertexMap[idx + 1] - vertexMap[idx - 1]).cross(vertexMap[idx + width] - vertexMap[idx - width])).z());
		}
		*/

		normalMap[idx] = (vertexMap[idx + 1] - vertexMap[idx - 1]).cross(vertexMap[idx + width] - vertexMap[idx - width]);
		// normalMap[idx] = (vertexMap[idx + width] - vertexMap[idx - width]).cross(vertexMap[idx + 1] - vertexMap[idx - 1]);
		normalMap[idx].normalize();
	}
}

__host__ PointCloud::PointCloud(float *depthMap, const Matrix3f &depthIntrinsics, const unsigned int width, const unsigned int height, int level, const unsigned int maxDistance) : m_width(width), m_height(height)
{
	// The provided depthmap should already be located on the DEVICE
	dim3 threadBlocks(20, 20);
	dim3 blocks(width / 20, height / 20);
	size_t m_memorySize = sizeof(Vector3f) * width * height;
	m_depthMap = depthMap;

	cudaMalloc(&m_points, m_memorySize);
	cudaMalloc(&m_normals, m_memorySize);

	computeVerticesKernel<<<blocks, threadBlocks>>>(m_depthMap, m_points, depthIntrinsics, width, height, level, MINF);
	computeNormalsKernel<<<blocks, threadBlocks>>>(m_depthMap, m_points, m_normals, width, height, MINF);

	// Set flag that the device memory still has to be copied
	m_pointsOnCPU = false;
	m_normalsOnCPU = false;
}

__host__ PointCloud::~PointCloud()
{
	cudaFree(m_points);
	cudaFree(m_normals);
	cudaFree(m_depthMap);
	if (m_pointsOnCPU)
		free(m_points_cpu);
	if (m_normalsOnCPU)
		free(m_normals_cpu);
}

__host__ Vector3f *PointCloud::getPointsCPU()
{
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