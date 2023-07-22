#include "CudaVoxelGrid.h"
#include <stdio.h>
#include <iostream>

__host__ VoxelGrid::VoxelGrid(Vector3f gridOrigin, unsigned int numberVoxelsWidth,
							  unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight,
							  unsigned int imageHeight, unsigned int imageWidth, float scale, float truncation) : m_gridOriginOffset(gridOrigin.x(), gridOrigin.y(), gridOrigin.z()),
																												  m_numberVoxelsWidth(numberVoxelsWidth), m_numberVoxelsDepth(numberVoxelsDepth),
																												  m_numberVoxelsHeight(numberVoxelsHeight), m_imageHeight(imageHeight), m_voxelGridCPU(nullptr),
																												  m_imageWidth(imageWidth), m_spatialVoxelScale(scale), m_truncation(truncation), m_memorySize(sizeof(Vector3f) * m_imageWidth * m_imageHeight)
{
	std::cout << "Setting up grid" << std::endl;
	unsigned long long numberVoxels = m_numberVoxelsWidth * m_numberVoxelsDepth * m_numberVoxelsHeight;
	cudaMalloc(&m_voxelGrid, sizeof(VoxelData) * numberVoxels);
	cudaMalloc(&m_vertexMapGPU, m_memorySize);
	cudaMalloc(&m_normalMapGPU, m_memorySize);

	VoxelData *cleanGrid = new VoxelData[numberVoxels];
	cudaMemcpy(m_voxelGrid, cleanGrid, sizeof(VoxelData) * numberVoxels, cudaMemcpyHostToDevice);
	// TODO: Check whether we have to initalize the vertex and normal memory also
	delete[] cleanGrid;
}

__host__ VoxelGrid::~VoxelGrid()
{
	std::cout << "Destructing VoxelGrid..." << std::endl;
	if (m_voxelGridCPU != nullptr)
	{
		delete[] m_voxelGridCPU;
	}
	if (m_voxelGrid != nullptr)
	{
		cudaFree(m_voxelGrid);
	}
	if (m_vertexMapGPU != nullptr)
	{
		cudaFree(m_vertexMapGPU);
	}
	if (m_normalMapGPU != nullptr)
	{
		cudaFree(m_normalMapGPU);
	}
	std::cout << "VoxelGrid destructed" << std::endl;
}

__host__ void VoxelGrid::sync()
{
	unsigned long long numberVoxels = m_numberVoxelsWidth * m_numberVoxelsDepth * m_numberVoxelsHeight;
	m_voxelGridCPU = new VoxelData[numberVoxels];
	cudaMemcpy(m_voxelGridCPU, m_voxelGrid, sizeof(VoxelData) * numberVoxels, cudaMemcpyDeviceToHost);
}

__global__ void updateTSDFKernel(Matrix4f extrinsics, Matrix3f intrinsics,
								 float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight,
								 float truncation, float offset_x, float offset_y, float offset_z, float spatialVoxelScale,
								 VoxelData *tsdf, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight)
{
	unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned h = blockIdx.y * blockDim.y + threadIdx.y;
	Matrix4f pose = extrinsics.inverse();
	for (size_t d = 0; d < numberVoxelsDepth; d++)
	{
		Vector3f worldCoordinatesOfGridCell = getWorldCoordinates(w, h, d, offset_x, offset_y, offset_z, spatialVoxelScale);

		Vector3f cameraCoordinatesOfGridCell = extrinsics.block<3, 3>(0, 0) * worldCoordinatesOfGridCell;

		cameraCoordinatesOfGridCell += extrinsics.block<3, 1>(0, 3);
		// voxel is behind or on the camera plane
		// Maybe change this to only work for pixels that are at least the focal length away from the camera center?
		if (cameraCoordinatesOfGridCell.z() <= 0)
		{
			continue;
		}
		Vector3f pixelCoordinates = (intrinsics * cameraCoordinatesOfGridCell) / cameraCoordinatesOfGridCell.z();
		assert(pixelCoordinates.z() == 1);

		// Ensure that the voxel is visible in the camera
		if (pixelCoordinates.x() < 0 || pixelCoordinates.y() < 0 || pixelCoordinates.x() >= depthMapWidth || pixelCoordinates.y() >= depthMapHeight)
		{
			continue;
		}
		// find depth in depthmap that is stored in row major
		float depth = depthMap[static_cast<int>(pixelCoordinates.x()) + static_cast<int>(pixelCoordinates.y()) * depthMapWidth];
		float distanceOfVoxelToCamera = (worldCoordinatesOfGridCell - pose.block<3, 1>(0, 3)).norm();
		if (depth > distanceOfVoxelToCamera)
		{
			// Our voxel is in front of the surface from the view of the camera
			getVoxelData(tsdf, w, h, d, numberVoxelsDepth, numberVoxelsHeight).freeSpace++;
		}

		// There is an alternative formulation to this in the second paper...
		// This will be > 0 for points between camera and surface and < 0 for points behind the surface
		float lambda = (intrinsics.inverse() * Vector3f(pixelCoordinates.x(), pixelCoordinates.y(), 1)).norm();
		float sdfEstimate = min(max(depth - distanceOfVoxelToCamera / lambda, -truncation), truncation);

		if (fabsf(sdfEstimate) < truncation)
		{
			VoxelData &voxel = getVoxelData(tsdf, w, h, d, numberVoxelsDepth, numberVoxelsHeight);
			// Just like in the paper
			float newWeight = 1;
			voxel.depthAverage = (voxel.depthAverage * voxel.weights + sdfEstimate * newWeight) / (voxel.weights + newWeight);
			voxel.weights += newWeight;
		}
	}
}

__host__ void VoxelGrid::updateTSDF(Matrix4f extrinsics, Matrix3f intrinsics, float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight)
{
	// Assume 240x240x240
	dim3 threadBlocks(20, 20);
	dim3 blocks(m_numberVoxelsWidth / 20, m_numberVoxelsHeight / 20);
	float *depthDataGPU;
	cudaMalloc(&depthDataGPU, sizeof(float) * depthMapWidth * depthMapHeight);
	cudaMemcpy(depthDataGPU, depthMap, sizeof(float) * depthMapWidth * depthMapHeight, cudaMemcpyHostToDevice);

	updateTSDFKernel<<<blocks, threadBlocks>>>(extrinsics, intrinsics, depthDataGPU, depthMapWidth, depthMapHeight, m_truncation, m_gridOriginOffset.x(), m_gridOriginOffset.y(), m_gridOriginOffset.z(), m_spatialVoxelScale, m_voxelGrid, m_numberVoxelsDepth, m_numberVoxelsHeight);
	cudaGetLastError();
	cudaDeviceSynchronize();
}

__global__ void raycastVoxelGridKernel(Matrix4f poseMatrix, Matrix4f extrinsics, Matrix3f intrinsics, Vector3f *vertexMap, Vector3f *normalMap,
									   Vector3f gridOffset, float spatialVoxelScale,
									   VoxelData *tsdf, unsigned int numberVoxelsDepth, unsigned int numberVoxelsWidth, unsigned int numberVoxelsHeight,
									   unsigned int imageWidth, float truncation, float mINF)
{
	unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned h = blockIdx.y * blockDim.y + threadIdx.y;
	{
		// First use the homogeneouse pixel coordinates
		Vector3f coordinates(w, h, 1);
		// Compute screen space coordinates with unit value along z axis
		coordinates = intrinsics.inverse() * coordinates;
		// Compute the point on the camera screen in world coordinates
		coordinates = poseMatrix.block<3, 3>(0, 0) * coordinates;
		Vector3f rayDirection = coordinates - poseMatrix.block<3, 1>(0, 3);
		// Ray of length 1 meter
		rayDirection.normalize();
		// We start tracing only 40 centimeters away from the camera
		float distanceTravelled = 0.4;
		Vector3f rayPosition = poseMatrix.block<3, 1>(0, 3) + distanceTravelled * rayDirection;

		// Vector3i gridCoordinates = getGridCoordinates(rayPosition);
		Vector3i gridCoordinates = getGridCoordinates(rayPosition, gridOffset, spatialVoxelScale);
		if (gridCoordinates.x() < 0 || gridCoordinates.x() >= numberVoxelsWidth || gridCoordinates.y() < 0 || gridCoordinates.y() >= numberVoxelsHeight || gridCoordinates.z() < 0 || gridCoordinates.z() >= numberVoxelsDepth)
		{
			return;
		}
		float initialTSDFValue = getVoxelData(tsdf, gridCoordinates.x(), gridCoordinates.y(), gridCoordinates.z(), numberVoxelsDepth, numberVoxelsHeight).depthAverage;
		if (initialTSDFValue < 0)
		{
			// Camera is too close to the surface
			vertexMap[w + h * imageWidth] = Vector3f(mINF, mINF, mINF);
			normalMap[w + h * imageWidth] = Vector3f(mINF, mINF, mINF);
			return;
		}

		while (distanceTravelled < 8.0)
		{
			// We assume that we are in the volume
			if (gridCoordinates.x() < 0 || gridCoordinates.x() >= numberVoxelsWidth || gridCoordinates.y() < 0 || gridCoordinates.y() >= numberVoxelsHeight || gridCoordinates.z() < 0 || gridCoordinates.z() >= numberVoxelsDepth)
			{
				// std::cout << "Outside of voxel grid";
				vertexMap[w + h * imageWidth] = Vector3f(mINF, mINF, mINF);
				normalMap[w + h * imageWidth] = Vector3f(mINF, mINF, mINF);
				return;
			}
			// Set up next step
			float stepSize = 0;
			float currentTSDFValue = getVoxelData(tsdf, gridCoordinates.x(), gridCoordinates.y(), gridCoordinates.z(), numberVoxelsDepth, numberVoxelsHeight).depthAverage;
			if (currentTSDFValue >= truncation)
			{
				stepSize = truncation;
				rayPosition = rayPosition + stepSize * rayDirection;
				distanceTravelled += stepSize;
				// There can be no interface here as we are >= truncation away from a surface
				continue;
			}
			// If we are close to a surface, make a small step
			stepSize = max(spatialVoxelScale, currentTSDFValue) * 0.5;
			distanceTravelled += stepSize;
			Vector3f newRayPosition = rayPosition + stepSize * rayDirection;
			Vector3i newGridCoordinates = getGridCoordinates(newRayPosition, gridOffset, spatialVoxelScale);
			if (newGridCoordinates.x() < 0 || newGridCoordinates.x() >= numberVoxelsWidth || newGridCoordinates.y() < 0 || newGridCoordinates.y() >= numberVoxelsHeight || newGridCoordinates.z() < 0 || newGridCoordinates.z() >= numberVoxelsDepth)
			{
				break;
			}
			float newTSDFValue = getVoxelData(tsdf, newGridCoordinates.x(), newGridCoordinates.y(), newGridCoordinates.z(), numberVoxelsDepth, numberVoxelsHeight).depthAverage;
			if (newTSDFValue < 0)
			{
				// Interpolation formula page 6 in the paper
				float interpolatedStepSize = stepSize * (currentTSDFValue / (currentTSDFValue - newTSDFValue));
				newRayPosition = rayPosition + interpolatedStepSize * rayDirection;
				newGridCoordinates = getGridCoordinates(newRayPosition, gridOffset, spatialVoxelScale);
				float depthValue = getVoxelData(tsdf, newGridCoordinates.x(), newGridCoordinates.y(), newGridCoordinates.z(), numberVoxelsDepth, numberVoxelsHeight).depthAverage;
				// On average the distance between measurements is the distance of voxel centers -> the voxel scale
				// Here we could definitely improve on accuracy
				if (newGridCoordinates.x() - 1 < 0 || newGridCoordinates.y() - 1 < 0 || newGridCoordinates.z() - 1 < 0)
				{
					break;
				}
				float deltaH = spatialVoxelScale;
				vertexMap[w + h * imageWidth] = newRayPosition;
				float deltaX = (getVoxelData(tsdf, newGridCoordinates.x() - 1, newGridCoordinates.y(), newGridCoordinates.z(), numberVoxelsDepth, numberVoxelsHeight).depthAverage - depthValue);
				float deltaY = (getVoxelData(tsdf, newGridCoordinates.x(), newGridCoordinates.y() - 1, newGridCoordinates.z(), numberVoxelsDepth, numberVoxelsHeight).depthAverage - depthValue);
				float deltaZ = (getVoxelData(tsdf, newGridCoordinates.x(), newGridCoordinates.y(), newGridCoordinates.z() - 1, numberVoxelsDepth, numberVoxelsHeight).depthAverage - depthValue);
				normalMap[w + h * imageWidth] = extrinsics.block<3, 3>(0, 0) * Vector3f(deltaX / deltaH, deltaY / deltaH, deltaZ / deltaH);

				// For now we just normalize these...
				normalMap[w + h * imageWidth].normalize();
				break;
			}
			rayPosition = newRayPosition;
			gridCoordinates = newGridCoordinates;
		}
	}
}

__host__ RaycastImage VoxelGrid::raycastVoxelGrid(Matrix4f extrinsics, Matrix3f intrinsics)
{
	dim3 threadBlocks(20, 20);
	dim3 blocks(m_imageWidth / 20, m_imageHeight / 20);
	RaycastImage image(m_imageWidth, m_imageHeight);
	// Clean GPU memory:
	cudaMemcpy(m_vertexMapGPU, image.m_vertexMap, m_memorySize, cudaMemcpyHostToDevice);
	cudaMemcpy(m_normalMapGPU, image.m_normalMap, m_memorySize, cudaMemcpyHostToDevice);
	raycastVoxelGridKernel<<<blocks, threadBlocks>>>(extrinsics.inverse(), extrinsics, intrinsics,
													 m_vertexMapGPU, m_normalMapGPU, m_gridOriginOffset,
													 m_spatialVoxelScale, m_voxelGrid,
													 m_numberVoxelsDepth, m_numberVoxelsWidth, m_numberVoxelsHeight,
													 m_imageWidth, m_truncation, MINF);
	cudaMemcpy(image.m_vertexMap, m_vertexMapGPU, m_memorySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(image.m_normalMap, m_normalMapGPU, m_memorySize, cudaMemcpyDeviceToHost);
	return image;
}

__device__ Vector3f getWorldCoordinates(int x, int y, int z, float offset_x, float offset_y, float offset_z, float spatialVoxelScale)
{
	Vector3f gridCellCoordinates(float(x), float(y), float(z));
	float xf = (x + 0.5) * spatialVoxelScale + offset_x;
	float yf = (y + 0.5) * spatialVoxelScale + offset_y;
	float zf = (z + 0.5) * spatialVoxelScale + offset_z;
	// grid origin is the offset from the world frame origin (point in voxel grid frame is offset by exactly that vector)
	return Vector3f(xf, yf, zf);
}

__device__ VoxelData &getVoxelData(VoxelData *voxelGrid, int w, unsigned int h, unsigned int d, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight)
{
	return voxelGrid[d + numberVoxelsDepth * h + numberVoxelsDepth * numberVoxelsHeight * w];
}

__device__ Vector3i getGridCoordinates(Vector3f worldCoordinates, Vector3f gridOriginOffset, float spatialVoxelScle)
{
	Vector3f coordinates = worldCoordinates - gridOriginOffset;
	coordinates = coordinates / spatialVoxelScle;
	return Vector3i(int(coordinates.x()), int(coordinates.y()), int(coordinates.z()));
}

__host__ VoxelData &VoxelGrid::getVoxelData(unsigned int w, unsigned int h, unsigned int d)
{
	return m_voxelGridCPU[d + m_numberVoxelsDepth * h + m_numberVoxelsDepth * m_numberVoxelsHeight * w];
}

__host__ Vector3i VoxelGrid::getGridCoordinates(Vector3f worldCoordinates)
{
	Vector3f coordinates = worldCoordinates - m_gridOriginOffset;
	coordinates = coordinates / m_spatialVoxelScale;
	return Vector3i(int(coordinates.x()), int(coordinates.y()), int(coordinates.z()));
}

__host__ Vector3f VoxelGrid::getCellCenterInWorldCoords(Vector3i gridCell)
{
	Vector3f gridCellCoordinates(float(gridCell.x()), float(gridCell.y()), float(gridCell.z()));
	Vector3f centerOffset(0.5, 0.5, 0.5);
	gridCellCoordinates += centerOffset;
	gridCellCoordinates *= m_spatialVoxelScale;
	// grid origin is the offset from the world frame origin (point in voxel grid frame is offset by exactly that vector)
	return gridCellCoordinates + m_gridOriginOffset;
}
