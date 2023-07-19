#include "CudaVoxelGrid.h"
#include <stdio.h>
#include <iostream>

__host__ VoxelGrid::VoxelGrid(Vector3f gridOrigin, unsigned int numberVoxelsWidth, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight, unsigned int imageHeight, unsigned int imageWidth, float scale, float truncation) : m_gridOriginOffset(gridOrigin.x(), gridOrigin.y(), gridOrigin.z()), m_numberVoxelsWidth(numberVoxelsWidth), m_numberVoxelsDepth(numberVoxelsDepth), m_numberVoxelsHeight(numberVoxelsHeight), m_imageHeight(imageHeight), m_imageWidth(imageWidth), m_spatialVoxelScale(scale), m_truncation(truncation)
{
	std::cout << "Setting up grid" << std::endl;
	unsigned long long numberVoxels = m_numberVoxelsWidth * m_numberVoxelsDepth * m_numberVoxelsHeight;
	std::cout << "Voxel width, height, depth: " << numberVoxelsWidth << " " << numberVoxelsHeight << " " << numberVoxelsHeight << std::endl;
	cudaMalloc(&m_voxelGrid, sizeof(VoxelData) * numberVoxels);
	std::cout << "Allocating " << sizeof(VoxelData) * numberVoxels << "bytes." << std::endl;
	
	VoxelData *cleanGrid = new VoxelData[numberVoxels];
	cudaMemcpy(m_voxelGrid, cleanGrid, sizeof(VoxelData) * numberVoxels, cudaMemcpyHostToDevice);
	std::cout << "Setting up grid done" << std::endl;
	delete[] cleanGrid;
}

__host__ VoxelGrid::~VoxelGrid()
{
	cudaFree(m_voxelGrid);
	delete[] m_voxelGridCPU;
}

__host__ void VoxelGrid::sync()
{
	unsigned long long numberVoxels = m_numberVoxelsWidth * m_numberVoxelsDepth * m_numberVoxelsHeight;
	m_voxelGridCPU = new VoxelData[numberVoxels];
	cudaMemcpy(m_voxelGridCPU, m_voxelGrid, sizeof(VoxelData) * numberVoxels, cudaMemcpyDeviceToHost);
}

__global__ void updateTSDFKernel(Matrix4f extrinsics, Matrix3f intrinsics, float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight, float truncation, float offset_x, float offset_y, float offset_z, float spatialVoxelScale, VoxelData *tsdf, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight)
{
	unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned h = blockIdx.y * blockDim.y + threadIdx.y;
	Matrix4f pose = extrinsics.inverse();
	for (size_t d = 0; d < numberVoxelsDepth; d++)
	{
		// std::cout << "Processing voxel: (" << w << ", " << h << ", " << d << ") (width, height, depth)" << std::endl;
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
		// This will be >0 for points between camera and surface and < 0 for points behind the surface
		float sdfEstimate = min(max(depth - distanceOfVoxelToCamera, -truncation), truncation);

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

__host__ void VoxelGrid::updateTSDF(Matrix4f extrinsics, Matrix3f intrinsics, float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight)
{
	// Assume 240x240x240
	dim3 threadBlocks(20, 20);
	dim3 blocks(12, 12);
	float *depthDataGPU;
	cudaMalloc(&depthDataGPU, sizeof(float) * depthMapWidth * depthMapHeight);
	cudaMemcpy(depthDataGPU, depthMap, sizeof(float) * depthMapWidth * depthMapHeight, cudaMemcpyHostToDevice);

	updateTSDFKernel<<<blocks, threadBlocks>>>(extrinsics, intrinsics, depthDataGPU, depthMapWidth, depthMapHeight, m_truncation, this->m_gridOriginOffset.x(), this->m_gridOriginOffset.y(), this->m_gridOriginOffset.z(), this->m_spatialVoxelScale, m_voxelGrid, m_numberVoxelsDepth, m_numberVoxelsHeight);
	cudaGetLastError();
	cudaDeviceSynchronize();
}

__host__ RaycastImage VoxelGrid::raycastVoxelGrid(Matrix4f extrinsics, Matrix3f intrinsics)
{
	RaycastImage result = RaycastImage(m_imageWidth, m_imageHeight);

	Matrix4f poseMatrix = extrinsics.inverse();

	#pragma omp parallel for collapse(2)
	for (size_t w = 0; w < m_imageWidth; w++)
	{
		for (size_t h = 0; h < m_imageHeight; h++)
		{
			// First use the homogeneouse pixel coordinates
			Vector3f coordinates(w, h, 1);
			// Compute screen space coordinates with unit value along z axis
			coordinates = intrinsics.inverse() * coordinates;
			// Compute the point on the camera screen in world coordinates
			coordinates = poseMatrix.block<3,3>(0,0) * coordinates;
			Vector3f rayDirection = coordinates - poseMatrix.block<3, 1>(0, 3);
			// Ray of length 1 meter
			rayDirection.normalize();
			// We start tracing only 40 centimeters away from the camera
			float distanceTravelled = 0.4;
			Vector3f rayPosition = poseMatrix.block<3, 1>(0, 3) + distanceTravelled * rayDirection;

			Vector3i gridCoordinates = getGridCoordinates(rayPosition);
			if (gridCoordinates.x() < 0 || gridCoordinates.x() >= m_numberVoxelsWidth || gridCoordinates.y() < 0 || gridCoordinates.y() >= m_numberVoxelsHeight || gridCoordinates.z() < 0 || gridCoordinates.z() >= m_numberVoxelsDepth)
			{
				std::cout << "Start position outside of voxel grid" << std::endl;
				continue;
			}
			float initialTSDFValue = this->getVoxelData(gridCoordinates.x(), gridCoordinates.y(), gridCoordinates.z()).depthAverage;
			if (initialTSDFValue < 0)
			{
				// Camera is too close to the surface
				continue;
			}

			while (distanceTravelled < 8.0)
			{
				// We assume that we are in the volume
				if (gridCoordinates.x() < 0 || gridCoordinates.x() >= m_numberVoxelsWidth || gridCoordinates.y() < 0 || gridCoordinates.y() >= m_numberVoxelsHeight || gridCoordinates.z() < 0 || gridCoordinates.z() >= m_numberVoxelsDepth)
				{
					// std::cout << "Outside of voxel grid";
					break;
				}
				// Set up next step
				float stepSize = 0;
				float currentTSDFValue = this->getVoxelData(gridCoordinates.x(), gridCoordinates.y(), gridCoordinates.z()).depthAverage;
				if (currentTSDFValue >= m_truncation)
				{
					stepSize = m_truncation;
					rayPosition = rayPosition + stepSize * rayDirection;
					distanceTravelled += stepSize;
					// There can be no interface here as we are >= truncation away from a surface
					continue;
				}
				// If we are close to a surface, make a small step
				stepSize = std::max(m_spatialVoxelScale, currentTSDFValue) * 0.5;
				distanceTravelled += stepSize;
				Vector3f newRayPosition = rayPosition + stepSize * rayDirection;
				Vector3i newGridCoordinates = getGridCoordinates(newRayPosition);
				if (newGridCoordinates.x() < 0 || newGridCoordinates.x() >= m_numberVoxelsWidth || newGridCoordinates.y() < 0 || newGridCoordinates.y() >= m_numberVoxelsHeight || newGridCoordinates.z() < 0 || newGridCoordinates.z() >= m_numberVoxelsDepth)
				{
					break;
				}
				float newTSDFValue = this->getVoxelData(newGridCoordinates.x(), newGridCoordinates.y(), newGridCoordinates.z()).depthAverage;
				if (newTSDFValue < 0)
				{
					// Interpolation formula page 6 in the paper
					float interpolatedStepSize = stepSize * (currentTSDFValue / (currentTSDFValue - newTSDFValue));
					rayPosition = rayPosition + interpolatedStepSize * rayDirection;
					newGridCoordinates = getGridCoordinates(rayPosition);
					float depthValue = getVoxelData(newGridCoordinates.x(), newGridCoordinates.y(), newGridCoordinates.z()).depthAverage;
					// On average the distance between measurements is the distance of voxel centers -> the voxel scale 
					// Here we could definitely improve on accuracy
					float deltaH = m_spatialVoxelScale;
					if (newGridCoordinates.x() + 1 >= m_numberVoxelsWidth || newGridCoordinates.y() + 1 >= m_numberVoxelsHeight || newGridCoordinates.z() + 1 >= m_numberVoxelsDepth)
					{
						break;
					}
					result.vertexMap[w + h * m_imageWidth] = rayPosition;
					float deltaX = (getVoxelData(newGridCoordinates.x() + 1, newGridCoordinates.y(), newGridCoordinates.z()).depthAverage - depthValue);
					float deltaY = (getVoxelData(newGridCoordinates.x(), newGridCoordinates.y() + 1, newGridCoordinates.z()).depthAverage - depthValue);
					float deltaZ = (getVoxelData(newGridCoordinates.x(), newGridCoordinates.y(), newGridCoordinates.z() + 1).depthAverage - depthValue);
					result.normalMap[w + h * m_imageWidth] = extrinsics.block<3,3>(0,0) * Vector3f(deltaX / deltaH, deltaY / deltaH, deltaZ / deltaH);
	
					// For now we just normalize these...
					result.normalMap[w + h * m_imageWidth].normalize();
					break;
				}
				rayPosition = newRayPosition;
				gridCoordinates = newGridCoordinates;
			}
		}
	}

	return result;
}

__host__ VoxelData &VoxelGrid::getVoxelData(unsigned int w, unsigned int h, unsigned int d)
{
	return m_voxelGridCPU[d + m_numberVoxelsDepth * h + m_numberVoxelsDepth * m_numberVoxelsHeight * w];
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
