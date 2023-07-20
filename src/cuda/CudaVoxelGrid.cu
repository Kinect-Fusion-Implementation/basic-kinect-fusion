#include "CudaVoxelGrid.h"
#include <stdio.h>
#include <iostream>

__host__ VoxelGrid::VoxelGrid(Vector3f gridOrigin, unsigned int numberVoxelsWidth, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight, unsigned int imageHeight, unsigned int imageWidth, float scale) : m_gridOriginOffset(gridOrigin.x(), gridOrigin.y(), gridOrigin.z()), m_numberVoxelsWidth(numberVoxelsWidth), m_numberVoxelsDepth(numberVoxelsDepth), m_numberVoxelsHeight(numberVoxelsHeight), m_imageHeight(imageHeight), m_imageWidth(imageWidth), m_spatialVoxelScale(scale)
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

__host__ void VoxelGrid::updateTSDF(Matrix4f extrinsics, Matrix3f intrinsics, float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight, float truncation)
{
	// Assume 240x240x240
	dim3 threadBlocks(20, 20);
	dim3 blocks(m_numberVoxelsWidth/20, m_numberVoxelsHeight/20);
	float *depthDataGPU;
	cudaMalloc(&depthDataGPU, sizeof(float) * depthMapWidth * depthMapHeight);
	cudaMemcpy(depthDataGPU, depthMap, sizeof(float) * depthMapWidth * depthMapHeight, cudaMemcpyHostToDevice);

	updateTSDFKernel<<<blocks, threadBlocks>>>(extrinsics, intrinsics, depthDataGPU, depthMapWidth, depthMapHeight, truncation, this->m_gridOriginOffset.x(), this->m_gridOriginOffset.y(), this->m_gridOriginOffset.z(), this->m_spatialVoxelScale, m_voxelGrid, m_numberVoxelsDepth, m_numberVoxelsHeight);
	cudaGetLastError();
	cudaDeviceSynchronize();
}

__host__ void VoxelGrid::raycastVoxelGrid(Matrix4f extrinsics)
{
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
