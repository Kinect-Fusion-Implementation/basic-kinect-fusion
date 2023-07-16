#include "CudaVoxelGrid.h"
#include <stdio.h>
#include <iostream>

__device__ __host__ Vector3f VoxelGrid::getCellCenterInWorldCoords(Vector3i gridCell)
{
	Vector3f gridCellCoordinates(float(gridCell.x()), float(gridCell.y()), float(gridCell.z()));
	Vector3f centerOffset(0.5, 0.5, 0.5);
	gridCellCoordinates += centerOffset;
	gridCellCoordinates *= m_spatialVoxelScale;
	// grid origin is the offset from the world frame origin (point in voxel grid frame is offset by exactly that vector)
	return gridCellCoordinates + m_gridOriginOffset;
}

__host__ VoxelGrid::VoxelGrid(Vector3f gridOrigin, unsigned int numberVoxelsWidth, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight, unsigned int imageHeight, unsigned int imageWidth, float scale) : m_gridOriginOffset(gridOrigin.x(), gridOrigin.y(), gridOrigin.z()), m_numberVoxelsWidth(numberVoxelsWidth), m_numberVoxelsDepth(numberVoxelsDepth), m_numberVoxelsHeight(numberVoxelsHeight), m_imageHeight(imageHeight), m_imageWidth(imageWidth), m_spatialVoxelScale(scale)
{
	unsigned long long numberVoxels = m_numberVoxelsWidth * m_numberVoxelsDepth * m_numberVoxelsHeight;
	cudaMalloc(&m_voxelGrid, sizeof(VoxelData) * numberVoxels);
	VoxelData *cleanGrid = new VoxelData[numberVoxels];
	cudaMemcpy(m_voxelGrid, cleanGrid, sizeof(VoxelData) * numberVoxels, cudaMemcpyHostToDevice);
}

__host__ VoxelGrid::~VoxelGrid()
{
	cudaFree(m_voxelGrid);
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

__device__ __host__ VoxelData &getVoxelData(VoxelData* voxelGrid, int w, unsigned int h, unsigned int d, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight)
{
	return voxelGrid[d + numberVoxelsDepth * h + numberVoxelsDepth * numberVoxelsHeight * w];
}

__global__ void updateTSDFKernel(Matrix4f extrinsics, Matrix3f intrinsics, float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight, float truncation, float offset_x, float offset_y, float offset_z, float spatialVoxelScale, VoxelData* tsdf, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight)
{
	unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned h = blockIdx.y * blockDim.y + threadIdx.y;

	for (size_t d = 0; d < numberVoxelsDepth; d++)
	{
		// std::cout << "Processing voxel: (" << w << ", " << h << ", " << d << ") (width, height, depth)" << std::endl;
		Vector3f worldCoordinatesOfGridCell = getWorldCoordinates(w,h,d, offset_x, offset_y, offset_z, spatialVoxelScale);

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

		float distanceOfVoxelToCamera = (worldCoordinatesOfGridCell - extrinsics.inverse().block<3, 1>(0, 3)).norm();
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
};

__host__ void VoxelGrid::updateTSDF(Matrix4f extrinsics, Matrix3f intrinsics, float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight, float truncation)
{
	// Assume 480 x 640
	dim3 threadBlocks(16, 16);
	dim3 blocks(30, 40);
	updateTSDFKernel<<<blocks, threadBlocks>>>(extrinsics, intrinsics, depthMap, depthMapWidth, depthMapHeight, truncation, this->m_gridOriginOffset.x(), this->m_gridOriginOffset.y(), this->m_gridOriginOffset.z(), this->m_spatialVoxelScale, m_voxelGrid, m_numberVoxelsDepth, m_numberVoxelsDepth);
}

__host__ void VoxelGrid::raycastVoxelGrid(Matrix4f extrinsics)
{
}

__device__ __host__ VoxelData &VoxelGrid::getVoxelData(unsigned int w, unsigned int h, unsigned int d)
{
	return m_voxelGrid[d + m_numberVoxelsDepth * h + m_numberVoxelsDepth * m_numberVoxelsHeight * w];
}
