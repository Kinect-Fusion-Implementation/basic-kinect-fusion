#include "VoxelGrid.h"
#include <cmath>
#include <algorithm>

VoxelGrid::VoxelGrid(Vector3f gridOrigin, unsigned int numberVoxelsWidth, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight, float scale) : m_gridOrigin(gridOrigin.x(), gridOrigin.y(), gridOrigin.z()), m_numberVoxelsWidth(numberVoxelsWidth), m_numberVoxelsDepth(numberVoxelsDepth), m_numberVoxelsHeight(numberVoxelsHeight), m_spatialVoxelScale(scale)
{
	unsigned long long numberVoxels = m_numberVoxelsWidth * m_numberVoxelsDepth * m_numberVoxelsHeight;
	m_voxelGrid = std::vector<VoxelData>(numberVoxels, VoxelData(0.0f, 0.0f));
}

    Vector3f VoxelGrid::voxelGridCenterToWorld(Vector3i gridCell)
    {
        Vector3f gridCellCoordinates(gridCell.x(), gridCell.y(), gridCell.z());
        Vector3f centerOffset(0.5f, 0.5f, 0.5f);
        gridCellCoordinates += centerOffset;
		gridCellCoordinates *= m_spatialVoxelScale;
		// grid origin is the offset from the world frame origin (point in voxel grid frame is offset by exactly that vector)
        return gridCellCoordinates + m_gridOrigin;
    }
	
	VoxelData& VoxelGrid::getVoxelData(unsigned int i, unsigned int j, unsigned int k)
	{
		return m_voxelGrid.at(k + m_numberVoxelsDepth * j + m_numberVoxelsDepth * m_numberVoxelsHeight * i);
	} 

	void VoxelGrid::updateTSDF(Matrix4f extrinsics, Matrix3f intrinsics, float* depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight, float truncation) {
		// i,j are the (x,y) coordinates of the frontal slice
		// Finaly iterate upwards
		for (size_t i = 0; i < m_numberVoxelsWidth; i++)
		{
			// Then iterate voxels to the right...
			for (size_t j = 0; j < m_numberVoxelsHeight; i++)
			{
				// First iterate in depth
				for (size_t k = 0; k < m_numberVoxelsDepth; i++)
				{
					Vector3i gridCoordinates(i, j, k);
					Vector3f worldCoordinatesOfGridCell = voxelGridCenterToWorld(gridCoordinates);
					Vector3f cameraCoordinatesOfGridCell = extrinsics.block(0,0,3,3) * worldCoordinatesOfGridCell + extrinsics.block(3,0,1,3);
					// voxel is behind the camera
					if (cameraCoordinatesOfGridCell.z() < 0) {
						break;
					}
					Vector3f pixelCoordinates = (intrinsics * cameraCoordinatesOfGridCell)/cameraCoordinatesOfGridCell.z();
					float depthOfVoxelInCamera = cameraCoordinatesOfGridCell.z();
					assert(pixelCoordinates.z() == 1);
					if (pixelCoordinates.x() < 0 || pixelCoordinates.y() < 0 || pixelCoordinates.x() >= depthMapWidth || pixelCoordinates.y() >= depthMapHeight) {
						break;
					}
					// find depth in depthmap that is stored in row major
					float depth = depthMap[static_cast<int>(pixelCoordinates.x()) + static_cast<int>(pixelCoordinates.y()) * depthMapWidth];
					
					if (depth > depthOfVoxelInCamera) {
						// Our voxel is in front of the surface from the view of the camera
						getVoxelData(i,j,k).freeSpace++;
					}
					// There is an alternative formulation to this in the second paper...
					float sdfEstimate = std::clamp(depth - depthOfVoxelInCamera, -truncation, truncation);
					if(sdfEstimate > -truncation) {
						VoxelData& voxel = getVoxelData(i,j,k);
						// Just like in the paper
						float newWeight = 1;
						voxel.depthAverage = (voxel.depthAverage * voxel.weights + sdfEstimate * newWeight) / (voxel.weights + newWeight); 
						voxel.weights += newWeight;
					}
				}
				
			}
		}
		
	}