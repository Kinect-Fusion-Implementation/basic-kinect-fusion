#include "VoxelGrid.h"
#include <cmath>
#include <algorithm>
#include <iostream>

VoxelGrid::VoxelGrid(Vector3f gridOrigin, unsigned int numberVoxelsWidth, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight, float scale) : m_gridOriginOffset(gridOrigin.x(), gridOrigin.y(), gridOrigin.z()), m_numberVoxelsWidth(numberVoxelsWidth), m_numberVoxelsDepth(numberVoxelsDepth), m_numberVoxelsHeight(numberVoxelsHeight), m_spatialVoxelScale(scale)
{
	unsigned long long numberVoxels = m_numberVoxelsWidth * m_numberVoxelsDepth * m_numberVoxelsHeight;
	m_voxelGrid = std::vector<VoxelData>(numberVoxels, VoxelData(0.0, 0.0));
}

Vector3f VoxelGrid::getCellCenterInWorldCoords(Vector3i gridCell)
{
	Vector3f gridCellCoordinates(float(gridCell.x()), float(gridCell.y()), float(gridCell.z()));
	Vector3f centerOffset(0.5, 0.5, 0.5);
	gridCellCoordinates += centerOffset;
	gridCellCoordinates *= m_spatialVoxelScale;
	// grid origin is the offset from the world frame origin (point in voxel grid frame is offset by exactly that vector)
	return gridCellCoordinates + m_gridOriginOffset;
}

VoxelData &VoxelGrid::getVoxelData(unsigned int w, unsigned int h, unsigned int d)
{
	return m_voxelGrid.at(d + m_numberVoxelsDepth * h + m_numberVoxelsDepth * m_numberVoxelsHeight * w);
}

// Raycast traversal
	/*for (size_t w = 0; w < depthMapWidth; w++) {
		for (size_t h = 0; h < depthMapHeight; h++) {
			// Start at view plane (TODO: needs intrinsics)
			Vector3f trajectory = extrinsics.block<3, 3>(0, 0) * Vector3f(0, 0, 1);
			Vector3f rayPos = extrinsics.block<3, 1>(0, 3) + 5 * trajectory;
			while (true) {
				// Set up next step
				Vector3i gridCoordinates(int(rayPos.x()), int(rayPos.y()), int(rayPos.z()));
				if (gridCoordinates.x() < 0 || gridCoordinates.x() >= m_numberVoxelsWidth || gridCoordinates.y() < 0 || gridCoordinates.y() >= m_numberVoxelsHeight || gridCoordinates.z() < 0 || gridCoordinates.z() >= m_numberVoxelsDepth) {
					continue;
				}
				Vector3f worldCoordinatesOfGridCell = getCellCenterInWorldCoords(gridCoordinates);
				// Process voxel (TODO)

				// Get next position on ray
				Vector3f nextCrossings = rayPos - Vector3f(float(gridCoordinates.x()), float(gridCoordinates.y()), float(gridCoordinates.z()));
				nextCrossings.x() = trajectory.x() > 0 ? 1 - nextCrossings.x() : -nextCrossings.x();
				nextCrossings.x() = trajectory.x() == 0 ? INF : nextCrossings.x() /= trajectory.x();
				nextCrossings.y() = trajectory.y() > 0 ? 1 - nextCrossings.y() : -nextCrossings.y();
				nextCrossings.y() = trajectory.y() == 0 ? INF : nextCrossings.y() /= trajectory.y();
				nextCrossings.z() = trajectory.z() > 0 ? 1 - nextCrossings.z() : -nextCrossings.z();
				nextCrossings.z() = trajectory.z() == 0 ? INF : nextCrossings.z() /= trajectory.z();
				rayPos += (std::min(std::min(nextCrossings.x(), nextCrossings.y()), nextCrossings.z()) + 0.0625f) * trajectory;
			}
		}
	}*/

	// w,h are the (x,y) coordinates of the frontal slice
	// Finally iterate to the right...

void VoxelGrid::updateTSDF(Matrix4f extrinsics, Matrix3f intrinsics, float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight, float truncation)
{
	// w,h are the (x,y) coordinates of the frontal slice
	// Finaly iterate upwards
	for (size_t w = 0; w < m_numberVoxelsWidth; w++)
	{
		// Then iterate voxels to the right...
		for (size_t h = 0; h < m_numberVoxelsHeight; h++)
		{
			// First iterate in depth
			for (size_t d = 0; d < m_numberVoxelsDepth; d++)
			{
				// std::cout << "Processing voxel: (" << w << ", " << h << ", " << d << ") (width, height, depth)" << std::endl;
				Vector3i gridCoordinates(w, h, d);
				Vector3f worldCoordinatesOfGridCell = getCellCenterInWorldCoords(gridCoordinates);
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
					getVoxelData(w, h, d).freeSpace++;
				}

				// There is an alternative formulation to this in the second paper...
				// This will be >0 for points between camera and surface and < 0 for points behind the surface
				float sdfEstimate = std::clamp(depth - distanceOfVoxelToCamera, -truncation, truncation);
				if (std::abs(sdfEstimate) < truncation)
				{
					VoxelData &voxel = getVoxelData(w, h, d);
					// Just like in the paper
					float newWeight = 1;
					voxel.depthAverage = (voxel.depthAverage * voxel.weights + sdfEstimate * newWeight) / (voxel.weights + newWeight);
					voxel.weights += newWeight;
				}
			}
		}
	}
}