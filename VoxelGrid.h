#pragma once
#include "Eigen.h"

/**
 * Implicit Representation of our scene as a discretized TSDF
 *
 */
class VoxelGrid
{
private:
    // Stores current TSDF estimate and current sum of weights per voxel
    std::vector<Eigen::Vector2f> m_voxelGrid;
    // Grid is orientied along the world frame axes, but we want to define the area it covers freely by shifting its (0,0) location relative to the world frame
    Vector3f m_gridOrigin;

    /// Defines how many voxels along each direction the grid will have
    unsigned int m_numberVoxelsWidth;
    unsigned int m_numberVoxelsDepth;
    unsigned int m_numberVoxelsHeight;

    // Defines the spatial extend each voxel will represent along each direction (side length of cube)
    float spatialVoxelScale;

    VoxelGrid() = delete;

public:
    VoxelGrid(Vector3f gridOrigin, unsigned int numberVoxelsWidth, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight, float scale);

    /**
     * Transforms coordinates in the voxel grids (grid indices along each direction (w, d, h)) into a corresponding point in world coordinates.
     * Note that this point corresponds to the center of the voxel grid cell corresponding to the index.
     */
    Vector3f voxelGridCenterToWorld(Vector3i gridCell)
    {
        Vector3f gridCellCoordinates(gridCell.x(), gridCell.x(), gridCell.x());
        Vector3f centerOffset(0.5f, 0.5f, 0.5f);
        gridCellCoordinates += centerOffset;
        return (gridCellCoordinates)*spatialVoxelScale;
    }
};
