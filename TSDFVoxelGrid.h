#pragma once
#include "Eigen.h"

class VoxelGrid
{
private:
    // Stores current TSDF estimate and current sum of weights per voxel
    std::vector<Eigen::Vector2f> m_voxelGrid;
    // Defines the spatial extend of the map (width and depth of the room)
    unsigned int m_spatialExtendHorizontally;
    // Defines the spatial extend of the map (height of the room)
    unsigned int m_spatialExtendVertically;
    // Defines how many voxels in per unit of space
    unsigned int m_voxelsPerUnit;

    VoxelGrid() = delete;
public:
    VoxelGrid(unsigned int spatialExtendHorizontally, unsigned int spatialExtendVertically,  unsigned int voxelsPerUnit);
};
