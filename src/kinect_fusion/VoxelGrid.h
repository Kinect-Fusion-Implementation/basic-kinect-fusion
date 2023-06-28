#pragma once
#include "Eigen.h"

struct VoxelData {
    double depthAverage = 0;
    double weights = 0;
    int freeSpace = 0;

    VoxelData() = delete;
    VoxelData(double depthAverage, double weights): depthAverage(depthAverage), weights(weights), freeSpace(0){};
};

/**
 * Implicit Representation of our scene as a discretized TSDF
 *
 */
class VoxelGrid
{
private:
    // Stores current TSDF estimate and current sum of weights per voxel
    // A voxel is identified by 3 coordinates (i,j,k) (along width, along height, along depth)
    // Stores them in (depth -> height -> width) (stores all voxels of one frontal slice pixel consecutive)
    // Thus coordinate (i, j, k) corresponds to linearized index k + max_depth * j + max_depth * max_height * i
    std::vector<VoxelData> m_voxelGrid;
    // Grid is orientied along the world frame axes, but we want to define the area it covers freely by shifting its (0,0) location relative to the world frame
    Vector3d m_gridOrigin;

    /// Defines how many voxels along each direction the grid will have

    // Defines the spatial extend each voxel will represent along each direction (side length of cube)
    float m_spatialVoxelScale;

    VoxelGrid() = delete;

public:
    unsigned int m_numberVoxelsWidth;
    unsigned int m_numberVoxelsDepth;
    unsigned int m_numberVoxelsHeight;

    
    VoxelGrid(Vector3d gridOrigin, unsigned int numberVoxelsWidth, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight, double scale);

    /**
     * Transforms coordinates in the voxel grids (grid indices along each direction (width, height, depth)) into a corresponding point in world coordinates.
     * Note that this point corresponds to the center of the voxel grid cell corresponding to the index.
     */
    Vector3d voxelGridCenterToWorld(Vector3i gridCell);

    /**
     * Updates TSDF Voxel grid using Volumetric Fusion algorithm
    */
   void updateTSDF(Matrix4d extrinsics, Matrix3d intrinsics, double* depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight, double truncation);


    /**
     * Provides the storage index of grid coordinates ((w)idth, (h)eight, (d)epth)
     * We store our sdf data in depth > height > width
    */
   VoxelData& getVoxelData(unsigned int w, unsigned int h, unsigned int d);
};
