#pragma once
#include "../kinect_fusion/Eigen.h"
#include <iostream>

struct VoxelData
{
    float depthAverage = 0;
    float weights = 0;
    int freeSpace = 0;

    VoxelData() {}
    VoxelData(float depthAverage, float weights) : depthAverage(depthAverage), weights(weights), freeSpace(0) {}
};

struct RaycastImage
{
    Vector3f *m_vertexMap;
    Vector3f *m_normalMap;
    Vector3f *m_vertexMapGPU;
	Vector3f *m_normalMapGPU;

    RaycastImage() : m_vertexMap(nullptr), m_normalMap(nullptr), m_vertexMapGPU(nullptr), m_normalMapGPU(nullptr){};
    RaycastImage(int width, int height): m_vertexMapGPU(nullptr), m_normalMapGPU(nullptr)
    {
        m_vertexMap = new Vector3f[width * height];
        m_normalMap = new Vector3f[width * height];
        for (int i = 0; i < width * height; i++)
        {
            m_vertexMap[i] = Vector3f(MINF, MINF, MINF);
            m_normalMap[i] = Vector3f(MINF, MINF, MINF);
        }
    };

    ~RaycastImage();
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
    // In our case this is stored in the CUDA GPU Memory
    VoxelData *m_voxelGrid;
    VoxelData *m_voxelGridCPU;

    // Grid is orientied along the world frame axes, but we want to define the area it covers freely by shifting its (0,0) location relative to the world frame

    // Defines the spatial extend each voxel will represent along each direction (side length of cube)

public:
    unsigned int m_numberVoxelsWidth;
    unsigned int m_numberVoxelsDepth;
    unsigned int m_numberVoxelsHeight;
    unsigned int m_imageHeight;
    unsigned int m_imageWidth;
    float m_spatialVoxelScale;
    float m_truncation;
    Vector3f m_gridOriginOffset;
    size_t m_memorySize;

    /**
     * Allocates memory on the GPU for the voxel grid and I/O from raycasting
     */
    VoxelGrid(Vector3f gridOrigin, unsigned int numberVoxelsWidth, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight, unsigned int imageHeight, unsigned int imageWidth, float scale, float truncation);

    VoxelGrid(){}

    ~VoxelGrid();

    void sync();

    /**
     * Calls the corresponding kernel to update TSDF.
     * Updates TSDF Voxel grid using Volumetric Fusion algorithm
     */
    void updateTSDF(Matrix4f extrinsics, Matrix3f intrinsics, float *depthMap, unsigned int depthMapWidth, unsigned int depthMapHeight);

    /**
     * Provides the point cloud that is the result of raycasting the voxel grid
     */
    RaycastImage raycastVoxelGrid(Matrix4f extrinsics, Matrix3f intrinsics);

    /**
     * Transforms coordinates in the voxel grids (grid indices along each direction (width, height, depth)) into a corresponding point in world coordinates.
     * Note that this point corresponds to the center of the voxel grid cell corresponding to the index.
     */
    Vector3f getCellCenterInWorldCoords(Vector3i gridCell);

    /**
     * Transforms coordinates in world coordinates into corresponding grid coordinates.
     */
    Vector3i getGridCoordinates(Vector3f worldCoordinates);

    /**
     * Provides the storage index of grid coordinates ((w)idth, (h)eight, (d)epth)
     * We store our sdf data in depth > height > width
     */
    VoxelData &getVoxelData(unsigned int w, unsigned int h, unsigned int d);
};

VoxelData &getVoxelData(VoxelData *voxelGrid, int w, unsigned int h, unsigned int d, unsigned int numberVoxelsDepth, unsigned int numberVoxelsHeight);

Vector3f getWorldCoordinates(int x, int y, int z, float offset_x, float offset_y, float offset_z, float spatialVoxelScale);

Vector3i getGridCoordinates(Vector3f worldCoordinates, Vector3f gridOriginOffset, float spatialVoxelScle);