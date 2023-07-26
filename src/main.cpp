#include <iostream>
#include <fstream>
#include <chrono>

#include "./sensor/VirtualSensor.h"
#include "./kinect_fusion/Eigen.h"
#include "./kinect_fusion/ICPOptimizer.h"
#include "./kinect_fusion/PointCloud.h"
#include "./kinect_fusion/PointCloudPyramid.h"
#include "./kinect_fusion/ICPOptimizer.h"
#include "./configuration/Configuration.h"
#include "./kinect_fusion/ICPOptimizer.h"
#include "CudaVoxelGrid.h"
#include "../visualization/MarchingCubes.h"
#include "./visualization/PointCloudToMesh.h"

int main()
{
    int result = 0;
    // return icp_accuracy_test();

    VirtualSensor sensor;
    sensor.init(Configuration::getDataSetPath());

    float sigmaS(2.0);
    float sigmaR(2.0);
    std::cout << "Using sigmaS: " << sigmaS << std::endl;
    std::cout << "Using sigmaR: " << sigmaR << std::endl;

    // Number of subsampling levels -> Without the basic level, the pyramid will contain subLevels + 1 point clouds
    const unsigned subLevels = 2;
    // Size of smoothing window
    const unsigned windowSize = 7;
    // Size of subsampling window
    const unsigned blockSize = 3;

    int roomWidthMeter = 6;
    int roomHeightMeter = 6;
    int roomDepthMeter = 6;
    float voxelsPerMeter = 100;
    float scale = 1 / voxelsPerMeter;
    float truncation = 0.125f;
    int numberVoxelsWidth = roomWidthMeter * voxelsPerMeter;
    int numberVoxelsHeight = roomHeightMeter * voxelsPerMeter;
    int numberVoxelsDepth = roomDepthMeter * voxelsPerMeter;

#if EVAL_MODE
    auto gridGenStart = std::chrono::high_resolution_clock::now();
#endif
    // x,y,z: widht, height, depth
    VoxelGrid grid(Vector3f(-2.0, -1.0, -2.0), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), scale, truncation);
#if EVAL_MODE
    auto gridGenEnd = std::chrono::high_resolution_clock::now();
    std::cout << "Setting up grid took: " << std::chrono::duration_cast<std::chrono::milliseconds>(gridGenEnd - gridGenStart).count() << " ms" << std::endl;
#endif
    int idx = 0;
    Matrix4f trajectoryOffset;
    auto totalComputeStart = std::chrono::high_resolution_clock::now();
    while (sensor.processNextFrame())
    {
        auto frameComputeStart = std::chrono::high_resolution_clock::now();
        float *depth = sensor.getDepth();
        // Trajectory:       world -> view space (Extrinsics)
        // InvTrajectory:    view -> world space (Pose)

        if (idx == 0)
        {
            // We express our world space based on the first trajectory (we set the first trajectory to eye matrix, and express all further camera positions relative to that first camera position)
            trajectoryOffset = sensor.getTrajectory().inverse();
        }
        idx++;

#if EVAL_MODE
        auto updateTSDFStart = std::chrono::high_resolution_clock::now();
#endif
        // Raycast
        // ICP
        grid.updateTSDF(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight());

        // Just for testing
        // pcloud(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getTrajectory() * trajectoryOffset, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 0);
        //writeMesh(pcloud.getPointsCPU(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "mesh_" + std::to_string(idx) + ".off");
        //return 0;
#if EVAL_MODE
        auto updateTSDFEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Computing the TSDF update (volumetric fusion) took: " << std::chrono::duration_cast<std::chrono::milliseconds>(updateTSDFEnd - updateTSDFStart).count() << " ms" << std::endl;
        auto pyramidComputeStart = std::chrono::high_resolution_clock::now();
#endif
        PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getTrajectory() * trajectoryOffset, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 0, windowSize, blockSize, sigmaR, sigmaS);
#if EVAL_MODE
        auto pyramidComputeEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Computing the pyramid took: " << std::chrono::duration_cast<std::chrono::milliseconds>(pyramidComputeEnd - pyramidComputeStart).count() << " ms" << std::endl;
        for (size_t i = 0; i < pyramid.getPointClouds().size(); i++)
        {
            std::cout << "Generating mesh for level " << i << std::endl;
            writeMesh(pyramid.getPointClouds().at(i).getPointsCPU(), sensor.getDepthImageWidth() >> i, sensor.getDepthImageHeight() >> i, Configuration::getOutputDirectory() + std::string("mesh_") + std::to_string(i));
        }
        return 0;
        auto raycastStart = std::chrono::high_resolution_clock::now();
#endif
        RaycastImage raycast = grid.raycastVoxelGrid(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics());
#if SAVE_IMAGE_MODE
        if (idx % 50 == 0 || idx > 70 && idx < 100)
        {
            ImageUtil::saveNormalMapToImage((float *)raycast.m_normalMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("RaycastedImage_") + std::to_string(idx), "");
            writeMesh(raycast.m_vertexMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "mesh_" + std::to_string(idx) + ".off");
        }
#endif
#if EVAL_MODE
        auto raycastStop = std::chrono::high_resolution_clock::now();
        auto frameComputeEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Computing raycasting took: " << std::chrono::duration_cast<std::chrono::milliseconds>(raycastStop - raycastStart).count() << " ms" << std::endl;
        std::cout << "Computing the frame took: " << std::chrono::duration_cast<std::chrono::milliseconds>(frameComputeEnd - frameComputeStart).count() << " ms" << std::endl;
#endif
    }

    auto totalComputeStop = std::chrono::high_resolution_clock::now();
    std::cout << "Computing for all frames took: " << std::chrono::duration_cast<std::chrono::milliseconds>(totalComputeStop - totalComputeStart).count() << " ms" << std::endl;
#if SAVE_IMAGE_MODE
    auto marchingCubesStart = std::chrono::high_resolution_clock::now();
    run_marching_cubes(grid, idx);
    auto marchingCubesStop = std::chrono::high_resolution_clock::now();
    std::cout << "Computing marching cubes took: " << std::chrono::duration_cast<std::chrono::milliseconds>(marchingCubesStop - marchingCubesStart).count() << " ms" << std::endl;
#endif
    return result;
}
