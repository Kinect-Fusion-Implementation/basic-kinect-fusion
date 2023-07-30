#include <iostream>
#include <fstream>
#include <chrono>

#include "./sensor/VirtualSensor.h"
#include "./kinect_fusion/Eigen.h"
#include "PointCloud.h"
#include "PointCloudPyramid.h"
#include "./configuration/Configuration.h"
#include "ICPOptimizer.h"
#include "CudaVoxelGrid.h"
#include "../visualization/MarchingCubes.h"
#include "./visualization/PointCloudToMesh.h"

int icp_accuracy_test()
{
        int result = 0;

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
        // x,y,z: width, height, depth
        VoxelGrid grid(Vector3f(-2.0, -1.0, -2.0), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), scale, truncation);

        float vertex_diff_threshold = 0.3;
        float normal_diff_threshold = 0.3;
        std::vector<int> iterations_per_level = {4, 4, 4};
        ICPOptimizer optimizer(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), vertex_diff_threshold, normal_diff_threshold, iterations_per_level);
        
        Matrix4f prevFrameToGlobal = Matrix4f::Identity();
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
                        // Update the TSDF w.r.t. the first camera frame C0 (all other poses/extrinsics are expressions relative to C0)
                        grid.updateTSDF(Matrix4f::Identity(), sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
                        idx++;
                        continue;
                }
                idx++;

                PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 2, windowSize, blockSize, sigmaR, sigmaS);

                if (idx % 100 == 0)
                {
                        // run_marching_cubes(grid, idx);
                        for (size_t i = 0; i < pyramid.getPointClouds().size(); i++)
                        {
                                std::cout << "Generating mesh for level " << i << std::endl;
                                ImageUtil::saveNormalMapToImage((float *)pyramid.getPointClouds().at(i).getNormalsCPU(), sensor.getDepthImageWidth() >> i, sensor.getDepthImageHeight() >> i, std::string("PyramidImage_") + std::to_string(idx) + "_level" + std::to_string(i), "");
                                writeMesh(pyramid.getPointClouds().at(i).getPointsCPU(), sensor.getDepthImageWidth() >> i, sensor.getDepthImageHeight() >> i, Configuration::getOutputDirectory() + std::string("PyramidMesh_") + std::to_string(idx) + "_level" + std::to_string(i) + ".off");
                        }
                }

                RaycastImage raycast = grid.raycastVoxelGrid(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics());
                if (idx % 100 == 0)
                {
                        ImageUtil::saveNormalMapToImage((float *)raycast.m_normalMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("RaycastedImage_") + std::to_string(idx), "");
                        writeMesh(raycast.m_vertexMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "RaycastedMesh_" + std::to_string(idx) + ".off");
                }
                Matrix4f gt_extrinsics = sensor.getTrajectory() * trajectoryOffset;
                Matrix4f gt_pose = gt_extrinsics.inverse();
                // Estimate the pose of the current frame
                Matrix4f estPose;
                estPose = optimizer.optimize(pyramid, raycast.m_vertexMapGPU, raycast.m_normalMapGPU, prevFrameToGlobal);
                std::cout << "Ground Truth: " << std::endl
                          << gt_pose << std::endl;
                std::cout << "Determinant: " << gt_pose.determinant() << std::endl;
                std::cout << "Estimated: " << std::endl
                          << estPose << std::endl;
                std::cout << "Determinant: " << estPose.determinant() << std::endl;
                // Use estimated pose as prevPose for next frame
                grid.updateTSDF(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
                prevFrameToGlobal = estPose;
        }
        run_marching_cubes(grid, idx);
                
        return 0;
}

int main()
{
        bool icp = true;
        if (icp)
        {
                return icp_accuracy_test();
        }
        else
        {
                int result = 0;

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
                float vertex_diff_threshold = 0.3;
                float normal_diff_threshold = 0.3;
                std::vector<int> iterations_per_level = {3, 3, 5};
                ICPOptimizer optimizer(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), vertex_diff_threshold, normal_diff_threshold, iterations_per_level);
                Matrix4f prevFrameToGlobal = Matrix4f::Identity();

#if EVAL_MODE
                auto gridGenStart = std::chrono::high_resolution_clock::now();
#endif
                // x,y,z: width, height, depth
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
#if EVAL_MODE
                        auto updateTSDFEnd = std::chrono::high_resolution_clock::now();
                        std::cout << "Computing the TSDF update (volumetric fusion) took: " << std::chrono::duration_cast<std::chrono::milliseconds>(updateTSDFEnd - updateTSDFStart).count() << " ms" << std::endl;
                        auto pyramidComputeStart = std::chrono::high_resolution_clock::now();
#endif
                        PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 2, windowSize, blockSize, sigmaR, sigmaS);
                        /* Use this to compare the normals and scenes of pyramid and raycast
                        ImageUtil::saveNormalMapToImage((float *) pyramid.getPointClouds().at(0).getNormalsCPU(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("PyramidImage_") + std::to_string(0), "");
                        writeMesh(pyramid.getPointClouds().at(0).getPointsCPU(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + std::string("mesh_") + std::to_string(0) + ".off");
                        ImageUtil::saveNormalMapToImage((float *) pyramid.getPointClouds().at(1).getNormalsCPU(), sensor.getDepthImageWidth() >> 1, sensor.getDepthImageHeight() >> 1, std::string("PyramidImage_") + std::to_string(1), "");
                        writeMesh(pyramid.getPointClouds().at(1).getPointsCPU(), sensor.getDepthImageWidth() >> 1, sensor.getDepthImageHeight() >> 1, Configuration::getOutputDirectory() + std::string("mesh_") + std::to_string(1) + ".off");
                        ImageUtil::saveNormalMapToImage((float *) pyramid.getPointClouds().at(2).getNormalsCPU(), sensor.getDepthImageWidth()>> 2, sensor.getDepthImageHeight()>> 2, std::string("PyramidImage_") + std::to_string(2), "");
                        writeMesh(pyramid.getPointClouds().at(2).getPointsCPU(), sensor.getDepthImageWidth() >> 2, sensor.getDepthImageHeight() >> 2, Configuration::getOutputDirectory() + std::string("mesh_") + std::to_string(2) + ".off");
                        */
#if EVAL_MODE
                        auto pyramidComputeEnd = std::chrono::high_resolution_clock::now();
                        std::cout << "Computing the pyramid took: " << std::chrono::duration_cast<std::chrono::milliseconds>(pyramidComputeEnd - pyramidComputeStart).count() << " ms" << std::endl;
#endif
#if SAVE_IMAGE_MODE
                        if (idx % 100 == 0)
                        {
                                // run_marching_cubes(grid, idx);
                                for (size_t i = 0; i < pyramid.getPointClouds().size(); i++)
                                {
                                        std::cout << "Generating mesh for level " << i << std::endl;
                                        ImageUtil::saveNormalMapToImage((float *)pyramid.getPointClouds().at(i).getNormalsCPU(), sensor.getDepthImageWidth() >> i, sensor.getDepthImageHeight() >> i, std::string("PyramidImage_") + std::to_string(idx) + "_level" + std::to_string(i), "");
                                        writeMesh(pyramid.getPointClouds().at(i).getPointsCPU(), sensor.getDepthImageWidth() >> i, sensor.getDepthImageHeight() >> i, Configuration::getOutputDirectory() + std::string("PyramidMesh_") + std::to_string(idx) + "_level" + std::to_string(i) + ".off");
                                }
                        }
#endif
#if EVAL_MODE
                        auto raycastStart = std::chrono::high_resolution_clock::now();
#endif
                        RaycastImage raycast = grid.raycastVoxelGrid(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics());
                        /* Use this to compare the normals and scenes of pyramid and raycast
                        ImageUtil::saveNormalMapToImage((float *)raycast.m_normalMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("RaycastedImage_") + std::to_string(idx), "");
                        writeMesh(raycast.m_vertexMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "mesh_" + std::to_string(idx) + ".off");
                        */
#if SAVE_IMAGE_MODE
                        if (idx % 100 == 0)
                        {
                                ImageUtil::saveNormalMapToImage((float *)raycast.m_normalMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("RaycastedImage_") + std::to_string(idx), "");
                                writeMesh(raycast.m_vertexMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "RaycastedMesh_" + std::to_string(idx) + ".off");
                        }
#endif
#if EVAL_MODE
                        auto raycastStop = std::chrono::high_resolution_clock::now();
                        std::cout << "Computing raycasting took: " << std::chrono::duration_cast<std::chrono::milliseconds>(raycastStop - raycastStart).count() << " ms" << std::endl;
                        auto icpStart = std::chrono::high_resolution_clock::now();
#endif
                        prevFrameToGlobal = optimizer.optimize(pyramid, raycast.m_vertexMapGPU, raycast.m_normalMapGPU, prevFrameToGlobal);
#if EVAL_MODE
                        auto icpEnd = std::chrono::high_resolution_clock::now();
                        auto frameComputeEnd = std::chrono::high_resolution_clock::now();
                        std::cout << "Computing ICP took: " << std::chrono::duration_cast<std::chrono::milliseconds>(icpEnd - icpStart).count() << " ms" << std::endl;
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
}
