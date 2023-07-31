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

        float vertex_diff_threshold = 0.05;
        // Approx 35 degrees
        float normal_diff_threshold = 0.15;
        std::vector<int> iterations_per_level = {10, 5, 3};
        ICPOptimizer optimizer(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), vertex_diff_threshold, normal_diff_threshold, iterations_per_level, 0.2f);

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

                PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 2, windowSize, blockSize, sigmaR, sigmaS);
                if (idx < 20 || idx % 100 == 0)
                {
                        ImageUtil::saveNormalMapToImage((float *)pyramid.getPointClouds().at(0).getNormalsCPU(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("PointCloudImage_") + std::to_string(idx), "");
                        // writeMesh(pyramid.getPointClouds().at(0).getPointsCPU(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "PointCloudMesh_" + std::to_string(idx) + ".off");
                }
                // RaycastImage raycast = grid.raycastVoxelGrid(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics());
                RaycastImage raycast = grid.raycastVoxelGrid(prevFrameToGlobal.inverse(), sensor.getDepthIntrinsics());
                if (idx < 20  || idx % 100 == 0)
                {
                        ImageUtil::saveNormalMapToImage((float *)raycast.m_normalMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("RaycastedImage_") + std::to_string(idx), "");
                        // writeMesh(raycast.m_vertexMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "RaycastedMesh_" + std::to_string(idx) + ".off");
                }
                // Estimate the pose of the current frame
                Matrix4f estPose;
                Matrix4f gt_extrinsics = sensor.getTrajectory() * trajectoryOffset;
                Matrix4f gt_pose = gt_extrinsics.inverse();

                estPose = optimizer.optimize(pyramid, raycast.m_vertexMapGPU, raycast.m_normalMapGPU, prevFrameToGlobal, idx);
                std::cout << "Ground Truth: " << std::endl
                          << gt_pose << std::endl;
                std::cout << "Estimated: " << std::endl
                          << estPose << std::endl;
                std::cout << "Determinant estimate: " << estPose.determinant() << std::endl;
                std::cout << "Norm: " << (estPose - gt_pose).norm() << std::endl;
                
                // Use estimated pose as prevPose for next frame
                prevFrameToGlobal = gt_pose;
                grid.updateTSDF(gt_extrinsics, sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
                //grid.updateTSDF(estPose.inverse(), sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
                idx++;
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
                if (!sensor.init(Configuration::getDataSetPath()))
                {
                        std::cerr << "Failed to initialize sensor data!" << std::endl;
                        return 0;
                }
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

                int roomWidthMeter = 8;
                int roomHeightMeter = 8;
                int roomDepthMeter = 8;
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
                VoxelGrid grid(Vector3f(-4.0, -4.0, -4.0), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), scale, truncation);
#if EVAL_MODE
                auto gridGenEnd = std::chrono::high_resolution_clock::now();
                std::cout << "Setting up grid took: " << std::chrono::duration_cast<std::chrono::milliseconds>(gridGenEnd - gridGenStart).count() << " ms" << std::endl;
#endif
                int idx = 0;
                Matrix4f trajectoryOffset;
                auto totalComputeStart = std::chrono::high_resolution_clock::now();
                std::vector<std::chrono::high_resolution_clock::duration> measurementsTsdfUpdate;
                std::vector<std::chrono::high_resolution_clock::duration> measurementsPyramidCompute;
                std::vector<std::chrono::high_resolution_clock::duration> measurementsRayCast;
                std::vector<std::chrono::high_resolution_clock::duration> measurementsICP;
                std::vector<std::chrono::high_resolution_clock::duration> measurementsFrame;
                measurementsICP.reserve(800);
                measurementsPyramidCompute.reserve(800);
                measurementsRayCast.reserve(800);
                measurementsTsdfUpdate.reserve(800);
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
                        measurementsTsdfUpdate.push_back(updateTSDFEnd - updateTSDFStart);
                        auto pyramidComputeStart = std::chrono::high_resolution_clock::now();
#endif
                        PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 2, windowSize, blockSize, sigmaR, sigmaS);
                        if (idx % 100 == 0)
                        {
                                ImageUtil::saveNormalMapToImage((float *)pyramid.getPointClouds().at(0).getNormalsCPU(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("PyramidImage_") + std::to_string(idx) + "_level" + std::to_string(0), "");
                        }
#if EVAL_MODE
                        auto pyramidComputeEnd = std::chrono::high_resolution_clock::now();
                        measurementsPyramidCompute.push_back(pyramidComputeEnd - pyramidComputeStart);
                        // std::cout << "Computing the pyramid took: " << std::chrono::duration_cast<std::chrono::milliseconds>(pyramidComputeEnd - pyramidComputeStart).count() << " ms" << std::endl;
#endif
#if SAVE_IMAGE_MODE
                        if (idx % 100 == 0)
                        {
                                // run_marching_cubes(grid, idx);
                                for (size_t i = 0; i < pyramid.getPointClouds().size(); i++)
                                {
                                        std::cout << "Generating mesh for level " << i << std::endl;
                                        ImageUtil::saveNormalMapToImage((float *)pyramid.getPointClouds().at(i).getNormalsCPU(), sensor.getDepthImageWidth() >> i, sensor.getDepthImageHeight() >> i, std::string("PyramidImage_") + std::to_string(idx) + "_level" + std::to_string(i), "");
                                        // writeMesh(pyramid.getPointClouds().at(i).getPointsCPU(), sensor.getDepthImageWidth() >> i, sensor.getDepthImageHeight() >> i, Configuration::getOutputDirectory() + std::string("PyramidMesh_") + std::to_string(idx) + "_level" + std::to_string(i) + ".off");
                                }
                        }
#endif
#if EVAL_MODE
                        auto raycastStart = std::chrono::high_resolution_clock::now();
#endif
                        RaycastImage raycast = grid.raycastVoxelGrid(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics());
                        ImageUtil::saveNormalMapToImage((float *)raycast.m_normalMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("RaycastedImage_") + std::to_string(idx), "");
#if SAVE_IMAGE_MODE
                        // writeMesh(raycast.m_vertexMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "RaycastedMesh_" + std::to_string(idx) + ".off");

#endif
#if EVAL_MODE
                        auto raycastStop = std::chrono::high_resolution_clock::now();
                        measurementsRayCast.push_back(raycastStop - raycastStart);
                        // std::cout << "Computing raycasting took: " << std::chrono::duration_cast<std::chrono::milliseconds>(raycastStop - raycastStart).count() << " ms" << std::endl;
                        auto icpStart = std::chrono::high_resolution_clock::now();
#endif
                        prevFrameToGlobal = optimizer.optimize(pyramid, raycast.m_vertexMapGPU, raycast.m_normalMapGPU, prevFrameToGlobal, idx);
#if EVAL_MODE
                        auto icpEnd = std::chrono::high_resolution_clock::now();
                        measurementsICP.push_back(icpEnd - icpStart);
                        auto frameComputeEnd = std::chrono::high_resolution_clock::now();
                        measurementsFrame.push_back(frameComputeEnd - frameComputeStart);
#endif
                }
                auto totalComputeStop = std::chrono::high_resolution_clock::now();
                std::cout << "Computing for all frames took: " << std::chrono::duration_cast<std::chrono::milliseconds>(totalComputeStop - totalComputeStart).count() << " ms" << std::endl;
#if EVAL_MODE
                double meanTSDF = 0;
                double meanPyramid = 0;
                double meanRaycast = 0;
                double meanICP = 0;
                double meanFrame = 0;
                for (size_t i = 0; i < measurementsTsdfUpdate.size(); i++)
                {
                        meanTSDF += std::chrono::duration_cast<std::chrono::milliseconds>(measurementsTsdfUpdate.at(i)).count();
                        meanPyramid += std::chrono::duration_cast<std::chrono::milliseconds>(measurementsPyramidCompute.at(i)).count();
                        meanRaycast += std::chrono::duration_cast<std::chrono::milliseconds>(measurementsRayCast.at(i)).count();
                        meanICP += std::chrono::duration_cast<std::chrono::milliseconds>(measurementsICP.at(i)).count();
                        meanFrame += std::chrono::duration_cast<std::chrono::milliseconds>(measurementsFrame.at(i)).count();
                }
                meanTSDF = meanTSDF / measurementsTsdfUpdate.size();
                meanPyramid = meanPyramid / measurementsTsdfUpdate.size();
                meanRaycast = meanRaycast / measurementsTsdfUpdate.size();
                meanICP = meanICP / measurementsTsdfUpdate.size();
                meanFrame = meanFrame / measurementsTsdfUpdate.size();
                double varianceTSDF = 0;
                double variancePyramid = 0;
                double varianceRaycast = 0;
                double varianceICP = 0;
                double varianceFrame = 0;
                for (size_t i = 0; i < measurementsTsdfUpdate.size(); i++)
                {
                        varianceTSDF += pow((meanTSDF - std::chrono::duration_cast<std::chrono::milliseconds>(measurementsTsdfUpdate.at(i)).count()), 2.0);
                        variancePyramid += pow((meanPyramid - std::chrono::duration_cast<std::chrono::milliseconds>(measurementsPyramidCompute.at(i)).count()), 2.0);
                        varianceRaycast += pow((meanRaycast - std::chrono::duration_cast<std::chrono::milliseconds>(measurementsRayCast.at(i)).count()), 2.0);
                        varianceICP += pow((meanICP - std::chrono::duration_cast<std::chrono::milliseconds>(measurementsICP.at(i)).count()), 2.0);
                        varianceFrame += pow((meanFrame - std::chrono::duration_cast<std::chrono::milliseconds>(measurementsFrame.at(i)).count()), 2.0);
                }
                varianceTSDF = varianceFrame / measurementsTsdfUpdate.size();
                variancePyramid = variancePyramid / measurementsTsdfUpdate.size();
                varianceRaycast = varianceRaycast / measurementsTsdfUpdate.size();
                varianceICP = varianceICP / measurementsTsdfUpdate.size();
                varianceFrame = varianceFrame / measurementsTsdfUpdate.size();

                std::cout << "TSDF: Mean: " << meanTSDF << "ms"
                          << " Variance: " << varianceTSDF << "ms" << std::endl;
                std::cout << "Pyramid: Mean: " << meanPyramid << "ms"
                          << " Variance: " << variancePyramid << "ms" << std::endl;
                std::cout << "Raycast: Mean: " << meanRaycast << "ms"
                          << " Variance: " << varianceRaycast << "ms" << std::endl;
                std::cout << "ICP: Mean: " << meanICP << "ms"
                          << " Variance: " << varianceICP << "ms" << std::endl;
                std::cout << "Frame: Mean: " << meanFrame << "ms"
                          << " Variance: " << varianceFrame << "ms" << std::endl;
#endif
#if SAVE_IMAGE_MODE
                auto marchingCubesStart = std::chrono::high_resolution_clock::now();
                run_marching_cubes(grid, idx);
                auto marchingCubesStop = std::chrono::high_resolution_clock::now();
                std::cout << "Computing marching cubes took: " << std::chrono::duration_cast<std::chrono::milliseconds>(marchingCubesStop - marchingCubesStart).count() << " ms" << std::endl;
#endif
                return result;
        }
}
