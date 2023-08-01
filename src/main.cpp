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

int run_kinect_fusion(float vertexDiffThreshold, float normalDiffThreshold, std::vector<float> &l1normDifferences, std::vector<float> &l2normDifferences)
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
        VoxelGrid grid(Vector3f(-3.0, -3.0, -3.0), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), scale, truncation);
        // On level 0, 1, 2
        std::vector<int> iterations_per_level = {4, 4, 3};
        ICPOptimizer optimizer(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), vertexDiffThreshold, normalDiffThreshold, iterations_per_level, 0.2f);

        Matrix4f prevFrameToGlobal = Matrix4f::Identity();
        int idx = 0;
        Matrix4f trajectoryOffset;
        auto totalComputeStart = std::chrono::high_resolution_clock::now();
#if EVAL_MODE
        std::vector<std::chrono::high_resolution_clock::duration> measurementsTsdfUpdate;
        std::vector<std::chrono::high_resolution_clock::duration> measurementsPyramidCompute;
        std::vector<std::chrono::high_resolution_clock::duration> measurementsRayCast;
        std::vector<std::chrono::high_resolution_clock::duration> measurementsICP;
        std::vector<std::chrono::high_resolution_clock::duration> measurementsFrame;
        measurementsICP.reserve(800);
        measurementsPyramidCompute.reserve(800);
        measurementsRayCast.reserve(800);
        measurementsTsdfUpdate.reserve(800);
#endif
        while (sensor.processNextFrame())
        {
                if (idx % 100 == 0)
                {
                        std::cout << "Processing frame " << idx << std::endl;
                }
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
#if EVAL_MODE
                auto frameStart = std::chrono::high_resolution_clock::now();
                auto pyramidStart = std::chrono::high_resolution_clock::now();
#endif

                PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 2, windowSize, blockSize, sigmaR, sigmaS);
#if EVAL_MODE
                auto pyramidEnd = std::chrono::high_resolution_clock::now();
                measurementsPyramidCompute.push_back(pyramidEnd - pyramidStart);
                auto raycastStart = std::chrono::high_resolution_clock::now();
#endif
                RaycastImage raycast = grid.raycastVoxelGrid(prevFrameToGlobal.inverse(), sensor.getDepthIntrinsics());
#if EVAL_MODE
                auto raycastEnd = std::chrono::high_resolution_clock::now();
                measurementsRayCast.push_back(raycastEnd - raycastStart);
#endif
#if SAVE_IMAGE_MODE
                ImageUtil::saveNormalMapToImage((float *)raycast.m_normalMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("RaycastedImage_") + std::to_string(idx), "");
#endif
                // Estimate the pose of the current frame
                Matrix4f estPose;
                Matrix4f gt_extrinsics = sensor.getTrajectory() * trajectoryOffset;
                Matrix4f gt_pose = gt_extrinsics.inverse();
#if EVAL_MODE
                auto icpStart = std::chrono::high_resolution_clock::now();
#endif
                estPose = optimizer.optimize(pyramid, raycast.m_vertexMapGPU, raycast.m_normalMapGPU, prevFrameToGlobal, idx);
#if EVAL_MODE
                auto icpEnd = std::chrono::high_resolution_clock::now();
                measurementsICP.push_back(icpEnd - icpStart);
#endif
                l1normDifferences.push_back((estPose - gt_pose).lpNorm<1>());
                l2normDifferences.push_back((estPose - gt_pose).lpNorm<2>());

                // Use estimated pose as prevPose for next frame
                prevFrameToGlobal = estPose;
#if EVAL_MODE
                auto updateTSDFStart = std::chrono::high_resolution_clock::now();
#endif
                grid.updateTSDF(estPose.inverse(), sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
#if EVAL_MODE
                auto updateTSDFEnd = std::chrono::high_resolution_clock::now();
                auto frameEnd = std::chrono::high_resolution_clock::now();
                measurementsFrame.push_back(frameEnd - frameStart);
                measurementsTsdfUpdate.push_back(updateTSDFEnd - updateTSDFStart);
#endif
                idx++;
                if(idx > 800) {
                        break;
                }
        }
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
        run_marching_cubes(grid, idx);
#endif

        return 0;
}

int gridSearch()
{
        std::vector<float> thresholds = {0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008};
        float bestMeanL1 = std::numeric_limits<float>::infinity();
        float bestMeanL2 = std::numeric_limits<float>::infinity();
        float bestMeanL1Variance = std::numeric_limits<float>::infinity();
        float bestMeanL2Variance = std::numeric_limits<float>::infinity();
        size_t bestiL1 = -1;
        size_t bestjL1 = -1;
        size_t bestiL2 = -1;
        size_t bestjL2 = -1;
        for (size_t i = 3; i < thresholds.size(); i++)
        {
                std::cout << "At i = " << i << std::endl;
                for (size_t j = 3; j < thresholds.size(); j++)
                {
                        std::cout << "At j = " << j << std::endl;
                        std::vector<float> l1Differences;
                        std::vector<float> l2Differences;
                        l1Differences.reserve(800);
                        l2Differences.reserve(800);
                        // Run for all combinations
                        run_kinect_fusion(thresholds.at(i), thresholds.at(j), l1Differences, l2Differences);
                        float meanL1 = 0;
                        float meanL2 = 0;
                        for (size_t k = 0; k < l1Differences.size(); k++)
                        {
                                meanL1 += l1Differences.at(k);
                                meanL2 += l2Differences.at(k);
                        }
                        meanL1 = meanL1 / l1Differences.size();
                        meanL2 = meanL2 / l2Differences.size();
                        float varianceL1 = 0;
                        float varianceL2 = 0;
                        for (size_t k = 0; k < l1Differences.size(); k++)
                        {
                                varianceL1 += pow((meanL1 - l1Differences.at(k)), 2.0);
                                varianceL2 += pow((meanL2 - l2Differences.at(k)), 2.0);
                        }
                        varianceL1 = varianceL1 / l1Differences.size();
                        varianceL2 = varianceL2 / l1Differences.size();
                        if (meanL1 < bestMeanL1)
                        {
                                bestMeanL1 = meanL1;
                                bestMeanL1Variance = varianceL1;
                                bestiL1 = i;
                                bestjL1 = j;
                        }
                        if (meanL2 < bestMeanL2)
                        {
                                bestMeanL2 = meanL2;
                                bestMeanL2Variance = varianceL2;
                                bestiL2 = i;
                                bestjL2 = j;
                        }
                        std::cout << "L1: Mean: " << meanL1
                                  << " Variance: " << varianceL1 << std::endl;
                        std::cout << "L2: Mean: " << meanL2
                                  << " Variance: " << varianceL2 << std::endl;
                }
        }

        std::cout << "Best L1: Mean: " << bestMeanL1
                  << " with Variance: " << bestMeanL1Variance << " with parameters i = " << bestiL1 << ", j = " << bestjL1
                  << std::endl;
        std::cout << "Best L2: Mean: " << bestMeanL2
                  << " with Variance: " << bestMeanL2Variance << " with parameters i = " << bestiL2 << ", j = " << bestjL2
                  << std::endl;

        return 0;
}

int run_kinect_virtual_sensor()
{
        std::vector<float> thresholds = {0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008};
        // 0.02 is best for xyz (5, 5), 0.04 best for rpy (4, 4)
        float pointThreshold = thresholds.at(5);
        float normalThreshold = thresholds.at(5);
        std::vector<float> l1Differences;
        std::vector<float> l2Differences;
        l1Differences.reserve(800);
        l2Differences.reserve(800);
        run_kinect_fusion(pointThreshold, normalThreshold, l1Differences, l2Differences);
        float meanL1 = 0;
        float meanL2 = 0;
        for (size_t k = 0; k < l1Differences.size(); k++)
        {
                meanL1 += l1Differences.at(k);
                meanL2 += l2Differences.at(k);
                /* code */
        }
        meanL1 = meanL1 / l1Differences.size();
        meanL2 = meanL2 / l2Differences.size();
        float varianceL1 = 0;
        float varianceL2 = 0;
        for (size_t k = 0; k < l1Differences.size(); k++)
        {
                varianceL1 += pow((meanL1 - l1Differences.at(k)), 2.0);
                varianceL2 += pow((meanL2 - l2Differences.at(k)), 2.0);
        }
        varianceL1 = varianceL1 / l1Differences.size();
        varianceL2 = varianceL2 / l1Differences.size();

        std::cout << "L1: Mean: " << meanL1
                  << " Variance: " << varianceL1 << std::endl;
        std::cout << "L2: Mean: " << meanL2
                  << " Variance: " << varianceL2 << std::endl;
        return 0;
}

int main()
{
        run_kinect_virtual_sensor();
}