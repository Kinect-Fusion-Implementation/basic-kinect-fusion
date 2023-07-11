#include <iostream>
#include <fstream>

#include "./sensor/VirtualSensor.h"
#include "./kinect_fusion/Eigen.h"
#include "./kinect_fusion/ICPOptimizer.h"
#include "./kinect_fusion/PointCloud.h"
#include "./kinect_fusion/VoxelGrid.h"
#include "./kinect_fusion/PointCloudPyramid.h"
#include "./configuration/Configuration.h"
#include "./visualization/MarchingCubes.h"
#include "./kinect_fusion/ICPOptimizer.h"

int icp_accuracy_test() {
    VirtualSensor sensor;
    std::string filenameIn = "../../Data/rgbd_dataset_freiburg1_xyz/";
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    // Setup
    float sigmaS(2.0);
    float sigmaR(2.0);
    std::cout << "Using sigmaS: " << sigmaS << std::endl;
    std::cout << "Using sigmaR: " << sigmaR << std::endl;
    // Number of subsampling levels
    const unsigned levels = 2;
    // Size of smoothing window
    const unsigned windowSize = 21;
    // Size of subsampling window
    const unsigned blockSize = 3;

    // Decide on first frame
    for (unsigned int i = 0; i < 10; ++i) {
        if (!sensor.processNextFrame()) {
            std::cout << "Failed to read test frame " << i << std::endl;
            return -1;
        }
    }

    // NOTE: "using invTraj is correct, since that maps from view to world space" | T_global_k-1
    Matrix4f prevFrameToGlobal = sensor.getTrajectory().inverse();
    std::cout << "From prev. frame to global: " << std::endl << prevFrameToGlobal << std::endl;
    PointCloudPyramid pyramid1(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);

    for (unsigned int i = 0; i < 1; ++i) {
        if (!sensor.processNextFrame()) {
            std::cout << "Failed to read test frame " << i << std::endl;
            return -1;
        }
    }

    // This matrix maps from k-th frame to global frame -> This is what we want to estimate | T_global_k
    Matrix4f gt = sensor.getTrajectory().inverse();
    std::cout << "Curr. frame to global: " << std::endl << gt << std::endl;
    PointCloudPyramid pyramid2(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);

    // Get true frame to frame transformation i.e. k-th frame to k-1th frame T_k-1_k
    Matrix4f gt_frame_to_prev = prevFrameToGlobal.inverse() * gt;
    std::cout << "kth frame to k-1th frame: " << std::endl << gt_frame_to_prev << std::endl;

    Eigen::Matrix3f cameraMatrix = sensor.getDepthIntrinsics();
    // FIXME: Currently Hardcoded in ICP Optimizer
    float vertex_diff_threshold = 0.3;
    float normal_diff_threshold = 0.3;
    std::vector<int> iterations_per_level = {10, 5, 4};
    ICPOptimizer optimizer(sensor, vertex_diff_threshold, normal_diff_threshold, iterations_per_level);
    Matrix4f est = optimizer.optimize(pyramid2, pyramid1, prevFrameToGlobal);

    std::cout << "Ground Truth: " << std::endl << gt << std::endl;
    std::cout << "Estimated: " << std::endl << est << std::endl;
    std::cout << "Diff to ID Error: " << (gt * est).norm() / (gt.norm() * gt.norm()) << std::endl;
    std::cout << "Diff Error: " << (gt-est).norm() << std::endl;
}


int main() {
    //return icp_accuracy_test();
    int result = 0;
    VirtualSensor sensor;
    sensor.init(Configuration::getDataSetPath());

    int roomWidhtMeter = 6;
    int roomHeightMeter = 6;
    int roomDepthMeter = 6;
    float voxelsPerMeter = 40;
    float scale = 1 / voxelsPerMeter;
    int numberVoxelsWidth = roomWidhtMeter * voxelsPerMeter;
    int numberVoxelsHeight = roomHeightMeter * voxelsPerMeter;
    int numberVoxelsDepth = roomDepthMeter * voxelsPerMeter;
    VoxelGrid grid(Vector3f(-3.0, -3.0, -3.0), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, scale);

    // TODO: Akkumulierter Fehler (ICP?)
    int idx = 0;
    Matrix4f trajectoryOffset;
    while (sensor.processNextFrame())
    {
        float* depth = sensor.getDepth();
        // Trajectory:       world -> view space (Extrinsics)
        // InvTrajectory:    view -> world space (Pose)

        if (idx == 0) {
            trajectoryOffset = sensor.getTrajectory().inverse();
        }
        /*Matrix4f extrinsics = sensor.getTrajectory() * trajectoryOffset;
        std::cout << "Extrinsics:\n" << extrinsics << "\n\n";*/

        grid.updateTSDF(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 0.125f);
        run_marching_cubes(grid, idx);
        idx++;
        if (idx > 50) {
            break;
        }
        /*
            float sigmaS(2.0);
            float sigmaR(2.0);
            std::cout << "Using sigmaS: " << sigmaS << std::endl;
            std::cout << "Using sigmaR: " << sigmaR << std::endl;

            // Number of subsampling levels
            const unsigned levels = 2;
            // Size of smoothing window
            const unsigned windowSize = 21;
            // Size of subsampling window
            const unsigned blockSize = 3;


            // Somehow all of this code does not work with the GT trajectory (extrinsics)
            PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getTrajectory(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);
            const std::vector<PointCloud> &cloud = pyramid.getPointClouds();

            break;
            */
    }

    return result;
}
