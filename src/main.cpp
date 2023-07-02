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

    // NOTE: "using invTraj is correct, since that maps from view to world space"
    Matrix4f prevFrameToGlobal = sensor.getTrajectory().inverse();
    std::cout << "From prev. frame to global: " << std::endl << prevFrameToGlobal << std::endl;
    PointCloudPyramid pyramid1(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);

    for (unsigned int i = 0; i < 1; ++i) {
        if (!sensor.processNextFrame()) {
            std::cout << "Failed to read test frame " << i << std::endl;
            return -1;
        }
    }

    // This matrix maps from k-th frame to global frame
    Matrix4f gt = sensor.getTrajectory().inverse();
    std::cout << "Curr. frame to global: " << std::endl << gt << std::endl;
    PointCloudPyramid pyramid2(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);

    Eigen::Matrix3f cameraMatrix = sensor.getDepthIntrinsics();
    float vertex_diff_threshold = 0;
    float normal_diff_threshold = 0;
    std::vector<int> iterations_per_level = {10, 5, 4};
    ICPOptimizer optimizer(sensor, vertex_diff_threshold, normal_diff_threshold, iterations_per_level);
    Matrix4f est = optimizer.optimize(pyramid1, pyramid2, Eigen::Matrix4f::Identity());

    std::cout << "Ground Truth: " << std::endl << gt << std::endl;
    std::cout << "Estimated: " << std::endl << est << std::endl;
    std::cout << "Diff to ID Error: " << (gt.inverse() * est).norm() << std::endl;
    std::cout << "Diff Error: " << (gt-est).norm() << std::endl;
}

int main() {
    return icp_accuracy_test();
    /*int result = 0;
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
    int idx = 0;
    while (sensor.processNextFrame())
    {
        float *depth = sensor.getDepth();
        // Trajectory:	 	world -> view space (Extrinsics)
        // InvTrajectory:	view -> world space (Pose)
        std::cout << "Trajectory:\n" << sensor.getTrajectory() << std::endl;
        // Somehow all of this code does not work with the GT trajectory (extrinsics)
        grid.updateTSDF(sensor.getDepthExtrinsics(), sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 10.0f);
        idx++;
        if (idx > 200) {
            break;
        }
        run_marching_cubes(grid, idx);
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

    }

    return result;*/

}
