#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "TSDFVoxelGrid.h"
#include "PointCloudPyramid.h"

int main() {
	int result = 0;
    // Dominik: std::string filenameIn = "../../cv-data/RGBD-Dataset-Freiburg1/";
    std::string filenameIn = "../../../Data/rgbd_dataset_freiburg1_xyz/";
	VirtualSensor sensor;
	sensor.init(filenameIn);

	VoxelGrid grid(3000, 3000, 1);

	int counter = 0;
	while (counter == 0 && sensor.processNextFrame()) {
		float* depth = sensor.getDepth();

		float sigmaS(0.1);
		float sigmaR(0.1);

		// Number of subsampling levels
		const unsigned levels = 2;
		// Size of smoothing window
		const unsigned windowSize = 7;
		// Size of subsampling window
		const unsigned blockSize = 3;

		PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);
		const std::vector<PointCloud>& cloud = pyramid.getPointClouds();

		counter++;
	}

	return result;
}
