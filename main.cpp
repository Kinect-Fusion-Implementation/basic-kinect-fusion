#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "PointCloudPyramid.h"

int main() {
	int result = 0;

	VirtualSensor sensor;
	sensor.init("../../../Data/rgbd_dataset_freiburg1_xyz/");

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
		const unsigned blockSize = 7;
		// If there are more pixels than this in the smoothing window, we don't smoothe.
		const unsigned invalidThreshold = 10;

		PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, invalidThreshold, sigmaS, sigmaR);
		const std::vector<PointCloud>& cloud = pyramid.getPointClouds();

		counter++;
	}

	return result;
}
