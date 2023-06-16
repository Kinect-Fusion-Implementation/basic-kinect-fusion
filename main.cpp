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

		PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 3, sigmaS, sigmaR);
		const std::vector<PointCloud>& cloud = pyramid.getPointClouds();

		counter++;
	}

	return result;
}
