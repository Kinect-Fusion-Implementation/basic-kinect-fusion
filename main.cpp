#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "VoxelGrid.h"
#include "PointCloudPyramid.h"
#include "Configuration.h"

int main()
{
	int result = 0;
	VirtualSensor sensor;
	sensor.init(Configuration::getDataSetPath());

	VoxelGrid grid(Vector3f(-1.0f, -1.0f, -1.0f), 200, 200, 200, 0.25f);
	Vector3f res = grid.voxelGridCenterToWorld(Vector3i(1, 1, 1));

	while (sensor.processNextFrame())
	{
		// grid.updateTSDF(sensor.getDepthExtrinsics(), sensor.getDepthIntrinsics(), sensor.getDepth(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 10);
		float *depth = sensor.getDepth();

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

		PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);
		const std::vector<PointCloud> &cloud = pyramid.getPointClouds();

		break;
	}

	return result;
}
