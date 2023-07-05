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

int main()
{
	int result = 0;
	VirtualSensor sensor;
	sensor.init(Configuration::getDataSetPath());

	int roomWidhtMeter = 4;
	int roomHeightMeter = 4;
	int roomDepthMeter = 4;
	float voxelsPerMeter = 50;
	float scale = 1 / voxelsPerMeter;
	int numberVoxelsWidth = roomWidhtMeter * voxelsPerMeter;
	int numberVoxelsHeight = roomHeightMeter * voxelsPerMeter;
	int numberVoxelsDepth = roomDepthMeter * voxelsPerMeter;
	VoxelGrid grid(Vector3f(-2.0, -2.0, -2.0), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, scale);
	int idx = 0;
	while (sensor.processNextFrame())
	{
		float *depth = sensor.getDepth();
		grid.updateTSDF(sensor.getDepthExtrinsics(), sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 10.0f);
		idx++;
		if (idx > 200) {
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

			PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);
			const std::vector<PointCloud> &cloud = pyramid.getPointClouds();

			break;
			*/
	}
	run_marching_cubes(grid);

	return result;
}
