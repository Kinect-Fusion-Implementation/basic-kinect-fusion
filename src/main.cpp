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
			*/
	}

	return result;
}
