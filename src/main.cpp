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

	int roomWidhtCentimeter = 300;
	int roomHeightCentimeter = 300;
	int roomDepthCentimeter = 300;
	double voxelsPerCentimeter = 1;
	double scale = 1/voxelsPerCentimeter;
	int numberVoxelsWidth = roomWidhtCentimeter * voxelsPerCentimeter; 
	int numberVoxelsHeight = roomHeightCentimeter * voxelsPerCentimeter;
	int numberVoxelsDepth = roomDepthCentimeter * voxelsPerCentimeter;
	VoxelGrid grid(Vector3d(-1.5, -1.5, -1.5), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, scale);
	Vector3d res = grid.voxelGridCenterToWorld(Vector3i(1, 1, 1));

	while (sensor.processNextFrame())
	{
		double *depth = sensor.getDepth();
		grid.updateTSDF(sensor.getDepthExtrinsics(), sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 10.0f);

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
