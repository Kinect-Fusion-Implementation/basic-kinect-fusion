#pragma once

#include "Eigen.h"
#include "libfreenect.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

typedef unsigned char BYTE;

class KinectSensor
{
private:

public:
	KinectSensor();

	bool init();

	bool processNextFrame();

	unsigned int getCurrentFrameCnt();

	// get current color data
	BYTE *getColorRGBX();

	// get current depth data
	float *getDepth();

	// color camera info
	Eigen::Matrix3f getColorIntrinsics();

	Eigen::Matrix4f getColorExtrinsics();

	unsigned int getColorImageWidth();

	unsigned int getColorImageHeight();

	// depth (ir) camera info
	Eigen::Matrix3f getDepthIntrinsics();

	Eigen::Matrix4f getDepthExtrinsics();

	unsigned int getDepthImageWidth();

	unsigned int getDepthImageHeight();
};