#pragma once
#include "../kinect_fusion/Eigen.h"
#include <iostream>

class PointCloud
{
private:
	PointCloud() = delete;
	bool m_pointsOnCPU = false;
	bool m_normalsOnCPU = false;
	unsigned int m_width;
	unsigned int m_height;
	float *m_depthMap;
	Vector3f *m_points;
	Vector3f *m_normals;
	Vector3f *m_points_cpu;
	Vector3f *m_normals_cpu;


public:
	/**
	 * Receives the depthMap pointer that points to the location of the depth values in DEVICE memory
	*/
	PointCloud(float *depthMap, const Matrix3f &depthIntrinsics, const Matrix4f &depthExtrinsics,
			   const unsigned int width, const unsigned int height, int level, const unsigned int maxDistance = 10);

	~PointCloud();
	
	Vector3f *getPoints()
	{
		return m_points;
	}

	Vector3f *getNormals()
	{
		return m_normals;
	}

	Vector3f *getPointsCPU();
	Vector3f *getNormalsCPU();
};
