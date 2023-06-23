#pragma once
#include "Eigen.h"

class PointCloud
{
private:
	std::vector<Vector3f> m_points;
	std::vector<Vector3f> m_normals;
	PointCloud() = delete;

public:

	/**
	 * represents the vertex and point map, relatively to the camera (in the camera frame)
	 */
	PointCloud(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, const unsigned int width, const unsigned int height, int level, const unsigned int maxDistance = 10)
	{
		// Get depth intrinsics.
		float fovX = depthIntrinsics(0, 0);
		float fovY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);
		const float maxDistanceHalved = maxDistance / 2.f;

		// Compute inverse depth extrinsics.
		Matrix4f depthExtrinsicsInv = depthExtrinsics.inverse();
		Matrix3f rotationInv = depthExtrinsicsInv.block(0, 0, 3, 3);
		Vector3f translationInv = depthExtrinsicsInv.block(0, 3, 3, 1);

		// Back-project the pixel depths into the camera space.
		std::vector<Vector3f> pointsTmp(width * height);

		// For every pixel row.
#pragma omp parallel for
		for (int v = 0; v < height; ++v)
		{
			// For every pixel in a row.
			for (int u = 0; u < width; ++u)
			{
				unsigned int idx = v * width + u; // linearized index
				float depth = depthMap[idx];
				if (depth == MINF)
				{
					pointsTmp[idx] = Vector3f(MINF, MINF, MINF);
				}
				else
				{
					// Back-projection to camera space.
					pointsTmp[idx] = rotationInv * Vector3f((u - cX) / fovX * depth, (v - cY) / fovY * depth, depth) + translationInv;
				}
			}
		}

		// We need to compute derivatives and then the normalized normal vector (for valid pixels).
		std::vector<Vector3f> normalsTmp(width * height);

#pragma omp parallel for
		for (int v = 1; v < height - 1; ++v)
		{
			for (int u = 1; u < width - 1; ++u)
			{
				unsigned int idx = v * width + u; // linearized index

				const float du = 0.5f * (depthMap[idx + 1] - depthMap[idx - 1]);
				const float dv = 0.5f * (depthMap[idx + width] - depthMap[idx - width]);
				if (!std::isfinite(du) || !std::isfinite(dv) || abs(du) > maxDistanceHalved || abs(dv) > maxDistanceHalved)
				{
					normalsTmp[idx] = Vector3f(MINF, MINF, MINF);
					continue;
				}

				// TODO: Compute the normals using central differences.
				normalsTmp[idx] = (pointsTmp[idx + 1] - pointsTmp[idx - 1]).cross(pointsTmp[idx + width] - pointsTmp[idx - width]); // Needs to be replaced.
				normalsTmp[idx].normalize();
			}
		}

		// We set invalid normals for border regions.
		for (int u = 0; u < width; ++u)
		{
			normalsTmp[u] = Vector3f(MINF, MINF, MINF);
			normalsTmp[u + (height - 1) * width] = Vector3f(MINF, MINF, MINF);
		}
		for (int v = 0; v < height; ++v)
		{
			normalsTmp[v * width] = Vector3f(MINF, MINF, MINF);
			normalsTmp[(width - 1) + v * width] = Vector3f(MINF, MINF, MINF);
		}

		
		ImageUtil::saveNormalMapToImage((float*)normalsTmp.data(), width, height, std::string("NormalMap_") + std::to_string(level), "Saving normal map...");
		
		// We filter out measurements where either point or normal is invalid.
		const unsigned nPoints = pointsTmp.size();
		m_points.reserve(std::floor(float(nPoints)));
		m_normals.reserve(std::floor(float(nPoints)));

		for (int i = 0; i < nPoints; i++)
		{
			const auto& point = pointsTmp[i];
			const auto& normal = normalsTmp[i];

			if (point.allFinite() && normal.allFinite())
			{
				m_points.push_back(point);
				m_normals.push_back(normal);
			}
		}
	}

	std::vector<Vector3f>& getPoints()
	{
		return m_points;
	}

	const std::vector<Vector3f>& getPoints() const
	{
		return m_points;
	}

	std::vector<Vector3f>& getNormals()
	{
		return m_normals;
	}

	const std::vector<Vector3f>& getNormals() const
	{
		return m_normals;
	}
};
