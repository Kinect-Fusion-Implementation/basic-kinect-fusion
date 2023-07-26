#pragma once

#include "../kinect_fusion/Eigen.h"
#include "PointCloud.h"
#include "PointCloudPyramid.h"
#include "vector"
#include "tuple"

// TODO: Implement PointToPoint ICP with given corresponances
// TODO: Implement other correspondance search method (No second downsampling)
// TODO: Implement Symmetric ICP

class ICPOptimizer
{
private:
    unsigned int m_width;
    unsigned int m_height;
    Eigen::Matrix3f m_intrinsics;
    // Threshold values for correspondance search
    float m_vertex_diff_threshold;
    float m_normal_diff_threshold;
    // Number of iterations per level
    std::vector<int> m_iterations_per_level;
    // Weight for point to point correspondances
    float m_pointToPointWeight;

public:
    ICPOptimizer(Matrix3f intrinsics, unsigned int width, unsigned int height, float vertex_diff_threshold, float normal_diff_threshold, std::vector<int> &iterations_per_level, float pointToPointWeight = 0.5);

    /**
     * We expect the vertecies of both pointclouds to have 3d coordinates with respect to the camera frame and not the global frame
     * source -> PointCloud of k-th frame, target -> PointCloud of k-1-th frame
     */
    Matrix4f optimize(PointCloudPyramid &sourcePyramid, Vector3f *vertexMap, Vector3f *normalMap, const Matrix4f &prevFrameToGlobal);

private:
    // ---- Correspondance Search ----
    /**
     * Correspondance Search
     * returns: list of correspondences i.e. V_k, N_k, V_k-1
     *
     */
    std::tuple<Vector3f*, Vector3f*> findCorrespondences(PointCloud &currentPointCloud, Vector3f *vertexMap, Vector3f *normalMap, const Matrix4f &currentFrameToPrevFrameTransformation, const Matrix4f &prevFrameToGlobalTransform);

    Matrix4f pointToPointAndPlaneICP(const std::vector<std::tuple<Vector3f, Vector3f, Vector3f, Vector3f>> &correspondences, const Matrix4f &globalToPreviousFrame, const Matrix4f &currentToPreviousFrame);

    std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> buildPointToPlaneErrorSystem(Vector3f &sourceVertex, Vector3f &targetVertex, Vector3f &targetNormal);

    std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> buildPointToPointErrorSystem(Vector3f &sourceVertex, Vector3f &targetVertex, Vector3f &targetNormal);
};