#pragma once

#include "../kinect_fusion/Eigen.h"
#include "PointCloudPyramid.h"

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
    ICPOptimizer(){}
    /**
     * We expect the vertecies of both pointclouds to have 3d coordinates with respect to the camera frame and not the global frame
     * source -> PointCloud of k-th frame, target -> PointCloud of k-1-th frame
     */
    Matrix4f optimize(PointCloudPyramid &sourcePyramid, Vector3f *vertexMap, Vector3f *normalMap, const Matrix4f &prevFrameToGlobal, unsigned int frameIdx);

private:
    // ---- Correspondance Search ----
    /**
     * Correspondance Search
     * returns: list of correspondences i.e. V_k, N_k, V_k-1
     *
     */
    Matrix4f pointToPointAndPlaneICP(Vector3f *currentFramePoints, Vector3f *currentFrameNormals, Vector3f *vertexMap, Vector3f *normalMap,
                                     const Matrix4f &currentFrameToPrevFrameTransformation, const Matrix4f &prevFrameToGlobalTransformation, Matrix4f currentFrameToGlobalTransformation,
                                     unsigned int level, unsigned int iteration, unsigned int frameIdx);
};

void buildPointToPlaneErrorSystem(unsigned int idx, Vector3f &currentVertex,
                                  Vector3f &matchedVertex, Vector3f &matchedNormal,
                                  Eigen::Matrix<float, 6, 6> *matrices, Eigen::Matrix<float, 6, 1> *vectors,
                                  float pointToPointWeight);

void buildPointToPointErrorSystem(unsigned int idx, Vector3f &currentVertex,
                                  Vector3f &matchedVertex, Vector3f &matchedNormal,
                                  Eigen::Matrix<float, 6, 6> *matrices, Eigen::Matrix<float, 6, 1> *vectors,
                                  float pointToPointWeight);