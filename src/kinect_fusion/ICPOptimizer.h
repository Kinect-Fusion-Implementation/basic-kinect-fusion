#pragma once

#include "Eigen.h"
#include "PointCloud.h"
#include "PointCloudPyramid.h"
#include "vector"
#include "tuple"

//TODO: Implement PointToPoint ICP with given corresponances
//TODO: Implement other correspondance search method (No second downsampling)
//TODO: Implement Symmetric ICP


class ICPOptimizer {
private:
    unsigned int m_width;
    unsigned int m_height;
    Eigen::Matrix3f m_cameraMatrix;
    // Threshold values for correspondance search
    float m_vertex_diff_threshold;
    float m_normal_diff_threshold;
    // Number of iterations per level
    std::vector<int> m_iterations_per_level;
    // Weight for point to point correspondances
    float m_pointToPointWeight;

public:
    ICPOptimizer(VirtualSensor& virtualSensor, float vertex_diff_threshold, float normal_diff_threshold, std::vector<int>& iterations_per_level, float pointToPointWeight = 0.5) {
        // Initialized with Camera Matrix and threshold values. These should stay the same for all iterations and frames
        m_cameraMatrix =  virtualSensor.getDepthIntrinsics();
        m_width = virtualSensor.getDepthImageWidth();
        m_height = virtualSensor.getDepthImageHeight();

        // Sets threshold values for vertex and normal difference in correspondance search
        m_vertex_diff_threshold = vertex_diff_threshold;
        m_normal_diff_threshold = normal_diff_threshold;

        // Sets number of iterations per level
        m_iterations_per_level = iterations_per_level;
    }

    /**
     * We expect the vertecies of both pointclouds to have 3d coordinates with respect to the camera frame and not the global frame 
     * source -> PointCloud of k-th frame, target -> PointCloud of k-1-th frame
     */
    Matrix4f optimize(PointCloudPyramid& sourcePyramid, Vector3f *vertexMap, Vector3f *normalMap, const Matrix4f& prevFrameToGlobal) {
        std::vector<PointCloud> sourcePointClouds = sourcePyramid.getPointClouds();
        // Initialize frame transformation with identity matrix
        Matrix4f currentToPreviousFrame = Matrix4f::Identity();
        // Iterate over levels for pointClouds | We assume that levels match for both pyramids
        for (int i = sourcePointClouds.size() - 1; i >= 0; i--) {
            std::cout << "Level: " << i << std::endl;
            for (unsigned int k = 0; k < m_iterations_per_level[i]; k++) {
                // CUDA:
                // TODO: Should this be always pointcloud 0 or point cloud k?
                std::vector<std::tuple<Vector3f, Vector3f, Vector3f, Vector3f>> correspondances = findCorrespondances(sourcePointClouds[0], vertexMap, normalMap, currentToPreviousFrame, prevFrameToGlobal);
                // TODO: Check if enough correspondances were found
                std::cout << "Level: " << i << " Iteration: " << k << std::endl;
                std::cout << "Number of correspondances: " << correspondances.size() << std::endl;
                Matrix4f inc = pointToPointAndPlaneICP(correspondances, prevFrameToGlobal, currentToPreviousFrame);
                std::cout << "Incremental Matrix: " << std::endl << inc << std::endl;
                std::cout << "Incremental Matrix det: " << std::endl << inc.determinant() << std::endl;
                std::cout << "Incremental Matrix norm: " << std::endl << inc.norm() << std::endl;
                currentToPreviousFrame = inc * currentToPreviousFrame;
                std::cout << "Current to Previous Frame det:" << std::endl << currentToPreviousFrame.determinant() << std::endl;
                std::cout << "Current to Previous Frame: " << std::endl << currentToPreviousFrame << std::endl;
            }
        }
        return prevFrameToGlobal * currentToPreviousFrame;
    }

private:
    // ---- Point to Plane ICP ----

    Matrix4f pointToPointAndPlaneICP(const std::vector<std::tuple<Vector3f,Vector3f, Vector3f, Vector3f>>& correspondences, const Matrix4f& globalToPreviousFrame, const Matrix4f& currentToPreviousFrame) {
        // designMatrix contains sum of all A_t * A matrices
        Eigen::Matrix<float, 6, 6> designMatrix = Eigen::Matrix<float, 6, 6>::Zero();

        // designVector contains sum of all A_t * b vectors
        Eigen::Matrix<float, 6, 1> designVector = Eigen::Matrix<float, 6, 1>::Zero();

        // --------- CUDAAAAAAAAAAA -----------
        for (unsigned int i = 0; i < correspondences.size(); i++) {
            // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1
            Vector3f currVertex = currentToPreviousFrame.block<3, 3>(0, 0) * std::get<0>(correspondences[i]) + currentToPreviousFrame.block<3, 1>(0, 3);
            Vector3f prevVertex = std::get<2>(correspondences[i]);
            Vector3f prevNormal = std::get<3>(correspondences[i]);

            // Construct Linear System to solve
            // Build summand point to point matrix and vector for current correspondance
            if (m_pointToPointWeight > 0) {
                std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> pointSystem = buildPointToPointErrorSystem(
                        currVertex, prevVertex, prevNormal);
                designMatrix += m_pointToPointWeight * std::get<0>(pointSystem);
                designVector += m_pointToPointWeight * std::get<1>(pointSystem);
            }

            // Build summand point to plane matrix and vector for current correspondance
            std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> planeSystem = buildPointToPlaneErrorSystem(
                    currVertex, prevVertex, prevNormal);
            designMatrix += (1-m_pointToPointWeight) * std::get<0>(planeSystem);
            designVector += (1-m_pointToPointWeight) * std::get<1>(planeSystem);
        }
        // solution -> (beta, gamma, alpha, tx, ty, tz)
        Eigen::Matrix<float, 6, 1> solution = (designMatrix.llt()).solve(designVector);

        Matrix4f output;
        output <<  1, solution(2), -solution(1), solution(3),
                -solution(2), 1, solution(0), solution(4),
                solution(1), -solution(0), 1, solution(5),
                0, 0, 0, 1;
        return output;
    }

    std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> buildPointToPlaneErrorSystem(Vector3f& sourceVertex, Vector3f& targetVertex, Vector3f& targetNormal) {
        // Returns the solution of the system of equations for point to plane ICP for one summand of the cost function
        // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1

        // solution -> (beta, gamma, alpha, tx, ty, tz)
        // G contains  the skew-symmetric matrix form of the sourceVertex | FIXME: Add .cross to calcualte skew-symmetric matrix
        // For vector (beta, gamma, alpha, tx, ty, tz) the skew-symmetric matrix form is:
        Eigen::Matrix<float, 3, 6> G;
        G <<    0, -sourceVertex(2), sourceVertex(1), 1, 0, 0,
                sourceVertex(2), 0, -sourceVertex(0), 0, 1, 0,
                -sourceVertex(1), sourceVertex(0), 0, 0, 0, 1;

        // A contains the dot product of the skew-symmetric matrix form of the sourceVertex and the targetNormal and is the matrix we are optimizing over
        Eigen::Matrix<float, 6, 1> A_t = G.transpose() * targetNormal;
        Eigen::Matrix<float, 6, 6> A_tA = A_t * A_t.transpose();
        // b contains the dot product of the targetNormal and the difference between the targetVertex and the sourceVertex
        Eigen::Matrix<float, 6, 1> b = A_t * (targetNormal.transpose() * (targetVertex - sourceVertex));

        return std::make_tuple(A_tA, b);
    }

    std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> buildPointToPointErrorSystem(Vector3f& sourceVertex, Vector3f& targetVertex, Vector3f& targetNormal) {
        // Returns the solution of the system of equations for point to point ICP for one summand of the cost function
        // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1

        // solution -> (beta, gamma, alpha, tx, ty, tz)
        // G contains  the skew-symmetric matrix form of the sourceVertex | FIXME: Add .cross to calcualte skew-symmetric matrix
        // For vector (beta, gamma, alpha, tx, ty, tz) the skew-symmetric matrix form is:
        Eigen::Matrix<float, 3, 6> G;
        G <<    0, -sourceVertex(2), sourceVertex(1), 1, 0, 0,
                sourceVertex(2), 0, -sourceVertex(0), 0, 1, 0,
                -sourceVertex(1), sourceVertex(0), 0, 0, 0, 1;

        // A contains the dot product of the skew-symmetric matrix form of the sourceVertex and the targetNormal and is the matrix we are optimizing over
        Eigen::Matrix<float, 6, 3> A_t = G.transpose();
        Eigen::Matrix<float, 6, 6> A_tA = A_t * A_t.transpose();
        // b contains the dot product of the targetNormal and the difference between the targetVertex and the sourceVertex
        Eigen::Matrix<float, 6, 1> b = A_t * (targetVertex - sourceVertex);

        return std::make_tuple(A_tA, b);
    }



    // ---- Correspondance Search ----
    /**
     * Correspondance Search
     * returns: correspondences i.e. V_k, N_k, V_k-1
     *          returns a list of 
     * 
    */
    std::vector<std::tuple<Vector3f, Vector3f, Vector3f, Vector3f>> findCorrespondances(PointCloud& currentPointCloud, Vector3f *vertexMap, Vector3f *normalMap, const Matrix4f& currentFrameToPrevFrameTransformation, const Matrix4f& prevFrameToGlobalTransform) {
        std::vector<std::tuple<Vector3f, Vector3f, Vector3f, Vector3f>> correspondences;
        // --------- CUDAAAAAAAAAAA -----------
        unsigned int numberPoints = m_width * m_height;
        for (unsigned int i = 0; i < numberPoints; i++) {
            Vector3f currentVertex = currentPointCloud.getPoints()[i];
            Vector3f currentNormal = currentPointCloud.getNormals()[i];
            if (currentVertex.allFinite() && currentNormal.allFinite()) {
                // Find corresponding vertex in previous frame
                std::tuple<Vector3f, Vector3f> correspondingPoint = findCorrespondingPoint(currentVertex, currentNormal, vertexMap, normalMap, currentFrameToPrevFrameTransformation, prevFrameToGlobalTransform);
                if (std::get<0>(correspondingPoint).allFinite() && std::get<1>(correspondingPoint).allFinite()) {
                    // V_k, N_k, V_k-1, N_k-1
                    correspondences.push_back(std::make_tuple(currentVertex, currentNormal,std::get<0>(correspondingPoint), std::get<1>(correspondingPoint)));
                }
            }
        }
        return correspondences;
    }

    /**
     * For a given vertex, returns the corresponding vertex and normal map from the raycasted version
     * returns: a pair of vectors (vertex, normal)
     *          If no valid correspondence is found, the vertices contain MINF values
    */
    std::tuple<Vector3f, Vector3f> findCorrespondingPoint(const Vector3f& currentVertex, const Vector3f& currentNormal, Vector3f *vertexMap, Vector3f *normalMap, const Matrix4f& currentFrameToPrevFrameTransformation, const Matrix4f& prevFrameToGlobalTransform) {
        // Find corresponding vertex in previous frame
        Vector3f transformedCurrentVertex = dehomogenize_3d(currentFrameToPrevFrameTransformation * homogenize_3d(currentVertex));
        Vector2f indexedCurrentVertex = dehomogenize_2d(m_cameraMatrix * transformedCurrentVertex);

        // Check if transformedCurrentVertex is in the image
        if (0 <= indexedCurrentVertex[0] && std::round(indexedCurrentVertex[0]) < m_width && 0 <= indexedCurrentVertex[1] && std::round(indexedCurrentVertex[1]) < m_height) {
            int x = std::round(indexedCurrentVertex[0]) + std::round(indexedCurrentVertex[1]) * m_width;
            Vector3f prevMatchedVertex = vertexMap[x];
            Vector3f prevMatchedNormal = normalMap[x];

            if (prevMatchedVertex.allFinite() && prevMatchedVertex.allFinite()) {
                if ((transformedCurrentVertex - prevMatchedVertex).norm() < m_vertex_diff_threshold){
                    Matrix3f rotation = (currentFrameToPrevFrameTransformation).block<3, 3>(0, 0);
                    if ((1 - prevMatchedNormal.dot(rotation * currentNormal)) < m_normal_diff_threshold) {
                        return std::make_tuple(prevMatchedVertex, prevMatchedNormal);
                    }
                }
            }
        }
        // No corresponding point found -> return invalid point
        return std::make_tuple(Vector3f(MINF, MINF, MINF), Vector3f(MINF, MINF, MINF));
    }

    // Helper methods for homogenization and dehomogenization in 2D and 3D
    Vector4f homogenize_3d(const Vector3f& point) {
        return Vector4f(point[0], point[1], point[2], 1.0f);
    }

    Vector3f homogenize_2d(const Vector2f& point) {
        return Vector3f(point[0], point[1], 1.0f);
    }

    Vector3f dehomogenize_3d(const Vector4f& point) {
        return Vector3f(point[0] / point[3], point[1] / point[3], point[2] / point[3]);
    }

    Vector2f dehomogenize_2d(const Vector3f& point) {
        return Vector2f(point[0] / point[2], point[1] / point[2]);
    }

};