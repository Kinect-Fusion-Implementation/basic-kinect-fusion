#pragma once

#include "Eigen.h"
#include "PointCloud.h"
#include "PointCloudPyramid.h"
#include "vector"
#include "tuple"

//TODO: Implement correspondance search between Vertecies
//TODO: Implement Point to Plane ICP with given corresponances

class ICPOptimizer {
public:
    ICPOptimizer(Eigen::Matrix<float, 3, 4>& cameraMatrix, float vertex_diff_threshold, float normal_diff_threshold, std::vector<int> iterations_per_level) {
        // Initialized with Camera Matrix and threshold values. These should stay the same for all iterations and frames
        m_cameraMatrix = cameraMatrix;

        // Sets threshold values for vertex and normal difference in correspondance search
        m_vertex_diff_threshold = vertex_diff_threshold;
        m_normal_diff_threshold = normal_diff_threshold;

        // Sets number of iterations per level
        m_iterations_per_level = iterations_per_level;
    }
    void setCameraMatrix(Eigen::Matrix<float, 3, 4>& cameraMatrix) {
        m_cameraMatrix = cameraMatrix;
    }
    void setVertex_diff_threshold(float vertex_diff_threshold) {
        m_vertex_diff_threshold = vertex_diff_threshold;
    }
    void setNormal_diff_threshold(float normal_diff_threshold) {
        m_normal_diff_threshold = normal_diff_threshold;
    }

    // We expect the vertecies of both pointclouds to have 3d coordinates with respect to the camera frame and not the global frame
    Matrix4f optimize(PointCloudPyramid& sourcePyramid, PointCloudPyramid& targetPyramid, Matrix4f globalToPreviousFrame) {
        // source -> PointCloud of k-th frame, target -> PointCloud of k-1-th frame
        std::vector<PointCloud> sourcePointClouds = sourcePyramid.getPointClouds();
        std::vector<PointCloud> targetPointClouds = targetPyramid.getPointClouds();
        // Initialize frame transformation with identity matrix
        Matrix4f previousToCurrentFrame = Matrix4f::Identity();
        // Iterate over levels for pointClouds | We assume that levels match for both pyramids
        for (unsigned int i = 0; i < sourcePointClouds.size(); i++) {
            for (unsigned int k = 0; k < m_iterations_per_level[i]; k++)
                previousToCurrentFrame = pointToPlaneICP(findCorrespondances(sourcePointClouds[i], targetPointClouds[i], previousToCurrentFrame, globalToPreviousFrame, m_cameraMatrix), globalToPreviousFrame, previousToCurrentFrame);
        }

        return globalToPreviousFrame * previousToCurrentFrame;
    }

private:
    Eigen::Matrix<float, 3, 4> m_cameraMatrix;
    // Threshold values for correspondance search
    float m_vertex_diff_threshold;
    float m_normal_diff_threshold;
    // Number of iterations per level
    std::vector<int> m_iterations_per_level;

    // ---- Point to Plane ICP ----

    Matrix4f pointToPlaneICP(const std::vector<std::tuple<Vector3f, Vector3f, Vector3f>>& correspondences, const Matrix4f& globalToPreviousFrame, const Matrix4f& previousToCurrentFrame) {
        Matrix4f output;
        output.setIdentity();

        // designMatrix contains sum of A_t * A matrices
        Eigen::Matrix<float, 6, 6> designMatrix;
        designMatrix.setZero();

        // designVector contains sum of A_t * b vectors
        Eigen::Matrix<float, 6, 1> designVector;
        designVector.setZero();

        for (unsigned int i = 0; i < correspondences.size(); i++) {
            // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1
            Vector3f sourceVertex = std::get<0>(correspondences[i]);
            Vector3f targetVertex = std::get<1>(correspondences[i]);
            Vector3f targetNormal = std::get<2>(correspondences[i]);

            // Build summand matrix and vector for current correspondance
            std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> system = buildSummandMatrixAndVector(
                    sourceVertex, targetVertex, targetNormal);
            designMatrix += std::get<0>(system);
            designVector += std::get<1>(system);

        }
        // solution -> (alpha, beta, gamma, tx, ty, tz)
        Eigen::Matrix<float, 6, 1> solution = (designMatrix.llt()).solve(designVector);
        output <<  1, solution(0), -solution(2), solution(3),
                    -solution(0), 1, solution(1), solution(4),
                    solution(2), -solution(1), 1, solution(5),
                    0, 0, 0, 1;
        return output;
    }

    std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> buildSummandMatrixAndVector(Vector3f& sourceVertex, Vector3f& targetVertex, Vector3f& targetNormal) {
        // Returns the solution of the system of equations for point to plane ICP for one summand of the cost function
        // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1

        // solution -> (alpha, beta, gamma, tx, ty, tz)
        Eigen::Matrix<float, 6, 1> solution;
        solution.setZero();
        // G contains  the skew-symmetric matrix form of the sourceVertex
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



    // ---- Correspondance Search ----

    // Source -> PointCloud of k-th frame, Target -> PointCloud of k-1-th frame | Rendered PointCloud at k-1-th position
    // FrameToFrameTransformation -> Transformation from k-th to k-1-th frame
    std::vector<std::tuple<Vector3f, Vector3f, Vector3f>> findCorrespondances(const PointCloud& sourcePointCloud, const PointCloud& targetPointCloud, const Matrix4f& frameToFrameTransformation, const Matrix4f& globalToPrevFrameTransform, const Eigen::Matrix<float, 3, 4>& cameraMatrix) {
        // Find correspondances between sourcePointCloud and targetPointCloud
        // Return correspondences i.e. V_k, N_k, V_k-1
        std::vector<std::tuple<Vector3f, Vector3f, Vector3f>> correspondences;

        // We will later compute the transformation matrix between the k-th and k-1-th frame which is why we need to iterate over the sourePointCloud to avoid computing inverse transformations
        for (unsigned int i = 0; i < sourcePointCloud.getPoints().size(); i++) {
            // Vertex -> V_k, Normal -> N_k
            Vector3f sourceVertex = sourcePointCloud.getPoints()[i];
            Vector3f normalVertex = sourcePointCloud.getNormals()[i];

            std::tuple<Vector3f, Vector3f> correspondingPoint = findCorrespondingPoint(sourceVertex, normalVertex, targetPointCloud, frameToFrameTransformation, globalToPrevFrameTransform, cameraMatrix);
            if (std::get<0>(correspondingPoint) != Vector3f::Zero() && std::get<1>(correspondingPoint) != Vector3f::Zero()) {
                // Add v_k, v_k-1, n_k-1 to correspondences
                correspondences.push_back(std::make_tuple(sourceVertex, std::get<0>(correspondingPoint), std::get<1>(correspondingPoint)));
            }

        }
        return correspondences;
    }

    // PyramidLevel -> Used to determine the size of the projected window
    std::tuple<Vector3f, Vector3f> findCorrespondingPoint(const Vector3f& sourceVertex, const Vector3f& sourceNormal, const PointCloud& targetPointCloud, const Matrix4f& frameToFrameTransformation, const Matrix4f& globalToPrevFrameTransform, const Eigen::Matrix<float, 3, 4>& cameraMatrix) {
        // Find corresponding point in sourcePointCloud for given targetVertex
        // Return corresponding point and normal
        Vector3f correspondingPoint = Vector3f::Zero();
        Vector3f correspondingNormal = Vector3f::Zero();

        Vector3f transformedSourceVector = cameraMatrix * frameToFrameTransformation * homogenize_3d(sourceVertex);
        Vector2f projectedSourceVector = dehomogenize_2d(transformedSourceVector);

        // FIXME: Change Lookup for Source Vertex i.e. second coordinate with row major

        Vector3f targetVertex = targetPointCloud.getPoints()[projectedSourceVector[0]];
        Vector3f targetNormal = targetPointCloud.getNormals()[projectedSourceVector[0]];

        // Check whether targetVertex is valid i.e. not MINF (M(u) = 1)
        if (targetVertex[0] == MINF || targetVertex[1] == MINF || targetVertex[2] == MINF || targetNormal[0] == MINF || targetNormal[1] == MINF || targetNormal[2] == MINF) {
            return std::make_tuple(correspondingPoint, correspondingNormal);
        }

        Vector4f targetVertexGlobal = globalToPrevFrameTransform * homogenize_3d(targetNormal);
        // T_g,k = T_g,k-1 * T_k-1,k
        Vector4f sourceVertexGlobal = globalToPrevFrameTransform * frameToFrameTransformation *
                homogenize_3d(sourceVertex);

        // Check if the distance between the sourceVertex and the targetVertex is too large (||T_g,k * v_k - T_g,k-1 * v_k-1|| > epsilon_d)
        if ((dehomogenize_3d(targetVertexGlobal) - dehomogenize_3d(sourceVertexGlobal)).norm() <= m_vertex_diff_threshold) {
            return std::make_tuple(correspondingPoint, correspondingNormal);
        }

        // Extract rotation of Frame k-1 to Frame k | (We don't to take the global rotation into account since we are only interested in the rotation between the two frames)
        Matrix3f rotation = (frameToFrameTransformation).block<3, 3>(0, 0);
        if (targetNormal.dot(rotation * sourceNormal) <= m_normal_diff_threshold) {
            return std::make_tuple(correspondingPoint, correspondingNormal);
        }

        correspondingPoint = sourceVertex;
        correspondingNormal = sourceNormal;

        return std::make_tuple(correspondingPoint, correspondingNormal);
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

    Vector2f dehomogenize_2d(const Vector3f point) {
        return Vector2f(point[0] / point[2], point[1] / point[2]);
    }


};