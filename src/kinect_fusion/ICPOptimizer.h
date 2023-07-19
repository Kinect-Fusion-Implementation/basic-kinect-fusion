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

    void setVertex_diff_threshold(float vertex_diff_threshold) {
        m_vertex_diff_threshold = vertex_diff_threshold;
    }
    void setNormal_diff_threshold(float normal_diff_threshold) {
        m_normal_diff_threshold = normal_diff_threshold;
    }

    // We expect the vertecies of both pointclouds to have 3d coordinates with respect to the camera frame and not the global frame
    Matrix4f optimize(PointCloudPyramid& sourcePyramid, PointCloudPyramid& targetPyramid, const Matrix4f& prevFrameToGlobal) {
        // FIXME: For testing purposes we provide the frame to frame transformation matrix
        Eigen::Matrix4f m;
        m << 0.999989,  0.00117746,  0.00197165, -0.00047487,
                -0.00117414, 0.99997,    0.00627088,  0.000739664,
                -0.00197134, -0.00628,    0.999963,    0.011073,
                0,          0,           0,           1;
        // source -> PointCloud of k-th frame, target -> PointCloud of k-1-th frame
        std::vector<PointCloud> sourcePointClouds = sourcePyramid.getPointClouds();
        std::vector<PointCloud> targetPointClouds = targetPyramid.getPointClouds();
        // Initialize frame transformation with identity matrix
        Matrix4f currentToPreviousFrame = Matrix4f::Identity();
        // Iterate over levels for pointClouds | We assume that levels match for both pyramids
        for (int i = sourcePointClouds.size() - 1; i >= 0; i--) {
            std::cout << "Level: " << i << std::endl;
            // Adjust camera matrix for current level of pyramid
            //Matrix3f currentCameraMatrix = m_cameraMatrix / pow(2, i);
            //currentCameraMatrix(2, 2) = 1;
            Matrix3f currentCameraMatrix = m_cameraMatrix;
            //int width = std::floor(m_width / pow(2, i));
            //int height = std::floor(m_height / pow(2, i));
            int width = m_width;
            int height = m_height;
            std::cout << "Current Camera Matrix: " << std::endl << currentCameraMatrix << std::endl;
            std::cout << "Current Width: " << width << std::endl;
            std::cout << "Current Height: " << height << std::endl;
            for (unsigned int k = 0; k < m_iterations_per_level[i]; k++) {
                std::vector<std::tuple<Vector3f, Vector3f, Vector3f>> correspondances = findCorrespondances(sourcePointClouds[0], targetPointClouds[i], currentToPreviousFrame, prevFrameToGlobal, currentCameraMatrix, width, height);
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

    // ---- Point to Plane ICP ----

    Matrix4f pointToPointAndPlaneICP(const std::vector<std::tuple<Vector3f, Vector3f, Vector3f>>& correspondences, const Matrix4f& globalToPreviousFrame, const Matrix4f& currentToPreviousFrame) {
        Matrix4f output = Matrix4f::Identity();

        // designMatrix contains sum of A_t * A matrices
        Eigen::Matrix<float, 6, 6> designMatrix = Eigen::Matrix<float, 6, 6>::Zero();

        // designVector contains sum of A_t * b vectors
        Eigen::Matrix<float, 6, 1> designVector = Eigen::Matrix<float, 6, 1>::Zero();

        // --------- CUDAAAAAAAAAAA -----------
        for (unsigned int i = 0; i < correspondences.size(); i++) {
            // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1
            Vector3f currVertex = currentToPreviousFrame.block<3, 3>(0, 0) * std::get<0>(correspondences[i]) + currentToPreviousFrame.block<3, 1>(0, 3);
            Vector3f prevVertex = std::get<1>(correspondences[i]);
            Vector3f prevNormal = std::get<2>(correspondences[i]);

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
    std::vector<std::tuple<Vector3f, Vector3f, Vector3f>> findCorrespondances(const PointCloud& currentPointCloud, const PointCloud& prevPointCloud, const Matrix4f& currentFrameToPrevFrameTransformation, const Matrix4f& prevFrameToGlobalTransform, const Eigen::Matrix3f cameraMatrix, int width, int height) {
        // Return correspondences i.e. V_k, N_k, V_k-1
        std::vector<std::tuple<Vector3f, Vector3f, Vector3f>> correspondences;

        // --------- CUDAAAAAAAAAAA -----------
        for (unsigned int i = 0; i < currentPointCloud.getPoints().size(); i++) {
            Vector3f currentVertex = currentPointCloud.getPoints()[i];
            Vector3f currentNormal = currentPointCloud.getNormals()[i];
            if (currentVertex.allFinite() && currentNormal.allFinite()) {
                // Find corresponding vertex in previous frame
                std::tuple<Vector3f, Vector3f> correspondingPoint = findCorrespondingPoint(currentVertex, currentNormal, prevPointCloud, currentFrameToPrevFrameTransformation, prevFrameToGlobalTransform, cameraMatrix, width, height);
                if (std::get<0>(correspondingPoint).allFinite() && std::get<1>(correspondingPoint).allFinite()) {
                    // V_k, V_k-1, N_k-1
                    correspondences.push_back(std::make_tuple(currentVertex, std::get<0>(correspondingPoint), std::get<1>(correspondingPoint)));
                }
            }
        }
        return correspondences;
    }

    std::tuple<Vector3f, Vector3f> findCorrespondingPoint(const Vector3f& currentVertex, const Vector3f& currentNormal, const PointCloud& prevPointCloud, const Matrix4f& currentFrameToPrevFrameTransformation, const Matrix4f& prevFrameToGlobalTransform, const Matrix3f& cameraMatrix, int width, int height) {
        // Find corresponding vertex in previous frame
        Vector3f transformedCurrentVertex = dehomogenize_3d(currentFrameToPrevFrameTransformation * homogenize_3d(currentVertex));
        Vector2f indexedCurrentVertex = dehomogenize_2d(cameraMatrix * transformedCurrentVertex);

        // Check if transformedCurrentVertex is in the image
        if (0 <= indexedCurrentVertex[0] && std::round(indexedCurrentVertex[0]) <= width && 0 <= indexedCurrentVertex[1] && std::round(indexedCurrentVertex[1]) <= height) {
            int x = std::round(indexedCurrentVertex[0]) + std::round(indexedCurrentVertex[1]) * width;
            Vector3f prevMatchedVertex = prevPointCloud.getPoints()[x];
            Vector3f prevMatchedNormal = prevPointCloud.getNormals()[x];

            if (prevMatchedVertex.allFinite() && prevMatchedVertex.allFinite()) {
                if ((transformedCurrentVertex - prevMatchedVertex).norm() < m_vertex_diff_threshold){
                    Matrix3f rotation = (currentFrameToPrevFrameTransformation).block<3, 3>(0, 0);
                    if (prevMatchedNormal.dot(rotation * currentNormal) < m_normal_diff_threshold) {
                        return std::make_tuple(prevMatchedVertex, prevMatchedNormal);
                    }
                }
            }
        }
        // No corresponding point found -> return invalid point
        return std::make_tuple(Vector3f(MINF, MINF, MINF), Vector3f(MINF, MINF, MINF));
    }


        /*
        // Source -> PointCloud of k-th frame, Target -> PointCloud of k-1-th frame | Rendered PointCloud at k-1-th position
        // FrameToFrameTransformation -> Transformation from k-th to k-1-th frame
        std::vector<std::tuple<Vector3f, Vector3f, Vector3f>> findCorrespondances(const PointCloud& sourcePointCloud, const PointCloud& targetPointCloud, const Matrix4f& frameToFrameTransformation, const Matrix4f& prevFrameToGlobalTransform, const Eigen::Matrix3f cameraMatrix, float width) {
            // Find correspondances between sourcePointCloud and targetPointCloud
            // Return correspondences i.e. V_k, N_k, V_k-1
            std::vector<std::tuple<Vector3f, Vector3f, Vector3f>> correspondences;

            // We will later compute the transformation matrix between the k-th and k-1-th frame which is why we need to iterate over the sourePointCloud to avoid computing inverse transformations
            for (unsigned int i = 0; i < sourcePointCloud.getPoints().size(); i++) {
                // Vertex -> V_k, Normal -> N_k
                Vector3f sourceVertex = sourcePointCloud.getPoints()[i];
                Vector3f sourceNormal = sourcePointCloud.getNormals()[i];
                if (sourceVertex[0] == MINF || sourceVertex[1] == MINF || sourceVertex[2] == MINF || sourceNormal[0] == MINF || sourceNormal[1] == MINF || sourceNormal[2] == MINF) {
                    continue;
                }

                std::tuple<Vector3f, Vector3f> correspondingPoint = findCorrespondingPoint(sourceVertex, sourceNormal, targetPointCloud, frameToFrameTransformation, prevFrameToGlobalTransform, cameraMatrix, width);
                if (std::get<0>(correspondingPoint) != Vector3f::Zero() && std::get<1>(correspondingPoint) != Vector3f::Zero()) {
                    // Add v_k, v_k-1, n_k-1 to correspondences
                    correspondences.push_back(std::make_tuple(sourceVertex, std::get<0>(correspondingPoint), std::get<1>(correspondingPoint)));
                }

            }
            return correspondences;
        }*/
/*
    // PyramidLevel -> Used to determine the size of the projected window
    std::tuple<Vector3f, Vector3f> findCorrespondingPoint(const Vector3f& sourceVertex, const Vector3f& sourceNormal, const PointCloud& targetPointCloud, const Matrix4f& frameToFrameTransformation, const Matrix4f& globalToPrevFrameTransform, const Matrix3f& cameraMatrix, float width) {
        // Find corresponding point in sourcePointCloud for given targetVertex
        // Return corresponding point and normal
        Vector3f correspondingPoint = Vector3f::Zero();
        Vector3f correspondingNormal = Vector3f::Zero();

        Vector3f transformedSourceVector = cameraMatrix * dehomogenize_3d(frameToFrameTransformation * homogenize_3d(sourceVertex));
        Vector2f projectedSourceVector = dehomogenize_2d(transformedSourceVector);

        // Get corresponding point in targetImage (V_k-1) | Row-Major Order
        int x = std::floor(projectedSourceVector[0] + projectedSourceVector[1] * width);
        if (x < 0 || x > targetPointCloud.getPoints().size()) {
            return std::make_tuple(correspondingPoint, correspondingNormal);
        }
        Vector3f targetVertex = targetPointCloud.getPoints()[x];
        Vector3f targetNormal = targetPointCloud.getNormals()[x];

        if (!targetVertex.allFinite() && !targetNormal.allFinite()) {
            return std::make_tuple(correspondingPoint, correspondingNormal);
        }
        // Check whether targetVertex is valid i.e. not MINF (M(u) = 1)
        if (targetVertex.allFinite() && targetNormal.allFinite()) {
            // Check whether the targetVertex is in the same direction as the sourceNormal (n_k-1 * (v_k-1 - t_g,k-1) > 0)
            if (targetNormal.dot(targetVertex - dehomogenize_3d(globalToPrevFrameTransform * homogenize_3d(sourceVertex))) > 0) {
                correspondingPoint = targetVertex;
                correspondingNormal = targetNormal;
            }
        }
        if (targetVertex[0] == MINF || targetVertex[1] == MINF || targetVertex[2] == MINF || targetNormal[0] == MINF || targetNormal[1] == MINF || targetNormal[2] == MINF) {
            return std::make_tuple(correspondingPoint, correspondingNormal);
        }

        // T_g,k = T_g,k-1 * T_k-1,k. The translation part of T_g,k-1 cancels out and the rotation is ||R|| = 1
        Vector3f sourceVertexFramed = dehomogenize_3d(frameToFrameTransformation * homogenize_3d(sourceVertex));
        // Check if the distance between the sourceVertex and the targetVertex is too large (||T_g,k * v_k - T_g,k-1 * v_k-1|| > epsilon_d)
        if ((targetVertex - sourceVertexFramed).norm() > 0.05) {
            return std::make_tuple(correspondingPoint, correspondingNormal);
        }

        // Extract rotation of Frame k-1 to Frame k | (We don't to take the global rotation into account since we are only interested in the rotation between the two frames)
        Matrix3f rotation = (frameToFrameTransformation).block<3, 3>(0, 0);
        if (targetNormal.dot(rotation * sourceNormal) > 0.05) {
            return std::make_tuple(correspondingPoint, correspondingNormal);
        }

        correspondingPoint = targetVertex;
        correspondingNormal = targetNormal;

        return std::make_tuple(correspondingPoint, correspondingNormal);
    }*/

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