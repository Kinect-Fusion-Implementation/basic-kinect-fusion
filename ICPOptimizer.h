#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "PointCloud.h"

#include <iostream>

/**
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f &input, T *output)
{
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}

/**
 * Pose increment is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 6.
 */
template <typename T>
class PoseIncrement
{
private:
    T *m_array;

public:
    explicit PoseIncrement(T *const array) : m_array{array} {}

    void setZero()
    {
        for (int i = 0; i < 6; ++i)
            m_array[i] = T(0);
    }

    T *getData() const
    {
        return m_array;
    }

    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
     * beforehand).
     */
    void apply(T *inputPoint, T *outputPoint) const
    {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T *rotation = m_array;
        const T *translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static Matrix4f convertToMatrix(const PoseIncrement<double> &poseIncrement)
    {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double *pose = poseIncrement.getData();
        double *rotation = pose;
        double *translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);
        matrix(0, 1) = float(rotationMatrix[3]);
        matrix(0, 2) = float(rotationMatrix[6]);
        matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);
        matrix(1, 1) = float(rotationMatrix[4]);
        matrix(1, 2) = float(rotationMatrix[7]);
        matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);
        matrix(2, 1) = float(rotationMatrix[5]);
        matrix(2, 2) = float(rotationMatrix[8]);
        matrix(2, 3) = float(translation[2]);

        return matrix;
    }
};

class PointToPlaneConstraint
{
public:
    PointToPlaneConstraint(const Vector3f &sourcePoint, const Vector3f &targetPoint, const Vector3f &targetNormal, const float weight) : m_sourcePoint{sourcePoint},
                                                                                                                                         m_targetPoint{targetPoint},
                                                                                                                                         m_targetNormal{targetNormal},
                                                                                                                                         m_weight{weight}
    {
    }

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        // TODO: Implemented the point-to-plane cost function.
        // The resulting 1D residual should be stored in residuals array. To apply the pose
        // increment (pose parameters) to the source point, you can use the PoseIncrement
        // class.
        // Important: Ceres automatically squares the cost function.

        T source[3];
        fillVector(m_sourcePoint, source);
        T transformed_source[3];
        apply(source, transformed_source, pose);
        residuals[0] = T(m_targetNormal[0]) * (transformed_source[0] - T(m_targetPoint[0])) + T(m_targetNormal[1]) * (transformed_source[1] - T(m_targetPoint[1])) + T(m_targetNormal[2]) * (transformed_source[2] - T(m_targetPoint[2]));

        return true;
    }

    static ceres::CostFunction *create(const Vector3f &sourcePoint, const Vector3f &targetPoint, const Vector3f &targetNormal, const float weight)
    {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
            new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, weight));
    }

    template <typename T>
    static void apply(T *inputPoint, T *outputPoint, const T *pose)
    {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T *rotation = pose;

        const T *translation = pose + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_targetNormal;
    const float m_weight;
    const float LAMBDA = 1.0f;
};

/**
 * ICP optimizer - Abstract Base Class
 */
class ICPOptimizer
{
protected:
    bool m_bUsePointToPlaneConstraints;
    unsigned m_nIterations;

public:
    ICPOptimizer() : m_nIterations{20}
    {
    }

    void setNbOfIterations(unsigned nIterations)
    {
        m_nIterations = nIterations;
    }

    virtual void estimatePose(const PointCloud &source, const PointCloud &target, Matrix4f &initialPose) = 0;

protected:
    std::vector<Vector3f> transformPoints(const std::vector<Vector3f> &sourcePoints, const Matrix4f &pose)
    {
        std::vector<Vector3f> transformedPoints;
        transformedPoints.reserve(sourcePoints.size());

        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);

        for (const auto &point : sourcePoints)
        {
            transformedPoints.push_back(rotation * point + translation);
        }

        return transformedPoints;
    }

    std::vector<Vector3f> transformNormals(const std::vector<Vector3f> &sourceNormals, const Matrix4f &pose)
    {
        std::vector<Vector3f> transformedNormals;
        transformedNormals.reserve(sourceNormals.size());

        const auto rotation = pose.block(0, 0, 3, 3);

        for (const auto &normal : sourceNormals)
        {
            transformedNormals.push_back(rotation.inverse().transpose() * normal);
        }

        return transformedNormals;
    }

    // TODO: Add matches to signature
    void pruneCorrespondences(const std::vector<Vector3f> &sourceNormals, const std::vector<Vector3f> &targetNormals)
    {
        const unsigned nPoints = sourceNormals.size();
    }
};

/**
 * ICP optimizer - using linear least-squares for optimization.
 */
class LinearICPOptimizer : public ICPOptimizer
{
public:
    LinearICPOptimizer() {}

    virtual void estimatePose(const PointCloud &source, const PointCloud &target, Matrix4f &initialPose) override
    {

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIterations; ++i)
        {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

            pruneCorrespondences(transformedNormals, target.getNormals());

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

            std::vector<Vector3f> sourcePoints;
            std::vector<Vector3f> targetPoints;

            // Add all matches to the sourcePoints and targetPoints vector,
            // so that the sourcePoints[i] matches targetPoints[i]. For every source point,
            // the matches vector holds the index of the matching target point.
            // TODO: Add matches
            for (int j = 0; j < transformedPoints.size(); j++)
            {
                /*
                const auto& match = matches[j];
                if (match.idx >= 0) {
                    sourcePoints.push_back(transformedPoints[j]);
                    targetPoints.push_back(target.getPoints()[match.idx]);
                }
                */
            }

            // Estimate the new pose
            estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals()) * estimatedPose;

            std::cout << "Optimization iteration done." << std::endl;
        }

        // Store result
        initialPose = estimatedPose;
    }

private:
    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals)
    {
        const unsigned nPoints = sourcePoints.size();

        // Build the system Ax = b
        MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
        VectorXf b = VectorXf::Zero(4 * nPoints);

        for (unsigned i = 0; i < nPoints; i++)
        {
            const auto &s = sourcePoints[i];
            const auto &d = targetPoints[i];
            const auto &n = targetNormals[i];

            // TODO: Add the point-to-plane constraints to the system
            // a's
            A(i * 4, 0) = n.z() * s.y() - n.y() * s.z();
            A(i * 4, 1) = n.x() * s.z() - n.z() * s.x();
            A(i * 4, 2) = n.y() * s.x() - n.x() * s.y();
            // n's
            A(i * 4, 3) = n.x();
            A(i * 4, 4) = n.y();
            A(i * 4, 5) = n.z();
            // b
            b(i * 4) = n.x() * d.x() + n.y() * d.y() + n.z() * d.z() - n.x() * s.x() - n.y() * s.y() - n.z() * s.z();

            // TODO: Add the point-to-point constraints to the system
            // Translation part is identity
            A(i * 4 + 1, 0) = 0;
            A(i * 4 + 1, 1) = s.z();
            A(i * 4 + 1, 2) = -s.y();
            A(i * 4 + 2, 0) = -s.z();
            A(i * 4 + 2, 1) = 0;
            A(i * 4 + 2, 2) = s.x();
            A(i * 4 + 3, 0) = s.y();
            A(i * 4 + 3, 1) = -s.x();
            A(i * 4 + 3, 2) = 0;
            A.block<3, 3>(i * 4 + 1, 3).setIdentity();
            // b TODO: Check whether sign is in correct orientation
            b(i * 4 + 1) = d.x() - s.x();
            b(i * 4 + 2) = d.y() - s.y();
            b(i * 4 + 3) = d.z() - s.z();
            // TODO: Optionally, apply a higher weight to point-to-plane correspondences
        }

        // TODO: Solve the system
        VectorXf x(6);
        BDCSVD<MatrixXf> svd = BDCSVD<MatrixXf>(A, ComputeThinU | ComputeThinV);
        x = svd.solve(b);

        float alpha = x(0), beta = x(1), gamma = x(2);

        // Build the pose matrix
        Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                            AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                            AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

        Vector3f translation = x.tail(3);

        // TODO: Build the pose matrix using the rotation and translation matrices
        Matrix4f estimatedPose = Matrix4f::Identity();
        estimatedPose.block<3, 3>(0, 0) = rotation;

        estimatedPose.block<3, 1>(0, 3) = translation;
        std::cout << estimatedPose << std::endl;
        return estimatedPose;
    }
};
