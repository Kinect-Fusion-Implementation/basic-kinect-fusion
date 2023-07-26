#include "ICPOptimizer.h"
#include <iostream>

// TODO: Implement PointToPoint ICP with given corresponances
// TODO: Implement other correspondance search method (No second downsampling)
// TODO: Implement Symmetric ICP

__host__ ICPOptimizer::ICPOptimizer(Matrix3f intrinsics, unsigned int width, unsigned int height, float vertex_diff_threshold, float normal_diff_threshold, std::vector<int> &iterations_per_level, float pointToPointWeight)
{
    // Initialized with Camera Matrix and threshold values. These should stay the same for all iterations and frames
    m_intrinsics = intrinsics;

    // Sets threshold values for vertex and normal difference in correspondance search
    m_vertex_diff_threshold = vertex_diff_threshold;
    m_normal_diff_threshold = normal_diff_threshold;

    // Sets number of iterations per level
    m_iterations_per_level = iterations_per_level;
}

/**
 * We expect the vertecies of both pointclouds to have 3d coordinates with respect to the camera frame and not the global frame
 * source -> PointCloud of k-th frame, target -> PointCloud of k-1-th frame
 * Receives the pyramid and raycasted vertex and normal map as points to DEVICE storage
 */
__host__ Matrix4f ICPOptimizer::optimize(PointCloudPyramid &currentFramePyramid, Vector3f *raycastVertexMap, Vector3f *raycastNormalMap, const Matrix4f &prevFrameToGlobal)
{
    std::vector<PointCloud> sourcePointClouds = currentFramePyramid.getPointClouds();
    // Initialize frame transformation with identity matrix
    Matrix4f currentToPreviousFrame = Matrix4f::Identity();
    // Iterate over levels for pointClouds | We assume that levels match for both pyramids
    for (int i = sourcePointClouds.size() - 1; i >= 0; i--)
    {
        std::cout << "Level: " << i << std::endl;
        for (unsigned int k = 0; k < m_iterations_per_level[i]; k++)
        {
            std::cout << "Level: " << i << " Iteration: " << k << std::endl;
            // TODO: Should this be always pointcloud 0 or point cloud i?
            Matrix4f inc = pointToPointAndPlaneICP(sourcePointClouds[i], raycastVertexMap, raycastNormalMap, currentToPreviousFrame, prevFrameToGlobal);
            // TODO: Check if enough correspondances were
            std::cout << "Incremental Matrix: " << std::endl
                      << inc << std::endl;
            std::cout << "Incremental Matrix det: " << std::endl
                      << inc.determinant() << std::endl;
            std::cout << "Incremental Matrix norm: " << std::endl
                      << inc.norm() << std::endl;
            currentToPreviousFrame = inc * currentToPreviousFrame;
            std::cout << "Current to Previous Frame det:" << std::endl
                      << currentToPreviousFrame.determinant() << std::endl;
            std::cout << "Current to Previous Frame: " << std::endl
                      << currentToPreviousFrame << std::endl;
        }
    }
    return prevFrameToGlobal * currentToPreviousFrame;
}

__device__ bool isFinite(Vector3f vector)
{
    return isfinite(vector.x()) && isfinite(vector.y()) && isfinite(vector.z());
}

__global__ void computeCorrespondencesAndSystemKernel(Vector3f *currentFrameVertices, Vector3f *currentFrameNormals, Vector3f *vertexMap, Vector3f *normalMap,
                                                      Vector3f *matchVertexMap, Vector3f *matchNormalMap,
                                                      Matrix3f intrinsics, const Matrix4f currentFrameToPrevFrameTransformation, const Matrix4f prevFrameToGlobalTransform,
                                                      unsigned int width, unsigned int height, float vertex_diff_threshold, float normal_diff_threshold, float minf,
                                                      Eigen::Matrix<float, 6, 6> *matrices, Eigen::Matrix<float, 6, 1> *vectors, float pointToPointWeight)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > width * height)
    {
        printf("returning for: idx= %i\n", idx);
        return;
    }
    Vector3f vertex = currentFrameVertices[idx];
    Vector3f normal = currentFrameNormals[idx];
    if (isFinite(vertex) && isFinite(normal))
    {
        // Find corresponding vertex in previous frame
        Vector3f transformedCurrentVertex = (currentFrameToPrevFrameTransformation.block<3, 3>(0, 0) * vertex) + currentFrameToPrevFrameTransformation.block<3, 1>(0, 3);
        Vector3f homogeneousScreenSpace = intrinsics * transformedCurrentVertex;
        Vector2f indexedCurrentVertex = Vector2f(homogeneousScreenSpace.x() / homogeneousScreenSpace.z(), homogeneousScreenSpace.y() / homogeneousScreenSpace.z());

        // Check if transformedCurrentVertex is in the image
        if (0 <= indexedCurrentVertex[0] && int(indexedCurrentVertex[0]) < width && 0 <= indexedCurrentVertex[1] && int(indexedCurrentVertex[1]) < height)
        {
            unsigned int pixelCoordinates = int(indexedCurrentVertex[0]) + int(indexedCurrentVertex[1]) * width;
            Vector3f matchedVertex = vertexMap[pixelCoordinates];
            Vector3f matchedNormal = normalMap[pixelCoordinates];

            if (isFinite(matchedVertex) && isFinite(matchedNormal))
            {
                if ((transformedCurrentVertex - matchedVertex).norm() < vertex_diff_threshold)
                {
                    Matrix3f rotation = (currentFrameToPrevFrameTransformation).block<3, 3>(0, 0);
                    if ((1 - matchedVertex.dot(rotation * normal)) < normal_diff_threshold)
                    {
                        // We found a match!
                        matchVertexMap[idx] = matchedVertex;
                        matchNormalMap[idx] = matchedNormal;

                        //
                        if (pointToPointWeight > 0)
                        {
                            buildPointToPointErrorSystem(idx, vertex, matchedVertex, matchedNormal, matrices, vectors, pointToPointWeight);
                        }
                        // Build summand point to plane matrix and vector for current correspondance
                        buildPointToPlaneErrorSystem(idx, vertex, matchedVertex, matchedNormal, matrices, vectors, pointToPointWeight);

                        return;
                    }
                }
            }
        }
    }
    matchVertexMap[idx] = Vector3f(minf, minf, minf);
    matchNormalMap[idx] = Vector3f(minf, minf, minf);
}

// ---- Correspondance Search ----
/**
 * Correspondance Search
 * returns: matched vertices and normals for every point stored on DEVICE
 *          If a point got no correspondence, the corresponding entry in the correspondence maps is MINF vector
 */
__host__ Matrix4f ICPOptimizer::pointToPointAndPlaneICP(PointCloud &currentPointCloud, Vector3f *vertexMap, Vector3f *normalMap, const Matrix4f &currentFrameToPrevFrameTransformation, const Matrix4f &prevFrameToGlobalTransform)
{
    unsigned int numberPoints = m_width * m_height;
    dim3 threadBlocks(20);
    dim3 blocks(numberPoints / 20);
    Vector3f *currentFrameVertices = currentPointCloud.getPoints();
    Vector3f *currentFrameNormals = currentPointCloud.getNormals();
    Vector3f *matchVertexMap;
    Vector3f *matchNormalMap;
    Eigen::Matrix<float, 6, 6> *matrices;
    Eigen::Matrix<float, 6, 1> *vectors;
    cudaMalloc(&matchVertexMap, numberPoints * sizeof(Vector3f));
    cudaMalloc(&matchNormalMap, numberPoints * sizeof(Vector3f));
    cudaMalloc(&matrices, sizeof(Eigen::Matrix<float, 6, 6>) * numberPoints);
    cudaMalloc(&vectors, sizeof(Eigen::Matrix<float, 6, 1>) * numberPoints);
    computeCorrespondencesAndSystemKernel<<<blocks, threadBlocks>>>(currentFrameVertices, currentFrameNormals, vertexMap, normalMap,
                                                                    matchVertexMap, matchNormalMap,
                                                                    m_intrinsics, currentFrameToPrevFrameTransformation, prevFrameToGlobalTransform,
                                                                    m_width, m_height, m_vertex_diff_threshold, m_normal_diff_threshold, MINF,
                                                                    matrices, vectors, m_pointToPointWeight);
    Eigen::Matrix<float, 6, 6> *matricesCPU = new Eigen::Matrix<float, 6, 6>[numberPoints];
    Eigen::Matrix<float, 6, 1> *vectorsCPU = new Eigen::Matrix<float, 6, 1>[numberPoints];
    Vector3f *matchedVertices = new Vector3f[numberPoints];
    cudaMemcpy(matricesCPU, matrices, numberPoints * sizeof(Eigen::Matrix<float, 6, 6>), cudaMemcpyDeviceToHost);
    cudaMemcpy(vectorsCPU, vectors, numberPoints * sizeof(Eigen::Matrix<float, 6, 1>), cudaMemcpyDeviceToHost);
    cudaMemcpy(matchedVertices, matchVertexMap, numberPoints * sizeof(Vector3f), cudaMemcpyDeviceToHost);
    Eigen::Matrix<float, 6, 6> designMatrix = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> designVector = Eigen::Matrix<float, 6, 1>::Zero();
    for (size_t i = 0; i < numberPoints; i++)
    {
        // If we got a match, the matched point is not minf vector, and then the system is valid
        if (matchedVertices[i].allFinite())
        {
            designMatrix += matricesCPU[i];
            designVector += vectorsCPU[i];
        }
    }
    // solution -> (beta, gamma, alpha, tx, ty, tz)
    Eigen::Matrix<float, 6, 1> solution = (designMatrix.llt()).solve(designVector);

    Matrix4f output;
    output << 1, solution(2), -solution(1), solution(3),
        -solution(2), 1, solution(0), solution(4),
        solution(1), -solution(0), 1, solution(5),
        0, 0, 0, 1;
    return output;
}

__device__ void buildPointToPlaneErrorSystem(unsigned int idx, Vector3f &currentVertex,
                                             Vector3f &matchedVertex, Vector3f &matchedNormal,
                                             Eigen::Matrix<float, 6, 6> *matrices, Eigen::Matrix<float, 6, 1> *vectors,
                                             float pointToPointWeight)
{
    // Returns the solution of the system of equations for point to plane ICP for one summand of the cost function
    // currentVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1

    // solution -> (beta, gamma, alpha, tx, ty, tz)
    // G contains  the skew-symmetric matrix form of the currentVertex | FIXME: Add .cross to calcualte skew-symmetric matrix
    // For vector (beta, gamma, alpha, tx, ty, tz) the skew-symmetric matrix form is:
    Eigen::Matrix<float, 3, 6> G;
    G << 0, -currentVertex(2), currentVertex(1), 1, 0, 0,
        currentVertex(2), 0, -currentVertex(0), 0, 1, 0,
        -currentVertex(1), currentVertex(0), 0, 0, 0, 1;

    // A contains the dot product of the skew-symmetric matrix form of the currentVertex and the targetNormal and is the matrix we are optimizing over
    Eigen::Matrix<float, 6, 1> A_t = G.transpose() * matchedNormal;
    Eigen::Matrix<float, 6, 6> A_tA = A_t * A_t.transpose();
    // b contains the dot product of the targetNormal and the difference between the targetVertex and the currentVertex
    Eigen::Matrix<float, 6, 1> b = A_t * (matchedNormal.transpose() * (matchedVertex - currentVertex));
    matrices[idx] = (1 - pointToPointWeight) * A_tA;
    vectors[idx] = (1 - pointToPointWeight) * b;
}

__device__ void buildPointToPointErrorSystem(unsigned int idx, Vector3f &currentVertex,
                                             Vector3f &matchedVertex, Vector3f &matchedNormal,
                                             Eigen::Matrix<float, 6, 6> *matrices, Eigen::Matrix<float, 6, 1> *vectors,
                                             float pointToPointWeight)
{
    // Returns the solution of the system of equations for point to point ICP for one summand of the cost function
    // currentVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1

    // solution -> (beta, gamma, alpha, tx, ty, tz)
    // G contains  the skew-symmetric matrix form of the currentVertex | FIXME: Add .cross to calcualte skew-symmetric matrix
    // For vector (beta, gamma, alpha, tx, ty, tz) the skew-symmetric matrix form is:
    Eigen::Matrix<float, 3, 6> G;
    G << 0, -currentVertex(2), currentVertex(1), 1, 0, 0,
        currentVertex(2), 0, -currentVertex(0), 0, 1, 0,
        -currentVertex(1), currentVertex(0), 0, 0, 0, 1;

    // A contains the dot product of the skew-symmetric matrix form of the currentVertex and the targetNormal and is the matrix we are optimizing over
    Eigen::Matrix<float, 6, 3> A_t = G.transpose();
    Eigen::Matrix<float, 6, 6> A_tA = A_t * A_t.transpose();
    // b contains the dot product of the targetNormal and the difference between the targetVertex and the currentVertex
    Eigen::Matrix<float, 6, 1> b = A_t * (matchedVertex - currentVertex);
    matrices[idx] = pointToPointWeight * A_tA;
    vectors[idx] = pointToPointWeight * b;
}