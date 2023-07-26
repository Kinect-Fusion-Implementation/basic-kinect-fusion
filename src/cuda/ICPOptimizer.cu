#include "ICPOptimizer.h"

// TODO: Implement PointToPoint ICP with given corresponances
// TODO: Implement other correspondance search method (No second downsampling)
// TODO: Implement Symmetric ICP

__host__ ICPOptimizer::ICPOptimizer(Matrix3f intrinsics, unsigned int width, unsigned int height, float vertex_diff_threshold, float normal_diff_threshold, std::vector<int> &iterations_per_level, float pointToPointWeight = 0.5)
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
            std::tuple<Vector3f*, Vector3f*> correspondences = findCorrespondences(sourcePointClouds[i], raycastVertexMap, raycastNormalMap, currentToPreviousFrame, prevFrameToGlobal);
            // TODO: Check if enough correspondances were 
            
            Matrix4f inc = pointToPointAndPlaneICP(correspondences, prevFrameToGlobal, currentToPreviousFrame);
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

__global__ void computeCorrespondencesKernel(Vector3f *currentFrameVertices, Vector3f *currentFrameNormals, Vector3f *vertexMap, Vector3f *normalMap,
                                             Vector3f *matchVertexMap, Vector3f *matchNormalMap,
                                             Matrix3f intrinsics, const Matrix4f currentFrameToPrevFrameTransformation, const Matrix4f prevFrameToGlobalTransform,
                                             unsigned int width, unsigned int height, float vertex_diff_threshold, float normal_diff_threshold, float minf)
{
    std::vector<std::tuple<Vector3f, Vector3f, Vector3f, Vector3f>> correspondences;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > width * height)
    {
        printf("returning for: idx= %i\n", idx);
        return;
    }
    Vector3f vertex = currentFrameVertices[idx];
    Vector3f normal = currentFrameNormals[idx];
    if (vertex.allFinite() && normal.allFinite())
    {
        // Find corresponding vertex in previous frame
        Vector3f transformedCurrentVertex = (currentFrameToPrevFrameTransformation.block<3, 3>(0, 0) * vertex) + currentFrameToPrevFrameTransformation.block<3, 1>(0, 3);
        Vector2f indexedCurrentVertex = dehomogenize_2d(intrinsics * transformedCurrentVertex);

        // Check if transformedCurrentVertex is in the image
        if (0 <= indexedCurrentVertex[0] && int(indexedCurrentVertex[0]) < width && 0 <= indexedCurrentVertex[1] && int(indexedCurrentVertex[1]) < height)
        {
            unsigned int pixelCoordinates = int(indexedCurrentVertex[0]) + int(indexedCurrentVertex[1]) * width;
            Vector3f matchedVertex = vertexMap[pixelCoordinates];
            Vector3f matchedNormal = normalMap[pixelCoordinates];

            if (matchedVertex.allFinite() && matchedNormal.allFinite())
            {
                if ((transformedCurrentVertex - matchedVertex).norm() < vertex_diff_threshold)
                {
                    Matrix3f rotation = (currentFrameToPrevFrameTransformation).block<3, 3>(0, 0);
                    if ((1 - matchedVertex.dot(rotation * normal)) < normal_diff_threshold)
                    {
                        // We found a match!
                        matchVertexMap[idx] = matchedVertex;
                        matchNormalMap[idx] = matchedNormal;
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
std::tuple<Vector3f *, Vector3f *> ICPOptimizer::findCorrespondences(PointCloud &currentPointCloud, Vector3f *vertexMap, Vector3f *normalMap, const Matrix4f &currentFrameToPrevFrameTransformation, const Matrix4f &prevFrameToGlobalTransform)
{
    std::vector<std::tuple<Vector3f, Vector3f, Vector3f, Vector3f>> correspondences;
    unsigned int numberPoints = m_width * m_height;
    dim3 threadBlocks(20);
    dim3 blocks(numberPoints / 20);
    Vector3f *currentFrameVertices = currentPointCloud.getPoints();
    Vector3f *currentFrameNormals = currentPointCloud.getNormals();
    Vector3f *matchVertexMap;
    Vector3f *matchNormalMap;
    cudaMalloc(&matchVertexMap, numberPoints * sizeof(Vector3f));
    cudaMalloc(&matchNormalMap, numberPoints * sizeof(Vector3f));
    computeCorrespondencesKernel<<<blocks, threadBlocks>>>(currentFrameVertices, currentFrameNormals, vertexMap, normalMap,
                                                           matchVertexMap, matchNormalMap,
                                                           m_intrinsics, currentFrameToPrevFrameTransformation, prevFrameToGlobalTransform,
                                                           m_width, m_height, m_vertex_diff_threshold, m_normal_diff_threshold, MINF);
    return std::make_tuple(matchVertexMap, matchNormalMap);
}

// ---- Point to Plane ICP ----

Matrix4f ICPOptimizer::pointToPointAndPlaneICP(const std::vector<std::tuple<Vector3f, Vector3f, Vector3f, Vector3f>> &correspondences, const Matrix4f &globalToPreviousFrame, const Matrix4f &currentToPreviousFrame)
{
    // designMatrix contains sum of all A_t * A matrices
    Eigen::Matrix<float, 6, 6> designMatrix = Eigen::Matrix<float, 6, 6>::Zero();

    // designVector contains sum of all A_t * b vectors
    Eigen::Matrix<float, 6, 1> designVector = Eigen::Matrix<float, 6, 1>::Zero();

    // --------- CUDAAAAAAAAAAA -----------
    for (unsigned int i = 0; i < correspondences.size(); i++)
    {
        // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1
        Vector3f currVertex = currentToPreviousFrame.block<3, 3>(0, 0) * std::get<0>(correspondences[i]) + currentToPreviousFrame.block<3, 1>(0, 3);
        Vector3f prevVertex = std::get<2>(correspondences[i]);
        Vector3f prevNormal = std::get<3>(correspondences[i]);

        // Construct Linear System to solve
        // Build summand point to point matrix and vector for current correspondance
        if (m_pointToPointWeight > 0)
        {
            std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> pointSystem = buildPointToPointErrorSystem(
                currVertex, prevVertex, prevNormal);
            designMatrix += m_pointToPointWeight * std::get<0>(pointSystem);
            designVector += m_pointToPointWeight * std::get<1>(pointSystem);
        }

        // Build summand point to plane matrix and vector for current correspondance
        std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> planeSystem = buildPointToPlaneErrorSystem(
            currVertex, prevVertex, prevNormal);
        designMatrix += (1 - m_pointToPointWeight) * std::get<0>(planeSystem);
        designVector += (1 - m_pointToPointWeight) * std::get<1>(planeSystem);
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

std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> ICPOptimizer::buildPointToPlaneErrorSystem(Vector3f &sourceVertex, Vector3f &targetVertex, Vector3f &targetNormal)
{
    // Returns the solution of the system of equations for point to plane ICP for one summand of the cost function
    // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1

    // solution -> (beta, gamma, alpha, tx, ty, tz)
    // G contains  the skew-symmetric matrix form of the sourceVertex | FIXME: Add .cross to calcualte skew-symmetric matrix
    // For vector (beta, gamma, alpha, tx, ty, tz) the skew-symmetric matrix form is:
    Eigen::Matrix<float, 3, 6> G;
    G << 0, -sourceVertex(2), sourceVertex(1), 1, 0, 0,
        sourceVertex(2), 0, -sourceVertex(0), 0, 1, 0,
        -sourceVertex(1), sourceVertex(0), 0, 0, 0, 1;

    // A contains the dot product of the skew-symmetric matrix form of the sourceVertex and the targetNormal and is the matrix we are optimizing over
    Eigen::Matrix<float, 6, 1> A_t = G.transpose() * targetNormal;
    Eigen::Matrix<float, 6, 6> A_tA = A_t * A_t.transpose();
    // b contains the dot product of the targetNormal and the difference between the targetVertex and the sourceVertex
    Eigen::Matrix<float, 6, 1> b = A_t * (targetNormal.transpose() * (targetVertex - sourceVertex));

    return std::make_tuple(A_tA, b);
}

std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> ICPOptimizer::buildPointToPointErrorSystem(Vector3f &sourceVertex, Vector3f &targetVertex, Vector3f &targetNormal)
{
    // Returns the solution of the system of equations for point to point ICP for one summand of the cost function
    // SourceVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1

    // solution -> (beta, gamma, alpha, tx, ty, tz)
    // G contains  the skew-symmetric matrix form of the sourceVertex | FIXME: Add .cross to calcualte skew-symmetric matrix
    // For vector (beta, gamma, alpha, tx, ty, tz) the skew-symmetric matrix form is:
    Eigen::Matrix<float, 3, 6> G;
    G << 0, -sourceVertex(2), sourceVertex(1), 1, 0, 0,
        sourceVertex(2), 0, -sourceVertex(0), 0, 1, 0,
        -sourceVertex(1), sourceVertex(0), 0, 0, 0, 1;

    // A contains the dot product of the skew-symmetric matrix form of the sourceVertex and the targetNormal and is the matrix we are optimizing over
    Eigen::Matrix<float, 6, 3> A_t = G.transpose();
    Eigen::Matrix<float, 6, 6> A_tA = A_t * A_t.transpose();
    // b contains the dot product of the targetNormal and the difference between the targetVertex and the sourceVertex
    Eigen::Matrix<float, 6, 1> b = A_t * (targetVertex - sourceVertex);

    return std::make_tuple(A_tA, b);
}

// Helper methods for homogenization and dehomogenization in 2D and 3D

Vector3f homogenize_2d(const Vector2f &point)
{
    return Vector3f(point[0], point[1], 1.0f);
}

Vector2f dehomogenize_2d(const Vector3f &point)
{
    return Vector2f(point[0] / point[2], point[1] / point[2]);
}