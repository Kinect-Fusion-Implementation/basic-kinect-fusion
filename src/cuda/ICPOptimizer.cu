#include "ICPOptimizer.h"
#include <iostream>
#include <chrono>

// TODO: Implement PointToPoint ICP with given corresponances
// TODO: Implement other correspondance search method (No second downsampling)
// TODO: Implement Symmetric ICP

__host__ ICPOptimizer::ICPOptimizer(Matrix3f intrinsics, unsigned int width, unsigned int height, float vertex_diff_threshold, float normal_diff_threshold, std::vector<int> &iterations_per_level, float pointToPointWeight)
{
    // Initialized with Camera Matrix and threshold values. These should stay the same for all iterations and frames
    m_intrinsics = intrinsics;
    m_width = width;
    m_height = height;

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
    // Initialize frame transformation with identity matrix
    Matrix4f currentToPreviousFrame = Matrix4f::Identity();
    // Iterate over levels for pointClouds the higher the level, the smaller the resolution (level 0 contains original resolution)
    for (int level = currentFramePyramid.getPointClouds().size() - 1; level >= 0; level--)
    {
        for (unsigned int iteration = 0; iteration < m_iterations_per_level[level]; iteration++)
        {
            // TODO: Should this be always pointcloud 0 or point cloud level?
            Matrix4f inc = pointToPointAndPlaneICP(currentFramePyramid.getPointClouds().at(level).getPoints(), currentFramePyramid.getPointClouds().at(level).getNormals(),
                                                   raycastVertexMap, raycastNormalMap, currentToPreviousFrame, prevFrameToGlobal, level, iteration);
            /*
            // TODO: Check if enough correspondances were
            std::cout << "Incremental Matrix: " << std::endl
                      << inc << std::endl;
            std::cout << "Incremental Matrix det: " << std::endl
                      << inc.determinant() << std::endl;
            std::cout << "Incremental Matrix norm: " << std::endl
                      << inc.norm() << std::endl;
            */
            currentToPreviousFrame = inc * currentToPreviousFrame;
            /*
            std::cout << "Current to Previous Frame det:" << std::endl
                      << currentToPreviousFrame.determinant() << std::endl;
            std::cout << "Current to Previous Frame: " << std::endl
                      << currentToPreviousFrame << std::endl;
            */
        }
    }
    // Current Frame -> Global (Pose matrix)
    return prevFrameToGlobal * currentToPreviousFrame;
}

__device__ bool isFinite(Vector3f vector)
{
    return isfinite(vector.x()) && isfinite(vector.y()) && isfinite(vector.z());
}

/**
 * Computes the correspondences for the currentFrame vertices
 * width and height have to be matched the original image size, as we project into the original image
 */
__global__ void computeCorrespondencesAndSystemKernel(Vector3f *currentFrameVertices, Vector3f *currentFrameNormals, Vector3f *raycastVertexMap, Vector3f *raycastNormalMap,
                                                      Vector3f *matchedVertexMap, Vector3f *matchedNormalMap,
                                                      Matrix3f intrinsics, const Matrix4f currentFrameToPrevFrameTransformation, const Matrix4f prevFrameToGlobalTransform,
                                                      unsigned int width, unsigned int height, float vertex_diff_threshold, float normal_diff_threshold, float minf,
                                                      Eigen::Matrix<float, 6, 6> *matrices, Eigen::Matrix<float, 6, 1> *vectors, float pointToPointWeight, unsigned int level)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > (width >> level) * (height >> level))
    {
        printf("returning for: idx= %i\n", idx);
        return;
    }
    Vector3f vertex = currentFrameVertices[idx];
    Vector3f normal = currentFrameNormals[idx];
    if (isFinite(vertex) && isFinite(normal))
    {

        // Find corresponding vertex in previous frame
        // First transform the vertex into the previous frame
        Vector3f transformedCurrentVertex = (currentFrameToPrevFrameTransformation.block<3, 3>(0, 0) * vertex) + currentFrameToPrevFrameTransformation.block<3, 1>(0, 3);
        // Project transformed Vertex into the cameras screen space
        Vector3f homogeneousScreenSpace = intrinsics * transformedCurrentVertex;
        Vector2f normalizedSceenSpaceCoordinates = Vector2f(homogeneousScreenSpace.x() / homogeneousScreenSpace.z(), homogeneousScreenSpace.y() / homogeneousScreenSpace.z());

        // Check if transformedCurrentVertex is in the image
        if (0 <= normalizedSceenSpaceCoordinates.x() && int(normalizedSceenSpaceCoordinates.x()) < width && 0 <= normalizedSceenSpaceCoordinates.y() && int(normalizedSceenSpaceCoordinates.y()) < height)
        {
            unsigned int pixelCoordinates = int(normalizedSceenSpaceCoordinates.x()) + int(normalizedSceenSpaceCoordinates.y()) * width;
            Vector3f matchedVertex = raycastVertexMap[pixelCoordinates];
            Vector3f matchedNormal = raycastNormalMap[pixelCoordinates];

            if (isFinite(matchedVertex) && isFinite(matchedNormal))
            {
                if ((transformedCurrentVertex - matchedVertex).norm() < vertex_diff_threshold)
                {
                    Matrix3f rotation = (currentFrameToPrevFrameTransformation).block<3, 3>(0, 0);
                    if ((1 - matchedVertex.dot(rotation * normal)) < normal_diff_threshold)
                    {
                        // We found a match!
                        matchedVertexMap[idx] = matchedVertex;
                        matchedNormalMap[idx] = matchedNormal;

                        if (pointToPointWeight > 0)
                        {
                            buildPointToPointErrorSystem(idx, transformedCurrentVertex, matchedVertex, matchedNormal, matrices, vectors, pointToPointWeight);
                        }
                        // Build summand point to plane matrix and vector for current correspondance
                        buildPointToPlaneErrorSystem(idx, transformedCurrentVertex, matchedVertex, matchedNormal, matrices, vectors, pointToPointWeight);

                        return;
                    }
                }
            }
        }
    }
    // If no correspondence was found, set correspondence invalid and the matrix and vector of the system to 0 (neutral element)
    matchedVertexMap[idx] = Vector3f(minf, minf, minf);
    matchedNormalMap[idx] = Vector3f(minf, minf, minf);
    matrices[idx] = Matrix<float, 6, 6>::Zero();
    vectors[idx] = Matrix<float, 6, 1>::Zero();
}

/**
 * Reduces an even number of elements (2 * #threads) per block
 * As every thread requires one shared memory entry, the size of the shared memory has to be equivalent to #threads of a block
 * Each block with id = i reduces the elements [i * n,..., (i+1) * n - 1]
 * For every block, one remaining summand is written into the summedConstraint global Device memory
 */
__global__ void reduce(Eigen::Matrix<float, 6, 6> *constraints, Eigen::Matrix<float, 6, 6> *summedConstraint)
{
    extern __shared__ Eigen::Matrix<float, 6, 6> matrices[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    matrices[tid] = constraints[i] + constraints[i + blockDim.x];

    __syncthreads();
    // Tracks the number of summands that are remaining to be summed up in total before the reduction of the current loop iteration
    unsigned int remainingSummands = blockDim.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            matrices[tid] += matrices[tid + s];
        }
        // In case of an uneven number of elements, the last thread also adds up this last element
        if (tid == s - 1 && remainingSummands % 2 == 1)
        {
            matrices[tid] += matrices[tid + s + 1];
        }
        remainingSummands = s;
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
    {
        summedConstraint[blockIdx.x] = matrices[0];
    }
}

/**
 * Reduces per block a total of blockSize * 2 many elements
 */
__global__ void reduce(Eigen::Matrix<float, 6, 1> *constraints, Eigen::Matrix<float, 6, 1> *summedConstraint)
{
    extern __shared__ Eigen::Matrix<float, 6, 1> vectors[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    vectors[tid] = constraints[i] + constraints[i + blockDim.x];

    __syncthreads();
    // Tracks the number of summands that are remaining to be summed up in total before the reduction of the current loop iteration
    unsigned int remainingSummands = blockDim.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            vectors[tid] += vectors[tid + s];
        }
        // In case of an uneven number of elements, the last thread also adds up this last element
        if (tid == s - 1 && remainingSummands % 2 == 1)
        {
            vectors[tid] += vectors[tid + s + 1];
        }
        remainingSummands = s;
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        summedConstraint[blockIdx.x] = vectors[0];
}

/**
 * Correspondance Search
 * returns: matched vertices and normals for every point stored on DEVICE
 *          If a point got no correspondence, the corresponding entry in the correspondence maps is MINF vector
 */
__host__ Matrix4f ICPOptimizer::pointToPointAndPlaneICP(Vector3f *currentFrameVertices, Vector3f *currentFrameNormals, Vector3f *raycastVertexMap, Vector3f *raycastNormalMap,
                                                        const Matrix4f &currentFrameToPrevFrameTransformation, const Matrix4f &prevFrameToGlobalTransform,
                                                        unsigned int level, unsigned int iteration)
{
    unsigned int numberPoints = (m_width >> level) * (m_height >> level);
    dim3 threadBlocks(256);
    dim3 blocks(numberPoints / 256);
    Vector3f *matchedVertexMap;
    Vector3f *matchedNormalMap;
    Eigen::Matrix<float, 6, 6> *matrices;
    Eigen::Matrix<float, 6, 1> *vectors;
    cudaMalloc(&matchedVertexMap, numberPoints * sizeof(Vector3f));
    cudaMalloc(&matchedNormalMap, numberPoints * sizeof(Vector3f));
    cudaMalloc(&matrices, sizeof(Eigen::Matrix<float, 6, 6>) * numberPoints);
    cudaMalloc(&vectors, sizeof(Eigen::Matrix<float, 6, 1>) * numberPoints);

    computeCorrespondencesAndSystemKernel<<<blocks, threadBlocks>>>(currentFrameVertices, currentFrameNormals, raycastVertexMap, raycastNormalMap,
                                                                    matchedVertexMap, matchedNormalMap,
                                                                    m_intrinsics, currentFrameToPrevFrameTransformation, prevFrameToGlobalTransform,
                                                                    m_width, m_height, m_vertex_diff_threshold, m_normal_diff_threshold, MINF,
                                                                    matrices, vectors, m_pointToPointWeight, level);
    cudaFree(matchedVertexMap);
    cudaFree(matchedNormalMap);

    Eigen::Matrix<float, 6, 6> *matrixSum;
    Eigen::Matrix<float, 6, 1> *vectorSum;

    unsigned int numberThreads = 64;
    unsigned int numberBlocks = numberPoints / (numberThreads * 2);

    Matrix<float, 6, 6> *sumMatricesGPU;
    cudaMalloc(&sumMatricesGPU, sizeof(Matrix<float, 6, 6>) * numberBlocks);
    Matrix<float, 6, 1> *sumVectorsGPU;
    cudaMalloc(&sumVectorsGPU, sizeof(Matrix<float, 6, 1>) * numberBlocks);
    reduce<<<numberBlocks, numberThreads, numberThreads * sizeof(Matrix<float, 6, 6>)>>>(matrices, sumMatricesGPU);
    reduce<<<numberBlocks, numberThreads, numberThreads * sizeof(Matrix<float, 6, 1>)>>>(vectors, sumVectorsGPU);

    Matrix<float, 6, 6> *matrixGPU;
    Matrix<float, 6, 6> *vectorGPU;
    if (numberBlocks < 512)
    {
        numberThreads = numberBlocks / 2;
        reduce<<<1, numberThreads, numberThreads * sizeof(Matrix<float, 6, 6>)>>>(sumMatricesGPU, sumMatricesGPU);
        reduce<<<1, numberThreads, numberThreads * sizeof(Matrix<float, 6, 1>)>>>(sumVectorsGPU, sumVectorsGPU);
    }
    else
    {
        numberThreads = ((numberBlocks / 2) / 30);
        numberBlocks = 30;
        reduce<<<numberBlocks, numberThreads, numberThreads * sizeof(Matrix<float, 6, 6>)>>>(sumMatricesGPU, sumMatricesGPU);
        reduce<<<numberBlocks, numberThreads, numberThreads * sizeof(Matrix<float, 6, 1>)>>>(sumVectorsGPU, sumVectorsGPU);
        numberThreads = numberBlocks / 2;
        reduce<<<1, numberThreads, numberThreads * sizeof(Matrix<float, 6, 6>)>>>(sumMatricesGPU, sumMatricesGPU);
        reduce<<<1, numberThreads, numberThreads * sizeof(Matrix<float, 6, 1>)>>>(sumVectorsGPU, sumVectorsGPU);
    }
    Eigen::Matrix<float, 6, 6> designMatrix;
    Eigen::Matrix<float, 6, 1> designVector;
    cudaMemcpy(&designMatrix, sumMatricesGPU, sizeof(Matrix<float, 6, 6>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&designVector, sumVectorsGPU, sizeof(Matrix<float, 6, 1>), cudaMemcpyDeviceToHost);
    cudaFree(sumMatricesGPU);
    cudaFree(sumVectorsGPU);

    std::cout << "Design matrix: " << designMatrix << std::endl;
    std::cout << "Design vector: " << designVector << std::endl;

    // solution -> (beta, gamma, alpha, tx, ty, tz)
    Eigen::Matrix<float, 6, 1> solution = designMatrix.llt().solve(designVector);
    Matrix4f output;
    output << 1, solution(2), -solution(1), solution(3),
        -solution(2), 1, solution(0), solution(4),
        solution(1), -solution(0), 1, solution(5),
        0, 0, 0, 1;
    cudaFree(matrixSum);
    cudaFree(vectorSum);
    return output;
}

__device__ void buildPointToPlaneErrorSystem(unsigned int idx, Vector3f &currentVertex,
                                             Vector3f &matchedVertex, Vector3f &matchedNormal,
                                             Eigen::Matrix<float, 6, 6> *matrices, Eigen::Matrix<float, 6, 1> *vectors,
                                             float pointToPointWeight)
{
    // Returns the solution of the system of equations for point to plane ICP for one summand of the cost function
    // currentVertex -> V_k, TargetVertex -> V_k-1, TargetNormal -> N_k-1

    // solution   -> (beta, gamma, alpha, tx, ty, tz)
    // G contains  the skew-symmetric matrix form of the currentVertex | FIXME: Add .cross to calcualte skew-symmetric matrix
    // For vector -> (beta, gamma, alpha, tx, ty, tz) the skew-symmetric matrix form is:
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