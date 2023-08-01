# Kinect Fusion

Our method is completely implemented in the `main` function in the `main.cpp` file.

## Setup

To run our method, one has to set the `dataSetPath` in the `Configuration::getDataSetPath()` method in the `Configuration.cpp` file.
There are also two flags that can be set within CMAKE:

 - `EVAL_MODE`: Builds the project with the performance measurement code
 - `SAVE_IMAGE_MODE`: Builds the project with the functionality to visualize and save the normal map of the raycast during every iteration as well as the final mesh after all frames. (Mesh of the complete TSDF)

Note that to use the `SAVE_IMAGE_MODE`, the `outputDirectory` in the `Configuration::getOutputDirectory()` method within the `Configuration.cpp` file has to point to an *existing* directory.

## Sensor

The sensor folder contains the virtual sensor that we used to load the datasets.

## Visualization

In the visualization folder we have multiple components to visualize various data types.
The `MarchingCubes.h` header file allows us to visualize the TSDF Voxel Grid as a mesh.
The PointCloudToMesh files contain the code to generating a mesh from the vertex map.
The FreeImageHelper files contain the code we used to visualize the normal and depth maps as images.

## Cuda

The Cuda folder contains every single kinect fusion component.

### PointCloudPyramid

The PointCloudPyramid files contain the code for smoothing and subsampling the kinect provided depth map and
constructing and managing the point clouds for the original and subsampled levels.
Puts the depth map from host to device memory.

### PointCloud

The PointCloud files contain the code to reconstruct the vertex and normal map from a given depth map.
Each level of the depth map pyramid results is transformed into one PointCloud instance.
The depth, vertex and normal map for every level is managed by the Point Cloud in the Device Memory.

### CudaVoxelGrid

The CudaVoxelGrid files realize the Voxel based TSDF. Each voxel is a struct and the whole voxel grid is stored sequentially in device memory.
It provides methods to update the TSDF given the camera pose and point cloud as well as raycasting the TSDF to retrieve a vertex and normal map from the implicit representation.

### ICPOptimizer
The ICPOptimizier files implement the projective Point-to-Point and Point-To-Plance linearized ICP. One kernel finds correspondences and computes the corresponding constraint matrix and vector. The other sums up all of these matrices and vectors on the GPU. Lastly we solve the linear system with SVD.

## Kinect_fusion

The kinect_fusion folder originally contained our kinect fusion components that worked on the CPU.
Since all of our components are now cudarized, this folder remained to make merging easier.
