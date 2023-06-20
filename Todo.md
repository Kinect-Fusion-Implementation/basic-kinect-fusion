# Todo List

Todo steps in our project

## Implementation steps

### Implement Volumetric Integration
Expected Input: Pose matrix, depth map from our new frame, current TSDF voxel volume
Expected Output: updated TSDF voxel volume, truncation value mu
Method:
1. Iterate over every voxel
2. Transform into World frame
3. Transform into Camera frame
4. Compute pixel coordinates
5. Check whether point is in frustum (should be in front of the camera (positive z coordinate in camera frame) and should land on image sensor)
6. Calculate truncated SDF value
7. update averaged tsdf

### Test the volumetric Integration
1. Add timing steps how long each voxel step takes
2. Add timing steps how long each complete volume update takes

### ICP
Expected Input: Previous Pose/Extrinsic matrix, Intrinsics Matrix, Raycasted Vertex and Normal map, Current Vertices Pyramid, Current Normal Map Pyramid, 
Expected Output: Pose of our new matrix 
Method:
1. Projective Data Association
2. Compute Linearized ICP on each pyramid level iteratively

### Testing
1. Compare Pose with GT pose (from data set)

### Surface Predictions - Raytracing
Expected Input: 
Expected Output:

### Cuda Programming

Parallelize all methods

## Debug

- Debug smoothing of depth map
- Debug subsampling of depth map