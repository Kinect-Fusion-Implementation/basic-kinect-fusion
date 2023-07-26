set(CUDA_HEADER_FILES 
    ./Test.h
    ./CudaVoxelGrid.h
    ./PointCloud.h
    ./PointCloudPyramid.h
    ./ICPOptimizer.h
    )
set(CUDA_SOURCE_FILES 
   ./Test.cu
   ./CudaVoxelGrid.cu
   ./PointCloud.cu
   ./PointCloudPyramid.cu
   ./ICPOptimizer.cu
    )

# create static library of cuda files 
add_library(Kinect_CUDA STATIC ${CUDA_HEADER_FILES} ${CUDA_SOURCE_FILES})
target_link_libraries(Kinect_CUDA Eigen3::Eigen)
set_target_properties(Kinect_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(Kinect_CUDA PUBLIC ${EIGEN3_INCLUDE_DIR})