cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(kinect_fusion)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/Glog/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")

# Use pre-compiled libraries on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
endif(WIN32)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(Eigen3_DIR /opt/homebrew/Cellar/eigen/3.4.0_1/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
    set(glog_DIR /opt/homebrew/Cellar/glog/0.6.0/lib/cmake/glog CACHE PATH "Path to installed glog")
    set(Ceres_DIR /opt/homebrew/Cellar/ceres-solver/2.1.0_3/lib/cmake/Ceres CACHE PATH "Path to installed Ceres")
    #set(Flann_INCLUDE_DIR /opt/homebrew/Cellar/flann/1.9.2_1/ CACHE PATH "Path to Flann source folder")
    set(FreeImage_INCLUDE_DIR /opt/homebrew/Cellar/freeimage/3.18.0/include/ CACHE PATH "Homebrew Path header")
    set(FreeImage_LIBRARY_DIR /opt/homebrew/Cellar/freeimage/3.18.0/lib/ CACHE PATH "Homebrew Path lib")
endif()

# Set C++ flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(EVAL_MODE ON CACHE BOOL "Enables time measurement code and prints")
add_compile_definitions(EVAL_MODE=${EVAL_MODE})

#find_package(nanoflann REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(SourceCode_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "Path to source code")
set(CoreCode_DIR kinect_fusion CACHE PATH "Directory of the kinect fusion core code") 
set(Configuration_DIR configuration CACHE PATH "Directory for configuration files") 
set(Sensor_DIR sensor CACHE PATH "Directory for configuration files") 
set(Visualization_DIR visualization CACHE PATH "Directory for configuration files") 

set(HEADER_FILES 
    ${SourceCode_DIR}/${CoreCode_DIR}/Eigen.h 
    ${SourceCode_DIR}/${CoreCode_DIR}/PointCloud.h 
    ${SourceCode_DIR}/${CoreCode_DIR}/VoxelGrid.h
    ${SourceCode_DIR}/${CoreCode_DIR}/ICPOptimizer.h 
    ${SourceCode_DIR}/${Sensor_DIR}/VirtualSensor.h
    ${SourceCode_DIR}/${Visualization_DIR}/FreeImageHelper.h
    ${SourceCode_DIR}/${Visualization_DIR}/MarchingCubes.h
    ${SourceCode_DIR}/${Visualization_DIR}/SimpleMesh.h
    ${SourceCode_DIR}/${Configuration_DIR}/Configuration.h
)

set(SOURCE_FILES 
    ${SourceCode_DIR}/${CoreCode_DIR}/VoxelGrid.cpp
    ${SourceCode_DIR}/${Visualization_DIR}/FreeImageHelper.cpp
    ${SourceCode_DIR}/${Configuration_DIR}/Configuration.cpp
)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(kinect_fusion ${SourceCode_DIR}/main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(kinect_fusion ceres freeimage Eigen3::Eigen)
target_include_directories(kinect_fusion PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR})

# On Windows copy dll to executable folder
if(WIN32)
    # Visual Studio properties
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT exercise_5)
    set_property(TARGET kinect_fusion PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

    # On Windows copy dll to executable folder
    add_custom_command(TARGET kinect_fusion POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET kinect_fusion POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different          
            "${GLOG_DLL_PATH_DEBUG}"                           
            ${CMAKE_BINARY_DIR})                     
    
    add_custom_command(TARGET kinect_fusion POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLOG_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
endif(WIN32)
