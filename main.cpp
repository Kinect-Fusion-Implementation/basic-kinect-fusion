#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"

int main() {
	int result = 0;
    std::string filenameIn = "../Data/rgbd_dataset_freiburg1_xyz/";
	VirtualSensor sensor;
	sensor.init(filenameIn);

    if (!sensor.Init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    while (sensor.processNextFrame()){
        float* depth = sensor.getDepth();
    }


	return result;
}
