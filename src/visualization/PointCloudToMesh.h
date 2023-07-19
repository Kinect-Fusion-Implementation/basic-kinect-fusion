#include <iostream>
#include <fstream>
#include <array>

#include "../kinect_fusion/Eigen.h"
#include "../sensor/VirtualSensor.h"

bool valid(Vector3f v);
float distance(Vector3f a, Vector3f b);

bool writeMesh(Vector3f *vertices, unsigned int width, unsigned int height, const std::string &filename);