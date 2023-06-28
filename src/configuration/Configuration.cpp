#include "Configuration.h"

std::string Configuration::getDataSetPath()
{
    std::string dataSetPath;
    if (USERNAME == "dominik")
    {
        dataSetPath = "../../cv-data/RGBD-Dataset-Freiburg1/";
    }

    return dataSetPath;
}

std::string Configuration::getOutputDirectory()
{
    std::string outputDirectory;
    if (USERNAME == "dominik")
    {
        outputDirectory = "./Output/";
    }

    return outputDirectory;
}