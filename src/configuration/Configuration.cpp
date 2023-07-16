#include "Configuration.h"
#include <filesystem>
#include <iostream>


std::string Configuration::getDataSetPath()
{
	std::string dataSetPath;
	if (USERNAME == std::string("dominik"))
	{
		dataSetPath = "../../../cv-data/RGBD-Dataset-Freiburg1/";
	}
	if (USERNAME == std::string("helga"))
	{
		dataSetPath = "../../../Data/rgbd_dataset_freiburg1_xyz/";
	}

	if (USERNAME == std::string("lisa"))
	{
		dataSetPath = "/home/lisa/cv_datasets/rgbd_dataset_freiburg1_xyz/";
	}
	return dataSetPath;
}

std::string Configuration::getOutputDirectory()
{
	std::string outputDirectory;
	if (USERNAME == std::string("dominik"))
	{
		outputDirectory = "./Output/";
	}
	if (USERNAME == std::string("helga"))
	{
		outputDirectory = "../Output/";
	}
		if (USERNAME == std::string("lisa"))
	{
		outputDirectory = "./Output/";
	}
	return outputDirectory;
}
