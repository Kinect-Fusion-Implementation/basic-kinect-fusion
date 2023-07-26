#pragma once
#include <string>

#ifndef USERNAME
#define USERNAME std::string("dominik")
#endif

namespace Configuration
{
    std::string getDataSetPath();

    std::string getOutputDirectory();
}