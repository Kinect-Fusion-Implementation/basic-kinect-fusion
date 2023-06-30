#pragma once
#include <string>

#ifndef USERNAME
#define USERNAME "dominik"
#endif

namespace Configuration
{
    std::string getDataSetPath();

    std::string getOutputDirectory();
}