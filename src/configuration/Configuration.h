#pragma once
#include <string>

#ifndef USERNAME
#define USERNAME std::string("lisa")
#endif

namespace Configuration
{
    std::string getDataSetPath();

    std::string getOutputDirectory();
}