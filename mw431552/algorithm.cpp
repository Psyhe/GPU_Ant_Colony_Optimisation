#include "algorithm.h"
#include <iostream>
#include <fstream>
#include <string>

std::string another_output_path(const std::string& output_file) {
    if (output_file.find('/') == std::string::npos && output_file.find('\\') == std::string::npos) {
        return "./" + output_file;
    } else {
        return output_file;
    }
}