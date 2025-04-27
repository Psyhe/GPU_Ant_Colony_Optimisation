#include "algorithm.h"
#include <iostream>
#include <fstream>
#include <string>

std::string another_output_path(const std::string& output_file) {
    if (output_file.find('/') == std::string::npos && output_file.find('\\') == std::string::npos) {
        // No slash -> assume it's a file name, save in current directory
        return "./" + output_file;
    } else {
        // Has slash -> treat as path to file
        return output_file;
    }
}

// void worker(const std::vector<std::vector<float>>& graph, int num_iter, double alpha, double beta, double evaporate, int seed, std::string output_file) {
//     std::cout << "Running WORKER algorithm...\n";
//     std::cout << "Parameters: NUM_ITER=" << num_iter << ", ALPHA=" << alpha 
//               << ", BETA=" << beta << ", EVAPORATE=" << evaporate << ", SEED=" << seed << std::endl;

//     std::string full_output_path = prepare_output_path(output_file);

//     std::ofstream ofs(full_output_path);
//     if (!ofs.is_open()) {
//         std::cerr << "Error: cannot open output file: " << full_output_path << std::endl;
//         return;
//     }

//     ofs << "Running WORKER algorithm...\n";
//     ofs << "Parameters: NUM_ITER=" << num_iter << ", ALPHA=" << alpha 
//         << ", BETA=" << beta << ", EVAPORATE=" << evaporate << ", SEED=" << seed << std::endl;
    
//     ofs.close();
// }

void queen(const std::vector<std::vector<float>>& graph, int num_iter, double alpha, double beta, double evaporate, int seed, std::string output_file) {
    std::cout << "Running QUEEN algorithm...\n";
    std::cout << "Parameters: NUM_ITER=" << num_iter << ", ALPHA=" << alpha 
              << ", BETA=" << beta << ", EVAPORATE=" << evaporate << ", SEED=" << seed << std::endl;

    std::string full_output_path = another_output_path(output_file);

    std::ofstream ofs(full_output_path);
    if (!ofs.is_open()) {
        std::cerr << "Error: cannot open output file: " << full_output_path << std::endl;
        return;
    }

    ofs << "Running QUEEN algorithm...\n";
    ofs << "Parameters: NUM_ITER=" << num_iter << ", ALPHA=" << alpha 
        << ", BETA=" << beta << ", EVAPORATE=" << evaporate << ", SEED=" << seed << std::endl;

    ofs.close();
}
