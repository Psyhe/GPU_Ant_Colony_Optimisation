#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "seq.h"

struct City {
    double x, y;
};

double euclideanDistance(const City& a, const City& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file_path>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << argv[1] << std::endl;
        return 1;
    }

    std::string line;
    int dimension = 0;

    // Read and skip metadata (first 6 lines), extract dimension
    for (int i = 0; i < 6; ++i) {
        std::getline(infile, line);
        if (line.find("DIMENSION") != std::string::npos) {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> tmp >> dimension;
        }
    }

    std::vector<City> cities(dimension);

    // Read NODE_COORD_SECTION
    for (int i = 0; i < dimension; ++i) {
        int index;
        double x, y;
        infile >> index >> x >> y;
        cities[index - 1] = {x, y};
    }

    infile.close();

    // Build distance matrix
    std::vector<std::vector<double>> dist(dimension, std::vector<double>(dimension, 0.0));

    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            if (i != j) {
                dist[i][j] = euclideanDistance(cities[i], cities[j]);
            }
        }
    }

    // Output distance matrix
    std::cout << std::fixed << std::setprecision(6);
    for (const auto& row : dist) {
        for (double d : row) {
            std::cout << d << " ";
        }
        std::cout << "\n";
    }

    std::cout << endl;

    double alpha = 1.0;
    double beta = 1.0;
    double evaporation_rate = 0.5;
    double Q = 1.0;

    int n_cities = dimension;
    int n_iterations = 20;
    int n_ants = 10;

    ant_colony_optimization(dist, 
        alpha, beta, evaporation_rate,
        n_iterations, n_ants, n_cities, Q);

    return 0;
}
