#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <climits>

using namespace std;

void single_ant_doing_stuff(std::vector<std::vector<double>> dist, 
    std::vector<std::vector<double>> feromone, double alpha, double beta, double evaporation_rate) {



}

void ant_colony_optimization(std::vector<std::vector<double>> dist, 
    std::vector<std::vector<double>> feromone, double alpha, double beta, double evaporation_rate,
    int n_iterations, int n_ants, int n_cities) {


    std::vector<std::vector<double>> feromone(n_cities, std::vector<double>(n_cities, 1.0));

    vector<int> best_path;
    int best_path_length = INT_MAX;

    for (int i = 0; i < n_iterations; i++) {
        std::vector<std::vector<int>> paths;

        std::vector<int> path_lengths;


        for (int j=0; j< n_ants; j++) {
            vector<bool> visited(n_cities, false);
            
        }
    }

}





