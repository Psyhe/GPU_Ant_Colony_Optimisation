#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <random>

using namespace std;

vector<int> single_ant_doing_stuff(std::vector<std::vector<double>> dist, 
    std::vector<std::vector<double>> pheromone, double alpha, double beta, double evaporation_rate,
    int ant_id, int n_cities) {

    vector<bool> visited(n_cities, false);

    int false_count = n_cities - 1;
        
    int current_point = rand() % n_cities;

    visited[current_point] = true;

    vector<int> path;
    path.push_back(current_point);
    
    while (false_count > 0) {
        vector<int> unvisited;

        for (int i = 0; i < n_cities; ++i) {
            if (!visited[i]) {
                unvisited.push_back(i);
            }
        }

        vector<double> probabilities(unvisited.size());

        for (size_t i = 0; i < unvisited.size(); ++i) {
            int unvisited_point = unvisited[i];
            double tau = pow(pheromone[current_point][unvisited_point], alpha);
            double eta = 1.0 / pow(dist[current_point][unvisited_point], beta);
            probabilities[i] = tau * eta;
        }

        double sum_probs = 0;

        for (auto &prob: probabilities) {
            sum_probs += prob;
        }

        for (auto &prob: probabilities) {
            prob = prob/sum_probs;
        }

        // Randomly choose the next point based on the probabilities
        random_device rd;
        mt19937 gen(rd());
        discrete_distribution<> distribution(probabilities.begin(), probabilities.end());

        int next_index = distribution(gen);
        int next_point = unvisited[next_index];

        path.push_back(next_point);
        visited[next_point] = true;
        current_point = next_point;
    }

    return path;
}

double count_path_length(std::vector<std::vector<double>> dist, vector<int> path) {

    double length = 0.0;

    for (int i = 1; i < path.size(); i++) {
        length += dist[i-1][i];
    }

    return length;
}


void ant_colony_optimization(std::vector<std::vector<double>> dist, 
    double alpha, double beta, double evaporation_rate,
    int n_iterations, int n_ants, int n_cities, double Q) {


    std::vector<std::vector<double>> pheromone(n_cities, std::vector<double>(n_cities, 1.0));

    vector<int> best_path;
    int best_path_length = std::numeric_limits<double>::max();

    for (int i = 0; i < n_iterations; i++) {
        std::vector<std::vector<int>> paths;

        std::vector<int> path_lengths;


        for (int j=0; j< n_ants; j++) {
            // uzupełnić paths
            vector<int> path = single_ant_doing_stuff(dist, pheromone, alpha, beta, evaporation_rate, j, n_cities);

            int path_length = count_path_length(dist, path);

            if (path_length < best_path_length) {
                best_path = path;
                best_path_length = path_length;
            }


        }

        for (auto& pher: pheromone) {
            for (auto& cell: pher) {
                cell = cell * evaporation_rate;
            }
        }   

        for (size_t k = 0; k < paths.size(); ++k) {
            const std::vector<int>& path = paths[k];
            double path_length = path_lengths[k];
        
            for (int i = 0; i < n_cities - 1; ++i) {
                int from = path[i];
                int to = path[i + 1];
                pheromone[from][to] += Q / path_length;
            }
        
            // Complete the cycle (return to start)
            int last = path[n_cities - 1];
            int first = path[0];
            pheromone[last][first] += Q / path_length;
        }
    }

    cout << "Best length" << endl;
    cout << best_path_length << endl;
    cout << "Path:" << endl;
    for (int i = 0; i < best_path.size(); i++) {
        cout << best_path[i] << " ";
    }

}





