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

vector<int> single_ant_doing_stuff(vector<vector<double>> dist, 
    vector<vector<double>> pheromone, double alpha, double beta, double evaporation_rate,
    int ant_id, int n_cities) {

    vector<bool> visited(n_cities, false);
    int current_point = rand() % n_cities;
    visited[current_point] = true;

    vector<int> path;
    path.push_back(current_point);

    while (path.size() < n_cities) {
        vector<int> unvisited;
        for (int i = 0; i < n_cities; ++i) {
            if (!visited[i]) {
                unvisited.push_back(i);
            }
        }

        vector<double> probabilities(unvisited.size());
        double sum_probs = 0.0;

        for (size_t i = 0; i < unvisited.size(); ++i) {
            int next = unvisited[i];
            double tau = pow(pheromone[current_point][next], alpha);
            double eta = 1.0 / pow(dist[current_point][next], beta);
            probabilities[i] = tau * eta;
            sum_probs += probabilities[i];
        }

        for (double &prob : probabilities) {
            prob /= sum_probs;
        }

        random_device rd;
        mt19937 gen(rd());
        discrete_distribution<> distribution(probabilities.begin(), probabilities.end());

        int next_index = distribution(gen);
        int next_point = unvisited[next_index];

        visited[next_point] = true;
        path.push_back(next_point);
        current_point = next_point;
    }

    return path;
}

double count_path_length(const vector<vector<double>> &dist, const vector<int> &path) {
    double length = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        length += dist[path[i - 1]][path[i]];
    }
    // Complete the cycle
    length += dist[path.back()][path[0]];
    return length;
}

void ant_colony_optimization(vector<vector<double>> dist, 
    double alpha, double beta, double evaporation_rate,
    int n_iterations, int n_ants, int n_cities, double Q) {

    vector<vector<double>> pheromone(n_cities, vector<double>(n_cities, 1.0));
    vector<int> best_path;
    double best_path_length = numeric_limits<double>::max();

    for (int iter = 0; iter < n_iterations; ++iter) {
        vector<vector<int>> paths;
        vector<double> path_lengths;

        for (int j = 0; j < n_ants; ++j) {
            vector<int> path = single_ant_doing_stuff(dist, pheromone, alpha, beta, evaporation_rate, j, n_cities);
            double path_length = count_path_length(dist, path);

            paths.push_back(path);
            path_lengths.push_back(path_length);

            if (path_length < best_path_length) {
                best_path = path;
                best_path_length = path_length;
            }
        }

        // Evaporate pheromones
        for (auto &row : pheromone) {
            for (auto &value : row) {
                value *= evaporation_rate;
            }
        }

        // Update pheromones
        for (size_t k = 0; k < paths.size(); ++k) {
            const vector<int> &path = paths[k];
            double length = path_lengths[k];

            for (int i = 0; i < n_cities - 1; ++i) {
                int from = path[i];
                int to = path[i + 1];
                double delta = Q / length;
                pheromone[from][to] += delta;
                pheromone[to][from] += delta; // assuming symmetric TSP
            }

            // Complete the cycle
            int last = path[n_cities - 1];
            int first = path[0];
            double delta = Q / length;
            pheromone[last][first] += delta;
            pheromone[first][last] += delta;
        }
    }

    cout << "Best length: " << best_path_length << endl;
    cout << "Best path: ";
    for (int city : best_path) {
        cout << city << " ";
    }
    cout << best_path[0] << endl; // to show return to start
}
