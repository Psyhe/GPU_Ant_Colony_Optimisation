#ifndef SEQ_H
#define SEQ_H

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

void ant_colony_optimization(std::vector<std::vector<double>> dist, 
    double alpha, double beta, double evaporation_rate,
    int n_iterations, int n_ants, int n_cities, double Q);

#endif