#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <vector>
#include <string>

void worker(const std::vector<std::vector<float>>& graph, int num_iter, double alpha, double beta, double evaporate, int seed, std::string output_file);
void queen(const std::vector<std::vector<float>>& graph, int num_iter, double alpha, double beta, double evaporate, int seed, std::string output_file);

#endif // ALGORITHM_H
