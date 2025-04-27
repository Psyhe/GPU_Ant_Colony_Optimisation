#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <vector>
#include <string>

void worker(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file);
void queen(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file);

#endif // ALGORITHM_H
