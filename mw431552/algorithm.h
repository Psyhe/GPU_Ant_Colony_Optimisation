#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <vector>
#include <string>

void worker_no_graph(const std::vector<std::vector<float>>& graph_constructed, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file);
void worker(const std::vector<std::vector<float>>& graph_constructed, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file);
void queen_no_graph(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file);
void queen(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file);

#endif // ALGORITHM_H
