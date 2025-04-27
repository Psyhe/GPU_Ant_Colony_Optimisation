#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>

std::vector<std::vector<float>> read_input_file(const std::string& filename);

void print_graph(const std::vector<std::vector<float>>& graph);

#endif // GRAPH_H
