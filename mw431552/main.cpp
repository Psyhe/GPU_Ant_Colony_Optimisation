#include <iostream>
#include <string>
#include "graph.h"
#include "algorithm.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 8) {
        cout << "Usage: ./acotsp <input_file> <output_file> <TYPE> <NUM_ITER> <ALPHA> <BETA> <EVAPORATE> <SEED>" << endl;
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2]; // Not used yet
    string type = argv[3];
    int num_iter = stoi(argv[4]);
    float alpha = stod(argv[5]);
    float beta = stod(argv[6]);
    float evaporate = stod(argv[7]);
    int seed = stoi(argv[8]);

    auto graph = read_input_file(input_file);

    // print_graph(graph);

    if (type == "WORKER") {
        worker(graph, num_iter, alpha, beta, evaporate, seed, output_file);
        worker_no_graph(graph, num_iter, alpha, beta, evaporate, seed, output_file);
        queen(graph, num_iter, alpha, beta, evaporate, seed, output_file);
        queen_no_graph(graph, num_iter, alpha, beta, evaporate, seed, output_file);
    } else if (type == "QUEEN") {
        queen(graph, num_iter, alpha, beta, evaporate, seed, output_file);
        queen_no_graph(graph, num_iter, alpha, beta, evaporate, seed, output_file);
    } else {
        cerr << "Unknown TYPE: " << type << ". Expected WORKER or QUEEN." << endl;
        return 1;
    }

    return 0;
}
