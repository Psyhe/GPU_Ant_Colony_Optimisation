#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>
#include <string>
#include <vector>

#define N_CITIES 1024

std::string prepare_output_path(const std::string& output_file);

void generate_output(float total_kernel, int num_iter, float total_time_ms, std::string output_file, std::vector<int> tours_host, int best_id, float best, int n_cities);

__global__ void init_rng(curandState* states, unsigned long seed);

__global__ void queenAntKernel(float *choice_info, float *distances, int *tours, float *tour_lengths, int n_cities, curandState *states);

__global__ void pheromoneUpdateKernel(
    float alpha,
    float beta,
    float evaporation_rate,
    float Q,
    float *pheromone,
    int *tours,
    int n_cities,
    int m,
    float *choice_info,
    float *distances,
    float *tour_lengths
);

__global__ void pheromoneUpdateKernelBasic(
    float alpha,
    float beta,
    float evaporation_rate,
    float Q,
    float *pheromone,
    int *tours,
    int n_cities,
    int m,
    float *choice_info,
    float *distances,
    float *tour_lengths
);

void runGraphIterations(cudaGraphExec_t graph_exec, cudaStream_t stream, int num_iter, float &total_kernel);


#endif // KERNELS_H