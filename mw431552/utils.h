#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>
#include <string>
#include <string>

#define N_CITIES 1024

std::string prepare_output_path(const std::string& output_file);

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



#endif // KERNELS_H