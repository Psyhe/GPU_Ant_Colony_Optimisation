#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <chrono>
#include "utils.h"

#define N_MAX_THREADS_PER_BLOCK 1024

std::string prepare_output_path(const std::string& output_file) {
    if (output_file.find('/') == std::string::npos && output_file.find('\\') == std::string::npos) {
        return "./" + output_file;
    } else {
        return output_file;
    }
}

void generate_output(float total_kernel, int num_iter, float total_time_ms, std::string output_file, std::vector<int> tours_host, int n_cities, std::vector<float> tour_lengths_host) {
    
    float best = 1e9;
    int best_id = 0;
    for (int i = 0; i < n_cities; ++i) {
        if (tour_lengths_host[i] < best) {
            best = tour_lengths_host[i];
            best_id = i;
        }
    }

    // std::cout << "Total kernel+pheromone time: " << total_kernel / 1000.0f << " seconds" << std::endl;
    // std::cout << "Average graph execution time: " << total_kernel / num_iter << " ms" << std::endl;
    // std::cout << "Total time: " << total_time_ms / 1000.0f << " seconds" << std::endl;

    std::string output_path = prepare_output_path(output_file);
    std::ofstream out(output_path);


    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }

    std::cout << best << std::endl;
    out << best << std::endl;

    for (int step = 0; step < n_cities; ++step) {
        std::cout << tours_host[best_id * n_cities + step] + 1 << " ";
        out << tours_host[best_id * n_cities + step] + 1 << " ";
    }
    std::cout << std::endl;
    out << std::endl;

    out.close();
}

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
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) return;

    float tour_len = tour_lengths[tid];

    for (int i = 0; i < n_cities - 1; i++) {
        int current_city = tours[tid * n_cities + i];
        int next_city = tours[tid * n_cities + i + 1];

        float delta_pheromone = Q / tour_len;

        // Atomic updates to avoid race conditions
        atomicAdd(&pheromone[current_city * n_cities + next_city], delta_pheromone);
        atomicAdd(&pheromone[next_city * n_cities + current_city], delta_pheromone);
    }

    int last_city = tours[tid * n_cities + n_cities - 1];
    int start_city = tours[tid * n_cities];

    float delta_pheromone = Q / tour_len;
    atomicAdd(&pheromone[last_city * n_cities + start_city], delta_pheromone);
    atomicAdd(&pheromone[start_city * n_cities + last_city], delta_pheromone);
}

__global__ void pheromoneEvaporationAndChoiceInfoKernel(
    float alpha,
    float beta,
    float evaporation_rate,
    float *pheromone,
    float *choice_info,
    float *distances,
    int n_cities
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = n_cities * n_cities;
    if (tid >= total_cells) return;

    pheromone[tid] *= (1.0f - evaporation_rate);

    float distance = distances[tid];
    if (distance > 0.0f) {
        float tau = __powf(pheromone[tid], alpha);
        float eta = __powf(1.0f / distance, beta);
        choice_info[tid] = tau * eta;
    } else {
        choice_info[tid] = 0.0f;
    }
}


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
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cities * n_cities) return;

    int local_X = tid % n_cities;
    int local_Y = tid / n_cities;

    pheromone[tid] *= (1.0f - evaporation_rate);

    float pheromone_update_value = 0.0f;

    for (int i = 0; i < m; i++) {
        int offset = i * n_cities;
        for (int j = 0; j < n_cities - 1; j++) {
            if ((tours[offset + j] == local_X && tours[offset + j + 1] == local_Y) ||
                (tours[offset + j] == local_Y && tours[offset + j + 1] == local_X))  {
                pheromone_update_value += Q / tour_lengths[i];
            }
        }
        if ((tours[offset + n_cities - 1] == local_X && tours[offset] == local_Y) ||
            (tours[offset + n_cities - 1] == local_Y && tours[offset] == local_X)) {
            pheromone_update_value += Q / tour_lengths[i];
        }
    }

    pheromone[tid] += pheromone_update_value;

    if (distances[local_X * n_cities + local_Y] > 0.0f) {
        float tau = __powf(pheromone[tid], alpha);
        float eta = __powf(1.0f / distances[local_X * n_cities + local_Y], beta);
        choice_info[tid] = tau * eta;
    } else {
        choice_info[tid] = 0.0f;
    }
}

__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

void runGraphIterations(cudaGraphExec_t graph_exec, cudaStream_t stream, int num_iter, float &total_kernel) {
    cudaEvent_t start_kernel, end_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);

    for (int iter = 0; iter < num_iter; ++iter) {
        cudaEventRecord(start_kernel, stream);

        cudaGraphLaunch(graph_exec, stream);
        cudaStreamSynchronize(stream);

        cudaEventRecord(end_kernel, stream);
        cudaEventSynchronize(end_kernel);

        float iter_time = 0.0f;
        cudaEventElapsedTime(&iter_time, start_kernel, end_kernel);

        total_kernel += iter_time;
    }

    cudaEventDestroy(start_kernel);
    cudaEventDestroy(end_kernel);
}