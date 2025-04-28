#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip> // for better formatting
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

void generate_output(float total_kernel, int num_iter, float total_time_ms, std::string output_file, float *tours_host, int best_id, float best) {
    std::cout << "Total kernel+pheromone time: " << total_kernel / 1000.0f << " seconds" << std::endl;
    std::cout << "Average graph execution time: " << total_kernel / num_iter << " ms" << std::endl;
    std::cout << "Total time: " << total_time_ms / 1000.0f << " seconds" << std::endl;

    std::string output_path = prepare_output_path(output_file);
    std::ofstream out(output_path);

    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }

    std::cout << "\nBest tour length: " << best << std::endl;
    out << "Best tour length: " << best << std::endl;

    for (int step = 0; step < n_cities; ++step) {
        std::cout << tours_host[best_id * n_cities + step] << " ";
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

    for (int i = 0; i < n_cities; i++) {
        pheromone[tid * n_cities + i] *= (1.0f - evaporation_rate);
    }

    float tour_len = tour_lengths[tid];

    for (int i = 0; i < n_cities-1; i++) {
        int current_city = tours[tid * n_cities + i];
        int next_city = tours[tid * n_cities + i + 1];

        pheromone[next_city * n_cities + current_city] += Q / tour_len;
        pheromone[current_city * n_cities + next_city] += Q / tour_len;
    }

    // Add return to starting city
    int current_city = tours[tid * n_cities + n_cities-1];
    int start_city = tours[tid * n_cities];

    pheromone[current_city * n_cities + start_city] += Q / tour_len;
    pheromone[start_city * n_cities + current_city] += Q / tour_len;

    
    // wszystkie watki musza zostawic swoje feromony
    __syncthreads();

    for (int i = 0; i < n_cities; i++) {
        float tau = __powf(pheromone[tid * n_cities + i], alpha);
        float eta = __powf(1.0f / distances[tid * n_cities + i], beta);
        choice_info[tid * n_cities + i] = tau * eta;
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