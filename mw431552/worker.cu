#include "algorithm.h"
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

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void workerAntKernelOptimizedStatic(
    int m, int n_cities,
    int* tours,
    float* choice_info,
    float* selection_prob_all,
    bool* visited,
    float* tour_lengths,
    float* distances,
    curandState* states
) {
    __shared__ float shared_selection_probs[N_MAX_THREADS_PER_BLOCK]; // STATIC shared array, fixed 1024

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) return;

    int offset = tid * n_cities;
    curandState localState = states[tid];

    // Local visited array in registers
    bool local_visited[N_MAX_THREADS_PER_BLOCK];

    // Initialize visited
    #pragma unroll
    for (int i = 0; i < n_cities; i++) {
        local_visited[i] = false;
    }

    int step = 0;
    int current_city = 0;
    tours[offset + step] = current_city;
    local_visited[current_city] = true;
    float tour_len = 0.0f;
    step++;

    while (step < n_cities) {
        // Load probabilities into shared memory
        for (int j = threadIdx.x; j < n_cities; j += blockDim.x) {
            if (local_visited[j]) {
                shared_selection_probs[j] = 0.0f;
            } else {
                shared_selection_probs[j] = choice_info[current_city * n_cities + j];
            }
        }

        __syncthreads();

        // Compute partial sum of probabilities
        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < n_cities; j += blockDim.x) {
            local_sum += shared_selection_probs[j];
        }

        // Reduce across warp
        local_sum = warpReduceSum(local_sum);

        // Now select next city
        int next_city = -1;
        if ((threadIdx.x & 31) == 0) {
            float r = curand_uniform(&localState) * local_sum;
            float cumulative_prob = 0.0f;
            for (int j = 0; j < n_cities; j++) {
                cumulative_prob += shared_selection_probs[j];
                if (cumulative_prob >= r) {
                    next_city = j;
                    break;
                }
            }
        }

        // Broadcast next_city to all threads in warp
        next_city = __shfl_sync(0xffffffff, next_city, 0);

        // Update tour
        tours[offset + step] = next_city;
        local_visited[next_city] = true;
        tour_len += distances[current_city * n_cities + next_city];

        current_city = next_city;
        step++;

        __syncthreads();
    }

    // Close the loop
    tour_len += distances[current_city * n_cities + tours[offset]];

    tour_lengths[tid] = tour_len;
    states[tid] = localState;
}

__global__ void workerAntKernel(
    int m, int n_cities,
    int* tours,
    float* choice_info,
    float* selection_prob_all,
    bool* visited,
    float* tour_lengths,
    float* distances,
    curandState* states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) return;

    int offset = tid * n_cities;
    curandState localState = states[tid];

    for (int i = 0; i < n_cities; i++) {
        visited[offset + i] = false;
    }

    int step = 0;
    int current_city = 0;
    tours[offset + step] = current_city;
    visited[offset + current_city] = true;
    float tour_len = 0.0f;

    step++;

    while (step < n_cities) {
        float sum_probs = 0.0f;
        for (int j = 0; j < n_cities; j++) {
            if (visited[offset + j]) {
                selection_prob_all[offset + j] = 0.0f;
            } else {
                float prob = choice_info[current_city * n_cities + j];
                selection_prob_all[offset + j] = prob;
                sum_probs += prob;
            }
        }

        if (sum_probs == 0.0f) break;

        float r = curand_uniform(&localState) * sum_probs;
        float cumulative_prob = 0.0f;
        int next_city = -1;

        for (int j = 0; j < n_cities; j++) {
            cumulative_prob += selection_prob_all[offset + j];
            if (cumulative_prob >= r) {
                next_city = j;
                break;
            }
        }

        tours[offset + step] = next_city;
        visited[offset + next_city] = true;
        tour_len += distances[current_city * n_cities + next_city];

        current_city = next_city;
        step++;
    }

    tour_len += distances[current_city * n_cities + tours[offset]];
    tour_lengths[tid] = tour_len;

    states[tid] = localState;
}

void worker(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file) {
    std::cout << "Running WORKER algorithm with CUDA...\n";

    cudaEvent_t start_total, end_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&end_total);
    cudaEventRecord(start_total);

    float total_kernel = 0.0f;
    float total_pheromone = 0.0f;

    int n_cities = graph.size();
    int m = n_cities;
    float Q = 1.0f;

    size_t matrix_size = n_cities * n_cities * sizeof(float);
    size_t array_size = m * n_cities * sizeof(int);
    size_t bool_array_size = m * n_cities * sizeof(bool);
    size_t float_array_size = m * n_cities * sizeof(float);
    size_t tour_lengths_size = m * sizeof(float);

    std::vector<float> distances_host(n_cities * n_cities);
    for (int i = 0; i < n_cities; ++i) {
        for (int j = 0; j < n_cities; ++j) {
            distances_host[i * n_cities + j] = graph[i][j];
        }
    }

    float *d_pheromone, *d_choice_info, *d_distances, *d_selection_prob_all, *d_tour_lengths;
    int *d_tours;
    bool *d_visited;
    curandState* d_states;

    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    cudaMalloc(&d_tours, array_size);
    cudaMalloc(&d_selection_prob_all, float_array_size);
    cudaMalloc(&d_visited, bool_array_size);
    cudaMalloc(&d_tour_lengths, tour_lengths_size);
    cudaMalloc(&d_states, m * sizeof(curandState));

    cudaMemcpy(d_distances, distances_host.data(), matrix_size, cudaMemcpyHostToDevice);
    std::vector<float> initial_pheromone(n_cities * n_cities, 1.0f);
    cudaMemcpy(d_pheromone, initial_pheromone.data(), matrix_size, cudaMemcpyHostToDevice);

    int thread_worker_count = std::min(N_MAX_THREADS_PER_BLOCK, m);
    int blocks_worker = (m + thread_worker_count - 1) / thread_worker_count;

    init_rng<<<blocks_worker, thread_worker_count>>>(d_states, seed);
    cudaDeviceSynchronize();

    int all_threads_pheromone = m * n_cities;
    int threads_pheromone = std::min(N_MAX_THREADS_PER_BLOCK, all_threads_pheromone);
    int blocks_pheromone = (all_threads_pheromone + threads_pheromone - 1) / threads_pheromone;

    std::vector<int> tours_host(m * n_cities);
    std::vector<float> choice_info_host(n_cities * n_cities);
    std::vector<float> tour_lengths_host(m);

    cudaEvent_t start_kernel, end_kernel;
    cudaEvent_t start_pheromone, end_pheromone;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    cudaEventCreate(&start_pheromone);
    cudaEventCreate(&end_pheromone);

    pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone>>>(
        alpha, beta, evaporate, Q,
        d_pheromone, d_tours, n_cities, m,
        d_choice_info, d_distances, d_tour_lengths
    );
    cudaDeviceSynchronize();

    for (int iter = 0; iter < num_iter; ++iter) {
        cudaEventRecord(start_kernel);
        // workerAntKernel<<<blocks_worker, thread_worker_count>>>(
        workerAntKernelOptimizedStatic<<<blocks_worker, thread_worker_count>>>(
            m, n_cities, d_tours, d_choice_info, d_selection_prob_all,
            d_visited, d_tour_lengths, d_distances, d_states
        );
        cudaDeviceSynchronize();
        cudaEventRecord(end_kernel);
        cudaEventSynchronize(end_kernel);

        float kernel_time = 0.0f;
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
        total_kernel += kernel_time;

        cudaEventRecord(start_pheromone);
        pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone>>>(
            alpha, beta, evaporate, Q,
            d_pheromone, d_tours, n_cities, m,
            d_choice_info, d_distances, d_tour_lengths
        );
        cudaDeviceSynchronize();
        cudaEventRecord(end_pheromone);
        cudaEventSynchronize(end_pheromone);

        float pheromone_time = 0.0f;
        cudaEventElapsedTime(&pheromone_time, start_pheromone, end_pheromone);
        total_pheromone += pheromone_time;
    }

    cudaMemcpy(tours_host.data(), d_tours, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(tour_lengths_host.data(), d_tour_lengths, tour_lengths_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(choice_info_host.data(), d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(initial_pheromone.data(), d_pheromone, matrix_size, cudaMemcpyDeviceToHost);

    float best = 1e9;
    int best_id = 0;
    for (int i = 0; i < m; ++i) {
        if (tour_lengths_host[i] < best) {
            best = tour_lengths_host[i];
            best_id = i;
        }
    }

    cudaFree(d_pheromone);
    cudaFree(d_choice_info);
    cudaFree(d_distances);
    cudaFree(d_tours);
    cudaFree(d_selection_prob_all);
    cudaFree(d_visited);
    cudaFree(d_tour_lengths);
    cudaFree(d_states);

    cudaEventRecord(end_total);
    cudaEventSynchronize(end_total);

    float total_time = 0.0f;
    cudaEventElapsedTime(&total_time, start_total, end_total);

    std::cout << "Total kernel time: " << total_kernel / 1000.0f << " s" << std::endl;
    std::cout << "Total kernel pheromone time: " << total_pheromone / 1000.0f << " s" << std::endl;
    std::cout << "Average kernel time: " << total_kernel / num_iter << " ms" << std::endl;
    std::cout << "Average pheromone kernel time: " << total_pheromone / num_iter << " ms" << std::endl;

    std::cout << "Total time: " << total_time / 1000.0f << " s" << std::endl;

    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(end_kernel);
    cudaEventDestroy(start_pheromone);
    cudaEventDestroy(end_pheromone);

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
