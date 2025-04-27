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
            if (cumulative_prob >= r && cumulative_prob > 0) {
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

std::string prepare_output_path(const std::string& output_file) {
    if (output_file.find('/') == std::string::npos && output_file.find('\\') == std::string::npos) {
        return "./" + output_file;
    } else {
        return output_file;
    }
}

void worker(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file) {
    std::cout << "Running WORKER algorithm with CUDA...\n";

    auto start_total = std::chrono::high_resolution_clock::now();

    auto total_kernel = std::chrono::duration<double>::zero();
    auto total_pheromone = std::chrono::duration<double>::zero();

    int n_cities = graph.size();
    int m = n_cities; // number of ants = number of cities
    float Q = 1.0f;

    size_t matrix_size = n_cities * n_cities * sizeof(float);
    size_t array_size = m * n_cities * sizeof(int);
    size_t bool_array_size = m * n_cities * sizeof(bool);
    size_t float_array_size = m * n_cities * sizeof(float);
    size_t tour_lengths_size = m * sizeof(float);

    // Host distances matrix
    std::cout << "Host distances" << std::endl;
    std::vector<float> distances_host(n_cities * n_cities);
    for (int i = 0; i < n_cities; ++i) {
        for (int j = 0; j < n_cities; ++j) {
            distances_host[i * n_cities + j] = graph[i][j];
            std::cout << graph[i][j] << " ";
        }

        std::cout << std::endl;
    }

    

    // Device memory
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

    int n_ants = n_cities;

    int thread_worker_count = min(N_MAX_THREADS_PER_BLOCK, n_ants);
    int blocks_worker = (n_ants / thread_worker_count) + 1;

    // int threads_count = n_cities;
    // int blocks = (m + threads_count - 1) / threads_count; // enough blocks for all ants

    init_rng<<<blocks_worker, thread_worker_count>>>(d_states, seed);
    cudaDeviceSynchronize();

    int all_threads_pheromone = n_ants * n_ants;
    int threads_pheromone = min(N_MAX_THREADS_PER_BLOCK, all_threads_pheromone);
    int blocks_pheromone = (all_threads_pheromone / threads_pheromone) + 1;

    // Host buffers to fetch data back from GPU
    std::vector<int> tours_host(m * n_cities);
    std::vector<float> choice_info_host(n_cities * n_cities);
    std::vector<float> tour_lengths_host(m);

    for (int iter = 0; iter < num_iter; ++iter) {
        // std::cout << "\n=== Iteration " << iter + 1 << " ===\n";
        auto start_kernel = std::chrono::high_resolution_clock::now();
        workerAntKernel<<<blocks_worker, thread_worker_count>>>(m, n_cities, d_tours, d_choice_info, d_selection_prob_all, d_visited, d_tour_lengths, d_distances, d_states);
        cudaDeviceSynchronize();
        auto end_kernel = std::chrono::high_resolution_clock::now();

        total_kernel += end_kernel - start_kernel;

        auto start_kernel_pheromone = std::chrono::high_resolution_clock::now();
        pheromoneUpdateKernelBasic<<<blocks_worker, thread_worker_count>>>(
        // pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone>>>(
            alpha, 
            beta,
            evaporate,
            Q,
            d_pheromone,
            d_tours,
            n_cities,
            m,
            d_choice_info,
            d_distances,
            d_tour_lengths
        );
        cudaDeviceSynchronize();

        auto end_kernel_pheromone = std::chrono::high_resolution_clock::now();

        total_pheromone += end_kernel_pheromone - start_kernel_pheromone;

        // Copy back tours and lengths
        cudaMemcpy(tours_host.data(), d_tours, array_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(tour_lengths_host.data(), d_tour_lengths, tour_lengths_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(choice_info_host.data(), d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(initial_pheromone.data(), d_pheromone, matrix_size, cudaMemcpyDeviceToHost);

        // Print tours
        for (int ant = 0; ant < m; ++ant) {
            std::cout << "Ant " << ant << " tour: ";
            for (int step = 0; step < n_cities; ++step) {
                std::cout << tours_host[ant * n_cities + step] << " ";
            }
            std::cout << " (length: " << tour_lengths_host[ant] << ")\n";
        }

        std::cout << "Pheromone Info Matrix:\n";
        for (int i = 0; i < n_cities; ++i) {
            for (int j = 0; j < n_cities; ++j) {
                std::cout << std::fixed << std::setprecision(4) << initial_pheromone[i * n_cities + j] << "\t";
            }
            std::cout << "\n";
        }

        // Print choice_info matrix
        std::cout << "Choice Info Matrix:\n";
        for (int i = 0; i < n_cities; ++i) {
            for (int j = 0; j < n_cities; ++j) {
                std::cout << std::fixed << std::setprecision(4) << choice_info_host[i * n_cities + j] << "\t";
            }
            std::cout << "\n";
        }
    }

    // cudaMemcpy(tours_host.data(), d_tours, array_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(tour_lengths_host.data(), d_tour_lengths, tour_lengths_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(choice_info_host.data(), d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(initial_pheromone.data(), d_pheromone, matrix_size, cudaMemcpyDeviceToHost);


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

    auto end_total = std::chrono::high_resolution_clock::now(); // End total timer
    std::chrono::duration<double> total_duration = end_total - start_total;

    std::cout << "Total kernel time: " << total_kernel.count() << std::endl;
    std::cout << "Total kernel pheromone time: " << total_pheromone.count() << std::endl;
    std::cout << "Total time: " << total_duration.count() << std::endl;


    std::string output_path = prepare_output_path(output_file);
    std::ofstream out(output_path);

    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }

    // Assume `best` and `best_id` are already calculated, and `tours_host` is available.
    std::cout << "\nBest tour length: " << best << std::endl;
    out << "Best tour length: " << best << std::endl;

    for (int step = 0; step < n_cities; ++step) {
        std::cout << tours_host[best_id * n_cities + step] << " ";
        out << tours_host[best_id * n_cities + step] << " ";
    }
    std::cout << std::endl;
    out << std::endl;

    out.close(); 


}