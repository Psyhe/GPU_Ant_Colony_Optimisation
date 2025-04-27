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
#define N_CITIES 1024

__global__ void queenAntKernel(float *choice_info, float *distances, int *tours, float *tour_lengths, int n_cities, curandState *states) {
    __shared__ int tabu_list[N_CITIES];
    __shared__ float probability_list[N_CITIES];

    int tid = threadIdx.x;
    int queen_id = blockIdx.x; // Each block is one queen

    curandState state = states[queen_id];

    if (tid < n_cities) {
        tabu_list[tid] = 1; // 1 = unvisited
        probability_list[tid] = 0.0;
    }
    __syncthreads();

    int current_city = 0;
    if (tid == 0) {
        tabu_list[current_city] = 0; // Start at city 0
        tours[queen_id * n_cities + 0] = current_city;
    }
    __syncthreads();

    float tour_len = 0.0f;

    for (int step = 1; step < n_cities; step++) {
        if (tid < n_cities) {
            probability_list[tid] = choice_info[current_city * n_cities + tid] * tabu_list[tid];
        }
        __syncthreads();

        if (tid == 0) {
            float total_prob = 0.0f;
            for (int i = 0; i < n_cities; i++) {
                total_prob += probability_list[i];
            }

            float rand_val = curand_uniform(&state) * total_prob;
            float cumulative = 0.0f;
            int selected_city = -1;
            for (int i = 0; i < n_cities; i++) {
                cumulative += probability_list[i];
                if (cumulative >= rand_val && rand_val > 0) {
                    selected_city = i;
                    break;
                }
            }

            if (selected_city == -1) selected_city = 0; // fallback

            tours[queen_id * n_cities + step] = selected_city;
            tour_len += distances[current_city * n_cities + selected_city];
            tabu_list[selected_city] = 0;
            current_city = selected_city;
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Add return to start city
        tour_len += distances[current_city * n_cities + tours[queen_id * n_cities]];
        tour_lengths[queen_id] = tour_len;
    }

    states[queen_id] = state;
}

void queen(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file) {
    std::cout << "Running QUEEN algorithm with CUDA...\n";

    auto start_total = std::chrono::high_resolution_clock::now();
    auto total_kernel = std::chrono::duration<double>::zero();
    auto total_pheromone = std::chrono::duration<double>::zero();

    int n_cities = graph.size();
    int m = n_cities; // number of queens = number of cities
    float Q = 1.0f;

    size_t matrix_size = n_cities * n_cities * sizeof(float);
    size_t array_size = m * n_cities * sizeof(int);
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
    float *d_pheromone, *d_choice_info, *d_distances, *d_tour_lengths;
    int *d_tours;
    curandState* d_states;

    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    cudaMalloc(&d_tours, array_size);
    cudaMalloc(&d_tour_lengths, tour_lengths_size);
    cudaMalloc(&d_states, m * sizeof(curandState));

    cudaMemcpy(d_distances, distances_host.data(), matrix_size, cudaMemcpyHostToDevice);

    std::vector<float> initial_pheromone(n_cities * n_cities, 1.0f);
    cudaMemcpy(d_pheromone, initial_pheromone.data(), matrix_size, cudaMemcpyHostToDevice);

    int thread_queen_count = std::min(N_MAX_THREADS_PER_BLOCK, n_cities);
    int blocks_queen = std::min(N_MAX_THREADS_PER_BLOCK, n_cities);

    init_rng<<<blocks_queen, thread_queen_count>>>(d_states, seed);
    cudaDeviceSynchronize();

    int all_threads_pheromone = n_cities * n_cities;
    int threads_pheromone = std::min(N_MAX_THREADS_PER_BLOCK, all_threads_pheromone);
    int blocks_pheromone = (all_threads_pheromone + threads_pheromone - 1) / threads_pheromone;

    std::vector<int> tours_host(m * n_cities);
    std::vector<float> tour_lengths_host(m);
    std::vector<float> choice_info_host(n_cities * n_cities);

    for (int iter = 0; iter < num_iter; ++iter) {
        auto start_kernel = std::chrono::high_resolution_clock::now();
        queenAntKernel<<<m, n_cities>>>(d_choice_info, d_distances, d_tours, d_tour_lengths, n_cities, d_states);
        cudaDeviceSynchronize();
        auto end_kernel = std::chrono::high_resolution_clock::now();
        total_kernel += end_kernel - start_kernel;

        auto start_kernel_pheromone = std::chrono::high_resolution_clock::now();
        pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone>>>(
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

        // Fetch results
        // cudaMemcpy(tours_host.data(), d_tours, array_size, cudaMemcpyDeviceToHost);
        // cudaMemcpy(tour_lengths_host.data(), d_tour_lengths, tour_lengths_size, cudaMemcpyDeviceToHost);
        // cudaMemcpy(choice_info_host.data(), d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
        // cudaMemcpy(initial_pheromone.data(), d_pheromone, matrix_size, cudaMemcpyDeviceToHost);

        // Print tours
        // for (int queen = 0; queen < m; ++queen) {
        //     std::cout << "Queen " << queen << " tour: ";
        //     for (int step = 0; step < n_cities; ++step) {
        //         std::cout << tours_host[queen * n_cities + step] << " ";
        //     }
        //     std::cout << " (length: " << tour_lengths_host[queen] << ")\n";
        // }

        // std::cout << "Pheromone Info Matrix:\n";
        // for (int i = 0; i < n_cities; ++i) {
        //     for (int j = 0; j < n_cities; ++j) {
        //         std::cout << std::fixed << std::setprecision(4) << initial_pheromone[i * n_cities + j] << "\t";
        //     }
        //     std::cout << "\n";
        // }

        // std::cout << "Choice Info Matrix:\n";
        // for (int i = 0; i < n_cities; ++i) {
        //     for (int j = 0; j < n_cities; ++j) {
        //         std::cout << std::fixed << std::setprecision(4) << choice_info_host[i * n_cities + j] << "\t";
        //     }
        //     std::cout << "\n";
        // }
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
    cudaFree(d_tour_lengths);
    cudaFree(d_states);

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total - start_total;

    std::cout << "Total kernel time: " << total_kernel.count() << std::endl;
    std::cout << "Total pheromone update time: " << total_pheromone.count() << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << std::endl;

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
        out << tours_host[best_id * n_cities + step] << " ";
    }
    std::cout << std::endl;
    out << std::endl;

    out.close();
}
