#include "algorithm.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#define THREADS_PER_BLOCK 256

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

void worker(const std::vector<std::vector<float>>& graph, int num_iter, double alpha, double beta, double evaporate, int seed, std::string output_file) {
    // std::cout << "Running WORKER algorithm with CUDA...\n";

    // int n_cities = graph.size();
    // int m = n_cities; // number of ants = number of cities
    // float Q = 100.0f;

    // size_t matrix_size = n_cities * n_cities * sizeof(float);
    // size_t array_size = m * n_cities * sizeof(int);
    // size_t bool_array_size = m * n_cities * sizeof(bool);
    // size_t float_array_size = m * n_cities * sizeof(float);
    // size_t tour_lengths_size = m * sizeof(float);

    // // Host distances matrix
    // std::vector<float> distances_host(n_cities * n_cities);
    // for (int i = 0; i < n_cities; ++i) {
    //     for (int j = 0; j < n_cities; ++j) {
    //         distances_host[i * n_cities + j] = graph[i][j];
    //     }
    // }

    // // Device memory
    // float *d_pheromone, *d_choice_info, *d_distances, *d_selection_prob_all, *d_tour_lengths;
    // int *d_tours;
    // bool *d_visited;
    // curandState* d_states;

    // cudaMalloc(&d_pheromone, matrix_size);
    // cudaMalloc(&d_choice_info, matrix_size);
    // cudaMalloc(&d_distances, matrix_size);
    // cudaMalloc(&d_tours, array_size);
    // cudaMalloc(&d_selection_prob_all, float_array_size);
    // cudaMalloc(&d_visited, bool_array_size);
    // cudaMalloc(&d_tour_lengths, tour_lengths_size);
    // cudaMalloc(&d_states, m * sizeof(curandState));

    // cudaMemcpy(d_distances, distances_host.data(), matrix_size, cudaMemcpyHostToDevice);
    // cudaMemset(d_pheromone, 0, matrix_size);

    // int blocks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // int blocks_matrix = (n_cities * n_cities + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // init_rng<<<blocks, THREADS_PER_BLOCK>>>(d_states, seed);
    // cudaDeviceSynchronize();

    // for (int iter = 0; iter < num_iter; ++iter) {
    //     workerAntKernel<<<blocks, THREADS_PER_BLOCK>>>(m, n_cities, d_tours, d_choice_info, d_selection_prob_all, d_visited, d_tour_lengths, d_distances, d_states);
    //     cudaDeviceSynchronize();

    //     pheromoneUpdateKernel<<<blocks_matrix, THREADS_PER_BLOCK>>>(
    //         static_cast<float>(alpha),
    //         static_cast<float>(beta),
    //         static_cast<float>(evaporate),
    //         Q,
    //         d_pheromone,
    //         d_tours,
    //         n_cities,
    //         m,
    //         d_choice_info,
    //         d_distances,
    //         d_tour_lengths
    //     );
    //     cudaDeviceSynchronize();
    // }

    // std::vector<float> tour_lengths_host(m);
    // cudaMemcpy(tour_lengths_host.data(), d_tour_lengths, tour_lengths_size, cudaMemcpyDeviceToHost);

    // float best = 1e9;
    // for (int i = 0; i < m; ++i) {
    //     if (tour_lengths_host[i] < best) {
    //         best = tour_lengths_host[i];
    //     }
    // }

    // std::string full_output_path = prepare_output_path(output_file);
    // std::ofstream ofs(full_output_path);
    // if (!ofs.is_open()) {
    //     std::cerr << "Error opening output file: " << full_output_path << std::endl;
    //     return;
    // }

    // ofs << best << "\n";
    // ofs.close();

    // cudaFree(d_pheromone);
    // cudaFree(d_choice_info);
    // cudaFree(d_distances);
    // cudaFree(d_tours);
    // cudaFree(d_selection_prob_all);
    // cudaFree(d_visited);
    // cudaFree(d_tour_lengths);
    // cudaFree(d_states);

    // std::cout << "Best tour length: " << best << std::endl;

    int n_cities = 5; // Number of cities
    int m = 5;       // Number of ants
    int n_iterations = 25; // Number of experiments

    // Allocate host memory
    std::vector<float> h_choice_info(n_cities * n_cities, 1.0f);
    std::vector<float> h_pheromone(n_cities * n_cities, 1.0f);
    std::vector<int> h_tours(m * n_cities);
    std::vector<float> h_tour_lengths(m);

    // Initialize random distance matrix
    std::vector<float> h_distances = {
        0.0f, 10.0f, 15.0f, 20.0f, 25.0f,
        10.0f, 0.0f, 35.0f, 25.0f, 30.0f,
        15.0f, 35.0f, 0.0f, 30.0f, 20.0f,
        20.0f, 25.0f, 30.0f, 0.0f, 15.0f,
        25.0f, 30.0f, 20.0f, 15.0f, 0.0f
    };

    // Print the distance matrix once
    std::cout << "Distance Matrix:\n";
    for (int i = 0; i < n_cities; i++) {
        for (int j = 0; j < n_cities; j++) {
            std::cout << h_distances[i * n_cities + j] << "\t";
        }
        std::cout << "\n";
    }

    // Allocate device memory
    float* d_distances;
    float* d_choice_info;
    float* d_pheromone;
    int* d_tours;
    float* d_tour_lengths;
    float* d_selection_prob_all;
    bool* d_visited;

    cudaMalloc(&d_distances, sizeof(float) * n_cities * n_cities);
    cudaMalloc(&d_choice_info, sizeof(float) * n_cities * n_cities);
    cudaMalloc(&d_pheromone, sizeof(float) * n_cities * n_cities);
    cudaMalloc(&d_tours, sizeof(int) * m * n_cities);
    cudaMalloc(&d_tour_lengths, sizeof(float) * m);
    cudaMalloc(&d_selection_prob_all, sizeof(float) * m * n_cities);
    cudaMalloc(&d_visited, sizeof(bool) * m * n_cities);

    // Copy static host data to device
    cudaMemcpy(d_distances, h_distances.data(), sizeof(float) * n_cities * n_cities, cudaMemcpyHostToDevice);

    curandState* d_states;
    cudaMalloc(&d_states, m * sizeof(curandState));

    unsigned long seed = 42;


    int threads_per_block = BLOCK_SIZE;
    int num_blocks_ants = (m + threads_per_block - 1) / threads_per_block;
    int num_blocks_pheromone = (n_cities * n_cities + threads_per_block - 1) / threads_per_block;

    float alpha = 1.0f;
    float beta = 2.0f;
    float evaporation_rate = 0.5f;
    float Q = 1.0f;
    // Initialize curand states on the device
    init_rng<<<num_blocks_ants, threads_per_block>>>(d_states, seed);
    cudaDeviceSynchronize();

    for (int iter = 0; iter < n_iterations; iter++) {
        std::cout << "\n=== Iteration " << iter + 1 << " ===\n";

        // Copy current pheromone and initial choice info to device
        cudaMemcpy(d_pheromone, h_pheromone.data(), sizeof(float) * n_cities * n_cities, cudaMemcpyHostToDevice);
        cudaMemcpy(d_choice_info, h_choice_info.data(), sizeof(float) * n_cities * n_cities, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tours, h_tours.data(), sizeof(int) * m * n_cities, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tour_lengths, h_tour_lengths.data(), sizeof(float) * m, cudaMemcpyHostToDevice);


        // Launch worker ants
        workerAntKernel<<<num_blocks_ants, threads_per_block>>>(
            m, n_cities,
            d_tours,
            d_choice_info,
            d_selection_prob_all,
            d_visited,
            d_tour_lengths,
            d_distances,
            d_states
        );
        cudaDeviceSynchronize();

        // Launch pheromone update
        pheromoneUpdateKernel<<<num_blocks_pheromone, threads_per_block>>>(
            alpha, beta, evaporation_rate, Q,
            d_pheromone,
            d_tours,
            n_cities,
            m,
            d_choice_info,
            d_distances,
            d_tour_lengths
        );
        cudaDeviceSynchronize();

        // Copy results back
        cudaMemcpy(h_tours.data(), d_tours, sizeof(int) * m * n_cities, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tour_lengths.data(), d_tour_lengths, sizeof(float) * m, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pheromone.data(), d_pheromone, sizeof(float) * n_cities * n_cities, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_choice_info.data(), d_choice_info, sizeof(float) * n_cities * n_cities, cudaMemcpyDeviceToHost);

        // Print results for this iteration
        std::cout << "Tours and their lengths:\n";
        for (int i = 0; i < m; i++) {
            std::cout << "Ant " << i << ": ";
            for (int j = 0; j < n_cities; j++) {
                std::cout << h_tours[i * n_cities + j] << " ";
            }
            std::cout << "| Length = " << h_tour_lengths[i] << "\n";
        }

        std::cout << "\nUpdated Pheromone Matrix:\n";
        for (int i = 0; i < n_cities; i++) {
            for (int j = 0; j < n_cities; j++) {
                std::cout << h_pheromone[i * n_cities + j] << " ";
            }
            std::cout << "\n";
        }
    }

    // Cleanup
    cudaFree(d_distances);
    cudaFree(d_choice_info);
    cudaFree(d_pheromone);
    cudaFree(d_tours);
    cudaFree(d_tour_lengths);
    cudaFree(d_selection_prob_all);
    cudaFree(d_visited);

    return 0;
}
