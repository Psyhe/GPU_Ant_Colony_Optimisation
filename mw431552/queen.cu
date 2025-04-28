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
#define N_CURRENT_CITIES 1

__global__ void queenAntKernelOptimized(float *choice_info, float *distances, int *tours, float *tour_lengths, int n_cities, curandState *states) {

    __shared__ int tabu[N_CITIES];
    __shared__ float probabilities[N_CITIES];
    __shared__ int current_city;
    
    int tid = threadIdx.x;
    
    if (tid >= n_cities)
        return;

    int queen_id = blockIdx.x;
    int n_threads = blockDim.x;

    int *tour = &tours[queen_id * n_cities];
    curandState localState = states[queen_id];
    
    tabu[tid] = 1; // Not visited yet
    
    __syncthreads();

    float tour_len = 0.0f;

    int start = queen_id % n_cities;
    if (tid == 0) {
        tour[0] = start;
        tabu[start] = 0; // Mark start city as visited
    }
    __syncthreads();

    current_city = start;

    for (int step = 1; step < n_cities; step++) {
        probabilities[tid] = choice_info[current_city * n_cities + tid] * tabu[tid];
        
        __syncthreads();

        // Warp-level reduction to compute total probability
        float local_prob = probabilities[tid];
        
        // Use warp shuffle reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            local_prob += __shfl_down_sync(0xffffffff, local_prob, offset);
        }

        __shared__ float total;
        if (tid % 32 == 0) { // Only one thread per warp writes
            atomicAdd(&total, local_prob);
        }
        __syncthreads();

        if (tid == 0) {
            double r = curand_uniform(&localState) * total;
            double cumulative = 0.0;
            int next_city = -1;
            for (int i = 0; i < n_cities; i++) {
                cumulative += probabilities[i];
                if (cumulative >= r) {
                    next_city = i;
                    break;
                }
            }
            if (next_city == -1) {
                // fallback
                for (int i = 0; i < n_cities; i++) {
                    if (tabu[i]) {
                        next_city = i;
                        break;
                    }
                }
            }
            tour[step] = next_city;
            tabu[next_city] = 0; // mark as visited
            tour_len += distances[current_city * n_cities + next_city];
            current_city = next_city;
        }
        __syncthreads();
    }

    if (tid == 0) {
        tour_len += distances[current_city * n_cities + tour[0]]; // Assuming you want a full tour
        tour_lengths[queen_id] = tour_len;
        states[queen_id] = localState;
    }
}


__global__ void queenAntKernel(float *choice_info, float *distances, int *tours, float *tour_lengths, int n_cities, curandState *states) {

    __shared__ int tabu[N_CITIES];
    __shared__ float probabilities[N_CITIES];
    __shared__ int current_city;
    
    int tid = threadIdx.x;
    
    if (tid >= n_cities)
        return;

    int queen_id = blockIdx.x;
    int n_threads = blockDim.x;

    int *tour = &tours[queen_id * (n_cities )];
    curandState localState = states[queen_id];
    
    tabu[tid] = 1; // Not visited yet
    
    __syncthreads();

    float tour_len = 0.0f;

    int start = queen_id % n_cities;
    if (tid == 0) {
        tour[0] = start;
        tabu[start] = 0; // Mark start city as visited
    }
    __syncthreads();

    current_city = start;

    for (int step = 1; step < n_cities; step++) {
        probabilities[tid] = choice_info[current_city * n_cities + tid] * tabu[tid];

        __syncthreads();

        // Thread 0 does roulette wheel selection
        double total = 0.0;
        if (tid == 0) {
            for (int i = 0; i < n_cities; i++) {
                total += probabilities[i];
            }
            double r = curand_uniform(&localState) * total;
            double cumulative = 0.0;
            int next_city = -1;
            for (int i = 0; i < n_cities; i++) {
                cumulative += probabilities[i];
                if (cumulative >= r) {
                    next_city = i;
                    break;
                }
            }
            if (next_city == -1) {
                // fallback
                for (int i = 0; i < n_cities; i++) {
                    if (tabu[i]) {
                        next_city = i;
                        break;
                    }
                }
            }
            tour[step] = next_city;
            tabu[next_city] = 0; // mark as visited
            tour_len += distances[current_city * n_cities + next_city];
            current_city = next_city;
        }
        __syncthreads();
    }

    if (tid == 0) {
        tour_len += distances[n_cities * queen_id + current_city];
        tour_lengths[queen_id] = tour_len;
        states[queen_id] = localState;
    }
}

void queen_old(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file) {
    std::cout << "Running QUEEN WORKER algorithm with CUDA...\n";

    cudaEvent_t start_total, end_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&end_total);
    cudaEventRecord(start_total);

    float total_kernel = 0.0f;
    float total_pheromone = 0.0f;

    int n_cities = graph.size();
    int m = n_cities; // number of ants = number of cities
    float Q = 1.0f;

    size_t matrix_size = n_cities * n_cities * sizeof(float);
    size_t array_size = m * n_cities * sizeof(int);
    size_t tour_lengths_size = m * sizeof(float);

    // Host distances matrix
    std::vector<float> distances_host(n_cities * n_cities);
    for (int i = 0; i < n_cities; ++i) {
        for (int j = 0; j < n_cities; ++j) {
            distances_host[i * n_cities + j] = graph[i][j];
        }
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

    int n_ants = n_cities;

    int thread_worker_count = n_cities; // one thread per city
    int blocks_worker = (n_ants); // one block per ant

    int all_threads_pheromone = n_ants * n_ants;
    int threads_pheromone = std::min(N_MAX_THREADS_PER_BLOCK, all_threads_pheromone);
    int blocks_pheromone = (all_threads_pheromone + threads_pheromone - 1) / threads_pheromone;

    init_rng<<<1, n_ants>>>(d_states, seed); // one RNG per ant
    cudaDeviceSynchronize();

    // Host buffers to fetch data back
    std::vector<int> tours_host(m * n_cities);
    std::vector<float> choice_info_host(n_cities * n_cities);
    std::vector<float> tour_lengths_host(m);

    cudaEvent_t start_kernel, end_kernel;
    cudaEvent_t start_pheromone, end_pheromone;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    cudaEventCreate(&start_pheromone);
    cudaEventCreate(&end_pheromone);

    for (int iter = 0; iter < num_iter; ++iter) {
        cudaEventRecord(start_kernel);

        queenAntKernel<<<m, n_cities>>>(
            d_choice_info,
            d_distances,
            d_tours,
            d_tour_lengths,
            n_cities,
            d_states
        );
        cudaDeviceSynchronize();

        cudaEventRecord(end_kernel);
        cudaEventSynchronize(end_kernel);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, start_kernel, end_kernel);
        total_kernel += kernel_ms;

        cudaEventRecord(start_pheromone);

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

        cudaEventRecord(end_pheromone);
        cudaEventSynchronize(end_pheromone);
        float pheromone_ms = 0.0f;
        cudaEventElapsedTime(&pheromone_ms, start_pheromone, end_pheromone);
        total_pheromone += pheromone_ms;
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

    cudaEventRecord(end_total);
    cudaEventSynchronize(end_total);
    float total_time_ms = 0.0f;
    cudaEventElapsedTime(&total_time_ms, start_total, end_total);

    cudaEventDestroy(start_kernel);
    cudaEventDestroy(end_kernel);
    cudaEventDestroy(start_pheromone);
    cudaEventDestroy(end_pheromone);
    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);

    std::cout << "Total kernel time: " << total_kernel / 1000.0f << " seconds" << std::endl;
    std::cout << "Total pheromone update time: " << total_pheromone / 1000.0f << " seconds" << std::endl;
    std::cout << "Average kernel time: " << total_kernel / num_iter << " ms" << std::endl;
    std::cout << "Average pheromone kernel time: " << total_pheromone / num_iter << " ms" << std::endl;

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

void queen(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file) {
    std::cout << "Running QUEEN WORKER algorithm with CUDA + Graphs...\n";

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
    size_t tour_lengths_size = m * sizeof(float);

    std::vector<float> distances_host(n_cities * n_cities);
    for (int i = 0; i < n_cities; ++i) {
        for (int j = 0; j < n_cities; ++j) {
            distances_host[i * n_cities + j] = graph[i][j];
        }
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

    int n_ants = n_cities;

    int thread_worker_count = n_cities; // one thread per city
    int blocks_worker = (n_ants); // one block per ant

    int all_threads_pheromone = n_ants * n_ants;
    int threads_pheromone = std::min(N_MAX_THREADS_PER_BLOCK, all_threads_pheromone);
    int blocks_pheromone = (all_threads_pheromone + threads_pheromone - 1) / threads_pheromone;

    init_rng<<<1, n_ants>>>(d_states, seed);
    cudaDeviceSynchronize();

    // Host buffers
    std::vector<int> tours_host(m * n_cities);
    std::vector<float> choice_info_host(n_cities * n_cities);
    std::vector<float> tour_lengths_host(m);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph_capture;
    cudaGraphExec_t graph_exec;

    // Start capturing the kernel and pheromone updates into a graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    queenAntKernel<<<m, n_cities, 0, stream>>>(
        d_choice_info,
        d_distances,
        d_tours,
        d_tour_lengths,
        n_cities,
        d_states
    );

    pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone, 0, stream>>>(
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

    cudaStreamEndCapture(stream, &graph_capture);
    cudaGraphInstantiate(&graph_exec, graph_capture, NULL, NULL, 0);

    runGraphIterations(graph_exec, stream, num_iter, total_kernel);


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

    // Cleanup
    cudaFree(d_pheromone);
    cudaFree(d_choice_info);
    cudaFree(d_distances);
    cudaFree(d_tours);
    cudaFree(d_tour_lengths);
    cudaFree(d_states);

    cudaGraphDestroy(graph_capture);
    cudaGraphExecDestroy(graph_exec);
    cudaStreamDestroy(stream);

    cudaEventRecord(end_total);
    cudaEventSynchronize(end_total);

    float total_time_ms = 0.0f;
    cudaEventElapsedTime(&total_time_ms, start_total, end_total);

    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);

    generate_output(total_kernel, num_iter, total_time_ms, output_file, tours_host, best_id, best, n_cities);
}
