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

#define N_MAX_THREADS_PER_BLOCK 256

__global__ void workerAntKernel(
    int m, int n_cities,
    int* tours,
    float* choice_info,
    float* selection_prob_all,
    float* tour_lengths,
    float* distances,
    curandState* states
) {
    extern __shared__ unsigned int shared_visited[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) return;

    const int ints_per_ant = (n_cities + 31) / 32;
    unsigned int* my_visited = &shared_visited[threadIdx.x * ints_per_ant];
    int offset = tid * n_cities;

    curandState localState = states[tid];

    for (int i = 0; i < ints_per_ant; i++) {
        my_visited[i] = 0;
    }

    int step = 0;
    int current_city = 0;
    tours[offset + step] = current_city;

    {
        int idx = current_city / 32;
        int bit = current_city % 32;
        my_visited[idx] |= (1U << bit);
    }

    float tour_len = 0.0f;
    step++;

    while (step < n_cities) {
        float sum_probs = 0.0f;
        for (int j = 0; j < n_cities; j++) {
            int idx = j / 32;
            int bit = j % 32;
            bool is_visited = (my_visited[idx] >> bit) & 1U;

            if (is_visited) {
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

        {
            int idx = next_city / 32;
            int bit = next_city % 32;
            my_visited[idx] |= (1U << bit);
        }

        tour_len += distances[current_city * n_cities + next_city];

        current_city = next_city;
        step++;
    }

    tour_len += distances[current_city * n_cities + tours[offset]];
    tour_lengths[tid] = tour_len;

    states[tid] = localState;
}


void worker(const std::vector<std::vector<float>>& graph_constructed, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file) {
    // std::cout << "Running WORKER algorithm with CUDA Graphs...\n";

    cudaEvent_t start_total, end_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&end_total);
    cudaEventRecord(start_total);

    float total_kernel = 0.0f;

    int n_cities = graph_constructed.size();
    int m = n_cities;
    float Q = 1.0f;

    size_t matrix_size = n_cities * n_cities * sizeof(float);
    size_t array_size = m * n_cities * sizeof(int);
    size_t float_array_size = m * n_cities * sizeof(float);
    size_t tour_lengths_size = m * sizeof(float);

    std::vector<float> distances_host(n_cities * n_cities);
    for (int i = 0; i < n_cities; ++i) {
        for (int j = 0; j < n_cities; ++j) {
            distances_host[i * n_cities + j] = graph_constructed[i][j];
        }
    }

    float *d_pheromone, *d_choice_info, *d_distances, *d_selection_prob_all, *d_tour_lengths;
    int *d_tours;
    curandState* d_states;

    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    cudaMalloc(&d_tours, array_size);
    cudaMalloc(&d_selection_prob_all, float_array_size);
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

    pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone>>>(
        alpha, beta, evaporate, Q,
        d_pheromone, d_tours, n_cities, m,
        d_choice_info, d_distances, d_tour_lengths
    );
    cudaDeviceSynchronize();

    const int ints_per_ant = (n_cities + 31) / 32;
    const size_t shared_memory_size = thread_worker_count * ints_per_ant * sizeof(unsigned int);

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    workerAntKernel<<<blocks_worker, thread_worker_count, shared_memory_size, stream>>>(
        m, n_cities, d_tours, d_choice_info, d_selection_prob_all,
        d_tour_lengths, d_distances, d_states
    );

    pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone, 0, stream>>>(
        alpha, beta, evaporate, Q,
        d_pheromone, d_tours, n_cities, m,
        d_choice_info, d_distances, d_tour_lengths
    );

    // pheromoneUpdateKernelBasic<<<blocks_worker, thread_worker_count, 0, stream>>>(
    //     alpha, beta, evaporate, Q,
    //     d_pheromone, d_tours, n_cities, m,
    //     d_choice_info, d_distances, d_tour_lengths
    // );

    // pheromoneEvaporationAndChoiceInfoKernel<<<blocks_pheromone, threads_pheromone, 0, stream>>>(
    //      alpha,
    //      beta,
    //      evaporate,
    //      d_pheromone,
    //      d_choice_info,
    //      d_distances,
    //      n_cities
    // );


    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

    runGraphIterations(graph_exec, stream, num_iter, total_kernel);


    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);
    cudaStreamDestroy(stream);

    std::vector<int> tours_host(m * n_cities);
    std::vector<float> choice_info_host(n_cities * n_cities);
    std::vector<float> tour_lengths_host(m);

    cudaMemcpy(tours_host.data(), d_tours, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(tour_lengths_host.data(), d_tour_lengths, tour_lengths_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(choice_info_host.data(), d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(initial_pheromone.data(), d_pheromone, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_pheromone);
    cudaFree(d_choice_info);
    cudaFree(d_distances);
    cudaFree(d_tours);
    cudaFree(d_selection_prob_all);
    cudaFree(d_tour_lengths);
    cudaFree(d_states);

    cudaEventRecord(end_total);
    cudaEventSynchronize(end_total);

    float total_time = 0.0f;
    cudaEventElapsedTime(&total_time, start_total, end_total);

    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);


    generate_output(total_kernel, num_iter, total_time, output_file, tours_host, n_cities, tour_lengths_host);
}

void worker_no_graph(const std::vector<std::vector<float>>& graph_constructed, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file) {
    // std::cout << "Running WORKER NO GRAPH algorithm with CUDA...\n";

    cudaEvent_t start_total, end_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&end_total);
    cudaEventRecord(start_total);

    float total_kernel = 0.0f;

    int n_cities = graph_constructed.size();
    int m = n_cities;
    float Q = 1.0f;

    size_t matrix_size = n_cities * n_cities * sizeof(float);
    size_t array_size = m * n_cities * sizeof(int);
    size_t float_array_size = m * n_cities * sizeof(float);
    size_t tour_lengths_size = m * sizeof(float);

    std::vector<float> distances_host(n_cities * n_cities);
    for (int i = 0; i < n_cities; ++i) {
        for (int j = 0; j < n_cities; ++j) {
            distances_host[i * n_cities + j] = graph_constructed[i][j];
        }
    }

    float *d_pheromone, *d_choice_info, *d_distances, *d_selection_prob_all, *d_tour_lengths;
    int *d_tours;
    curandState* d_states;

    cudaMalloc(&d_pheromone, matrix_size);
    cudaMalloc(&d_choice_info, matrix_size);
    cudaMalloc(&d_distances, matrix_size);
    cudaMalloc(&d_tours, array_size);
    cudaMalloc(&d_selection_prob_all, float_array_size);
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

    const int ints_per_ant = (n_cities + 31) / 32;
    const size_t shared_memory_size = thread_worker_count * ints_per_ant * sizeof(unsigned int);

    for (int iter = 0; iter < num_iter; ++iter) {
        cudaEventRecord(start_kernel);
        workerAntKernel<<<blocks_worker, thread_worker_count, shared_memory_size>>>(
            m, n_cities, d_tours, d_choice_info, d_selection_prob_all,
            d_tour_lengths, d_distances, d_states
        );
        cudaDeviceSynchronize();

        pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone>>>(
            alpha, beta, evaporate, Q,
            d_pheromone, d_tours, n_cities, m,
            d_choice_info, d_distances, d_tour_lengths
        );
        cudaDeviceSynchronize();
        cudaEventRecord(end_kernel);
        cudaEventSynchronize(end_kernel);

        float kernel_time = 0.0f;
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
        total_kernel += kernel_time;

    }

    cudaMemcpy(tours_host.data(), d_tours, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(tour_lengths_host.data(), d_tour_lengths, tour_lengths_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(choice_info_host.data(), d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(initial_pheromone.data(), d_pheromone, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_pheromone);
    cudaFree(d_choice_info);
    cudaFree(d_distances);
    cudaFree(d_tours);
    cudaFree(d_selection_prob_all);
    cudaFree(d_tour_lengths);
    cudaFree(d_states);

    cudaEventRecord(end_total);
    cudaEventSynchronize(end_total);

    float total_time = 0.0f;
    cudaEventElapsedTime(&total_time, start_total, end_total);

    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);
    cudaEventDestroy(start_pheromone);
    cudaEventDestroy(end_pheromone);

    generate_output(total_kernel, num_iter, total_time, output_file, tours_host, n_cities, tour_lengths_host);
}