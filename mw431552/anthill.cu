// #include <cuda_runtime.h>
// #include <curand_kernel.h>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <iomanip> // for better formatting
// #include <fstream>
// #include <chrono>

// #define N_MAX_THREADS_PER_BLOCK 1024
// #define N_CITIES 1024
// #define QUEEN "QUEEN"
// #define WORKER "WORKER"


// std::string prepare_output_path(const std::string& output_file) {
//     if (output_file.find('/') == std::string::npos && output_file.find('\\') == std::string::npos) {
//         return "./" + output_file;
//     } else {
//         return output_file;
//     }
// }

// void generate_output(float total_kernel, int num_iter, float total_time_ms, std::string output_file, float *tours_host, int best_id, float best) {
//     std::cout << "Total kernel+pheromone time: " << total_kernel / 1000.0f << " seconds" << std::endl;
//     std::cout << "Average graph execution time: " << total_kernel / num_iter << " ms" << std::endl;
//     std::cout << "Total time: " << total_time_ms / 1000.0f << " seconds" << std::endl;

//     std::string output_path = prepare_output_path(output_file);
//     std::ofstream out(output_path);

//     if (!out.is_open()) {
//         std::cerr << "Failed to open output file: " << output_path << std::endl;
//         return;
//     }

//     std::cout << "\nBest tour length: " << best << std::endl;
//     out << "Best tour length: " << best << std::endl;

//     for (int step = 0; step < n_cities; ++step) {
//         std::cout << tours_host[best_id * n_cities + step] << " ";
//         out << tours_host[best_id * n_cities + step] + 1 << " ";
//     }
//     std::cout << std::endl;
//     out << std::endl;

//     out.close();
// }

// __global__ void pheromoneUpdateKernelBasic(
//     float alpha,
//     float beta,
//     float evaporation_rate,
//     float Q,
//     float *pheromone,
//     int *tours,
//     int n_cities,
//     int m,
//     float *choice_info,
//     float *distances,
//     float *tour_lengths
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= m) return;

//     for (int i = 0; i < n_cities; i++) {
//         pheromone[tid * n_cities + i] *= (1.0f - evaporation_rate);
//     }

//     float tour_len = tour_lengths[tid];

//     for (int i = 0; i < n_cities-1; i++) {
//         int current_city = tours[tid * n_cities + i];
//         int next_city = tours[tid * n_cities + i + 1];

//         pheromone[next_city * n_cities + current_city] += Q / tour_len;
//         pheromone[current_city * n_cities + next_city] += Q / tour_len;
//     }

//     // Add return to starting city
//     int current_city = tours[tid * n_cities + n_cities-1];
//     int start_city = tours[tid * n_cities];

//     pheromone[current_city * n_cities + start_city] += Q / tour_len;
//     pheromone[start_city * n_cities + current_city] += Q / tour_len;

    
//     // wszystkie watki musza zostawic swoje feromony
//     __syncthreads();

//     for (int i = 0; i < n_cities; i++) {
//         float tau = __powf(pheromone[tid * n_cities + i], alpha);
//         float eta = __powf(1.0f / distances[tid * n_cities + i], beta);
//         choice_info[tid * n_cities + i] = tau * eta;
//     }
// }


// __global__ void pheromoneUpdateKernel(
//     float alpha,
//     float beta,
//     float evaporation_rate,
//     float Q,
//     float *pheromone,
//     int *tours,
//     int n_cities,
//     int m,
//     float *choice_info,
//     float *distances,
//     float *tour_lengths
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= n_cities * n_cities) return;

//     int local_X = tid % n_cities;
//     int local_Y = tid / n_cities;

//     pheromone[tid] *= (1.0f - evaporation_rate);

//     float pheromone_update_value = 0.0f;

//     for (int i = 0; i < m; i++) {
//         int offset = i * n_cities;
//         for (int j = 0; j < n_cities - 1; j++) {
//             if ((tours[offset + j] == local_X && tours[offset + j + 1] == local_Y) ||
//                 (tours[offset + j] == local_Y && tours[offset + j + 1] == local_X))  {
//                 pheromone_update_value += Q / tour_lengths[i];
//             }
//         }
//         if ((tours[offset + n_cities - 1] == local_X && tours[offset] == local_Y) ||
//             (tours[offset + n_cities - 1] == local_Y && tours[offset] == local_X)) {
//             pheromone_update_value += Q / tour_lengths[i];
//         }
//     }

//     pheromone[tid] += pheromone_update_value;

//     if (distances[local_X * n_cities + local_Y] > 0.0f) {
//         float tau = __powf(pheromone[tid], alpha);
//         float eta = __powf(1.0f / distances[local_X * n_cities + local_Y], beta);
//         choice_info[tid] = tau * eta;
//     } else {
//         choice_info[tid] = 0.0f;
//     }
// }

// __global__ void init_rng(curandState* states, unsigned long seed) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     curand_init(seed, idx, 0, &states[idx]);
// }

// __global__ void queenAntKernelOptimized(float *choice_info, float *distances, int *tours, float *tour_lengths, int n_cities, curandState *states) {

//     __shared__ int tabu[N_CITIES];
//     __shared__ float probabilities[N_CITIES];
//     __shared__ int current_city;
    
//     int tid = threadIdx.x;
    
//     if (tid >= n_cities)
//         return;

//     int queen_id = blockIdx.x;
//     int n_threads = blockDim.x;

//     int *tour = &tours[queen_id * n_cities];
//     curandState localState = states[queen_id];
    
//     tabu[tid] = 1; // Not visited yet
    
//     __syncthreads();

//     float tour_len = 0.0f;

//     int start = queen_id % n_cities;
//     if (tid == 0) {
//         tour[0] = start;
//         tabu[start] = 0; // Mark start city as visited
//     }
//     __syncthreads();

//     current_city = start;

//     for (int step = 1; step < n_cities; step++) {
//         probabilities[tid] = choice_info[current_city * n_cities + tid] * tabu[tid];
        
//         __syncthreads();

//         // Warp-level reduction to compute total probability
//         float local_prob = probabilities[tid];
        
//         // Use warp shuffle reduction
//         for (int offset = 16; offset > 0; offset /= 2) {
//             local_prob += __shfl_down_sync(0xffffffff, local_prob, offset);
//         }

//         __shared__ float total;
//         if (tid % 32 == 0) { // Only one thread per warp writes
//             atomicAdd(&total, local_prob);
//         }
//         __syncthreads();

//         if (tid == 0) {
//             double r = curand_uniform(&localState) * total;
//             double cumulative = 0.0;
//             int next_city = -1;
//             for (int i = 0; i < n_cities; i++) {
//                 cumulative += probabilities[i];
//                 if (cumulative >= r) {
//                     next_city = i;
//                     break;
//                 }
//             }
//             if (next_city == -1) {
//                 // fallback
//                 for (int i = 0; i < n_cities; i++) {
//                     if (tabu[i]) {
//                         next_city = i;
//                         break;
//                     }
//                 }
//             }
//             tour[step] = next_city;
//             tabu[next_city] = 0; // mark as visited
//             tour_len += distances[current_city * n_cities + next_city];
//             current_city = next_city;
//         }
//         __syncthreads();
//     }

//     if (tid == 0) {
//         tour_len += distances[current_city * n_cities + tour[0]]; // Assuming you want a full tour
//         tour_lengths[queen_id] = tour_len;
//         states[queen_id] = localState;
//     }
// }


// __global__ void queenAntKernel(float *choice_info, float *distances, int *tours, float *tour_lengths, int n_cities, curandState *states) {

//     __shared__ int tabu[N_CITIES];
//     __shared__ float probabilities[N_CITIES];
//     __shared__ int current_city;
    
//     int tid = threadIdx.x;
    
//     if (tid >= n_cities)
//         return;

//     int queen_id = blockIdx.x;
//     int n_threads = blockDim.x;

//     int *tour = &tours[queen_id * (n_cities )];
//     curandState localState = states[queen_id];
    
//     tabu[tid] = 1; // Not visited yet
    
//     __syncthreads();

//     float tour_len = 0.0f;

//     int start = queen_id % n_cities;
//     if (tid == 0) {
//         tour[0] = start;
//         tabu[start] = 0; // Mark start city as visited
//     }
//     __syncthreads();

//     current_city = start;

//     for (int step = 1; step < n_cities; step++) {
//         probabilities[tid] = choice_info[current_city * n_cities + tid] * tabu[tid];

//         __syncthreads();

//         // Thread 0 does roulette wheel selection
//         double total = 0.0;
//         if (tid == 0) {
//             for (int i = 0; i < n_cities; i++) {
//                 total += probabilities[i];
//             }
//             double r = curand_uniform(&localState) * total;
//             double cumulative = 0.0;
//             int next_city = -1;
//             for (int i = 0; i < n_cities; i++) {
//                 cumulative += probabilities[i];
//                 if (cumulative >= r) {
//                     next_city = i;
//                     break;
//                 }
//             }
//             if (next_city == -1) {
//                 // fallback
//                 for (int i = 0; i < n_cities; i++) {
//                     if (tabu[i]) {
//                         next_city = i;
//                         break;
//                     }
//                 }
//             }
//             tour[step] = next_city;
//             tabu[next_city] = 0; // mark as visited
//             tour_len += distances[current_city * n_cities + next_city];
//             current_city = next_city;
//         }
//         __syncthreads();
//     }

//     if (tid == 0) {
//         tour_len += distances[n_cities * queen_id + current_city];
//         tour_lengths[queen_id] = tour_len;
//         states[queen_id] = localState;
//     }
// }

// _global__ void workerAntKernel(
//     int m, int n_cities,
//     int* tours,
//     float* choice_info,
//     float* selection_prob_all,
//     bool* visited_global, // unused now
//     float* tour_lengths,
//     float* distances,
//     curandState* states
// ) {
//     extern __shared__ unsigned int shared_visited[];
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= m) return;

//     const int ints_per_ant = (n_cities + 31) / 32;
//     unsigned int* my_visited = &shared_visited[threadIdx.x * ints_per_ant];
//     int offset = tid * n_cities;

//     curandState localState = states[tid];

//     // Initialize visited bits
//     for (int i = 0; i < ints_per_ant; i++) {
//         my_visited[i] = 0;
//     }

//     int step = 0;
//     int current_city = 0;
//     tours[offset + step] = current_city;

//     // Mark starting city visited
//     {
//         int idx = current_city / 32;
//         int bit = current_city % 32;
//         my_visited[idx] |= (1U << bit);
//     }

//     float tour_len = 0.0f;
//     step++;

//     while (step < n_cities) {
//         float sum_probs = 0.0f;
//         for (int j = 0; j < n_cities; j++) {
//             int idx = j / 32;
//             int bit = j % 32;
//             bool is_visited = (my_visited[idx] >> bit) & 1U;

//             if (is_visited) {
//                 selection_prob_all[offset + j] = 0.0f;
//             } else {
//                 float prob = choice_info[current_city * n_cities + j];
//                 selection_prob_all[offset + j] = prob;
//                 sum_probs += prob;
//             }
//         }

//         if (sum_probs == 0.0f) break;

//         float r = curand_uniform(&localState) * sum_probs;
//         float cumulative_prob = 0.0f;
//         int next_city = -1;

//         for (int j = 0; j < n_cities; j++) {
//             cumulative_prob += selection_prob_all[offset + j];
//             if (cumulative_prob >= r) {
//                 next_city = j;
//                 break;
//             }
//         }

//         tours[offset + step] = next_city;

//         // Mark next city visited
//         {
//             int idx = next_city / 32;
//             int bit = next_city % 32;
//             my_visited[idx] |= (1U << bit);
//         }

//         tour_len += distances[current_city * n_cities + next_city];

//         current_city = next_city;
//         step++;
//     }

//     tour_len += distances[current_city * n_cities + tours[offset]];
//     tour_lengths[tid] = tour_len;

//     states[tid] = localState;
// }











// void ant(const std::vector<std::vector<float>>& graph, int num_iter, float alpha, float beta, float evaporate, int seed, std::string output_file, std::string type) {
//     cudaEvent_t start_total, end_total;
//     cudaEventCreate(&start_total);
//     cudaEventCreate(&end_total);
//     cudaEventRecord(start_total);

//     float total_kernel = 0.0f;
//     float total_pheromone = 0.0f;

//     int n_cities = graph.size();
//     int m = n_cities; // number of ants = number of cities
//     float Q = 1.0f;

//     size_t matrix_size = n_cities * n_cities * sizeof(float);
//     size_t array_size = m * n_cities * sizeof(int);
//     size_t tour_lengths_size = m * sizeof(float);

//     // Host distances matrix
//     std::vector<float> distances_host(n_cities * n_cities);
//     for (int i = 0; i < n_cities; ++i) {
//         for (int j = 0; j < n_cities; ++j) {
//             distances_host[i * n_cities + j] = graph[i][j];
//         }
//     }

//     // Device memory
//     float *d_pheromone, *d_choice_info, *d_distances, *d_tour_lengths;
//     int *d_tours;
//     curandState* d_states;

//     cudaMalloc(&d_pheromone, matrix_size);
//     cudaMalloc(&d_choice_info, matrix_size);
//     cudaMalloc(&d_distances, matrix_size);
//     cudaMalloc(&d_tours, array_size);
//     cudaMalloc(&d_tour_lengths, tour_lengths_size);
//     cudaMalloc(&d_states, m * sizeof(curandState));

//     cudaMemcpy(d_distances, distances_host.data(), matrix_size, cudaMemcpyHostToDevice);
//     std::vector<float> initial_pheromone(n_cities * n_cities, 1.0f);
//     cudaMemcpy(d_pheromone, initial_pheromone.data(), matrix_size, cudaMemcpyHostToDevice);

//     int n_ants = n_cities;

//     int thread_worker_count = n_cities; // one thread per city
//     int blocks_worker = (n_ants); // one block per ant

//     int all_threads_pheromone = n_ants * n_ants;
//     int threads_pheromone = std::min(N_MAX_THREADS_PER_BLOCK, all_threads_pheromone);
//     int blocks_pheromone = (all_threads_pheromone + threads_pheromone - 1) / threads_pheromone;

//     init_rng<<<1, n_ants>>>(d_states, seed);
//     cudaDeviceSynchronize();

//     // Host buffers
//     std::vector<int> tours_host(m * n_cities);
//     std::vector<float> choice_info_host(n_cities * n_cities);
//     std::vector<float> tour_lengths_host(m);

//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     cudaGraph_t graph_capture;
//     cudaGraphExec_t graph_exec;

//     // Start capturing the kernel and pheromone updates into a graph
//     cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

//     queenAntKernel<<<m, n_cities, 0, stream>>>(
//         d_choice_info,
//         d_distances,
//         d_tours,
//         d_tour_lengths,
//         n_cities,
//         d_states
//     );

//     pheromoneUpdateKernel<<<blocks_pheromone, threads_pheromone, 0, stream>>>(
//         alpha,
//         beta,
//         evaporate,
//         Q,
//         d_pheromone,
//         d_tours,
//         n_cities,
//         m,
//         d_choice_info,
//         d_distances,
//         d_tour_lengths
//     );

//     cudaStreamEndCapture(stream, &graph_capture);
//     cudaGraphInstantiate(&graph_exec, graph_capture, NULL, NULL, 0);

//     // Events to time things
//     cudaEvent_t start_kernel, end_kernel;
//     cudaEventCreate(&start_kernel);
//     cudaEventCreate(&end_kernel);

//     for (int iter = 0; iter < num_iter; ++iter) {
//         cudaEventRecord(start_kernel, stream);

//         cudaGraphLaunch(graph_exec, stream);
//         cudaStreamSynchronize(stream);

//         cudaEventRecord(end_kernel, stream);
//         cudaEventSynchronize(end_kernel);

//         float iter_time = 0.0f;
//         cudaEventElapsedTime(&iter_time, start_kernel, end_kernel);

//         total_kernel += iter_time;
//     }

//     // Fetch results
//     cudaMemcpy(tours_host.data(), d_tours, array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(tour_lengths_host.data(), d_tour_lengths, tour_lengths_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(choice_info_host.data(), d_choice_info, matrix_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(initial_pheromone.data(), d_pheromone, matrix_size, cudaMemcpyDeviceToHost);

//     float best = 1e9;
//     int best_id = 0;
//     for (int i = 0; i < m; ++i) {
//         if (tour_lengths_host[i] < best) {
//             best = tour_lengths_host[i];
//             best_id = i;
//         }
//     }

//     // Cleanup
//     cudaFree(d_pheromone);
//     cudaFree(d_choice_info);
//     cudaFree(d_distances);
//     cudaFree(d_tours);
//     cudaFree(d_tour_lengths);
//     cudaFree(d_states);

//     cudaGraphDestroy(graph_capture);
//     cudaGraphExecDestroy(graph_exec);
//     cudaStreamDestroy(stream);

//     cudaEventDestroy(start_kernel);
//     cudaEventDestroy(end_kernel);

//     cudaEventRecord(end_total);
//     cudaEventSynchronize(end_total);

//     float total_time_ms = 0.0f;
//     cudaEventElapsedTime(&total_time_ms, start_total, end_total);

//     cudaEventDestroy(start_total);
//     cudaEventDestroy(end_total);

//     generate_output(total_kernel, num_iter, total_time_ms, output_file, tours_host, best_id, best);
// }
