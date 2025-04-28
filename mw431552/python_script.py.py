import subprocess
import re
import numpy as np
import sys

# Check for input parameter
if len(sys.argv) != 2:
    print("Usage: python script.py <test_name>")
    print("Example: python script.py a280")
    sys.exit(1)

test_name = sys.argv[1]  # Get the test name from command line argument

# Configuration
x_values = [8, 32, 42, 64, 80]
command_base = "./acotsp ../input/{}.tsp output.txt WORKER 1000 1 2 0.5 {{}}".format(test_name)

# The four sections to parse
sections = [
    ("WORKER", "Running WORKER algorithm with CUDA Graphs..."),
    ("WORKER_NO_GRAPH", "Running WORKER NO GRAPH algorithm with CUDA..."),
    ("QUEEN", "Running QUEEN algorithm with CUDA + Graphs..."),
    ("QUEEN_NO_GRAPH", "Running QUEEN NO GRAPH algorithm with CUDA...")
]

# Function to run command and capture output
def run_command(x_value):
    cmd = command_base.format(x_value)
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout

# Function to parse metrics from a specific section
def parse_section(output, header):
    pattern = re.escape(header) + r"(.*?)(Running|$)"  # Capture until next Running or end
    match = re.search(pattern, output, re.DOTALL)
    if not match:
        print(f"Warning: Section {header} not found.")
        return None, None, None

    section_text = match.group(1)

    avg_graph_time_match = re.search(r"Average graph execution time:\s*([\d\.]+)\s*ms", section_text)
    total_time_match = re.search(r"Total time:\s*([\d\.]+)\s*seconds", section_text)
    best_tour_match = re.search(r"Best tour length:\s*([\d\.]+)", section_text)

    avg_graph_time = float(avg_graph_time_match.group(1)) if avg_graph_time_match else None
    total_time = float(total_time_match.group(1)) if total_time_match else None
    best_tour_length = float(best_tour_match.group(1)) if best_tour_match else None

    return avg_graph_time, total_time, best_tour_length

# Data storage
results = {alg[0]: {'avg_graph_times': [], 'total_times': [], 'best_tour_lengths': []} for alg in sections}

# Main loop
for x in x_values:
    output = run_command(x)
    for alg_name, header in sections:
        avg_graph_time, total_time, best_tour_length = parse_section(output, header)

        if avg_graph_time is not None:
            results[alg_name]['avg_graph_times'].append(avg_graph_time)
        if total_time is not None:
            results[alg_name]['total_times'].append(total_time)
        if best_tour_length is not None:
            results[alg_name]['best_tour_lengths'].append(best_tour_length)

# Final analysis
for alg_name in results:
    print(f"\n=== Results for {alg_name} ===")
    data = results[alg_name]

    if data['avg_graph_times']:
        print(f"Average graph execution time: mean = {np.mean(data['avg_graph_times']):.4f} ms, std = {np.std(data['avg_graph_times']):.4f} ms")
    if data['total_times']:
        print(f"Total time: mean = {np.mean(data['total_times']):.4f} s, std = {np.std(data['total_times']):.4f} s")
    if data['best_tour_lengths']:
        print(f"Best tour length: mean = {np.mean(data['best_tour_lengths']):.4f}, std = {np.std(data['best_tour_lengths']):.4f}")
