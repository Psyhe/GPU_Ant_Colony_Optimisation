import math

# Step 1: Read coordinates from d198.tscp
coordinates = {}

with open('d198.tsp', 'r') as file:
    reading_coords = False
    for line in file:
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            reading_coords = True
            continue
        if line == "EOF":
            break
        if reading_coords:
            parts = line.split()
            node = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coordinates[node] = (x, y)

# Step 2: The tour you provided
tour = [
    1, 2, 7, 6, 3, 4, 5, 8, 9, 10, 11, 12, 13, 41, 40, 15, 14, 17, 16, 18, 19, 20, 21, 24,
    22, 23, 29, 30, 28, 27, 26, 25, 35, 51, 50, 48, 49, 63, 79, 80, 91, 92, 102, 93, 94, 95,
    101, 96, 90, 89, 97, 98, 88, 83, 82, 75, 62, 53, 46, 52, 65, 81, 76, 77, 78, 64, 103, 104,
    105, 114, 113, 112, 106, 107, 111, 110, 109, 108, 120, 119, 118, 117, 116, 115, 122, 121,
    123, 139, 154, 155, 156, 157, 158, 159, 160, 161, 162, 152, 153, 148, 147, 142, 143, 141,
    140, 134, 132, 133, 130, 131, 126, 125, 169, 124, 138, 145, 150, 149, 146, 136, 135, 144,
    129, 127, 170, 128, 171, 172, 166, 165, 164, 163, 151, 137, 168, 167, 84, 87, 99, 100, 86,
    85, 71, 69, 58, 57, 42, 43, 56, 55, 60, 59, 68, 72, 73, 70, 67, 44, 38, 31, 32, 37, 36,
    33, 34, 39, 47, 66, 74, 61, 54, 45, 182, 189, 188, 191, 190, 193, 186, 187, 192, 195, 198,
    197, 196, 179, 185, 184, 180, 178, 194, 176, 173, 174, 175, 177, 181, 183
]

# Step 3: Calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

total_length = 0.0
for i in range(len(tour)):
    current_node = tour[i]
    next_node = tour[(i + 1) % len(tour)]  # wrap around to start
    total_length += euclidean_distance(coordinates[current_node], coordinates[next_node])

print(f"Total tour length: {total_length:.2f}")
