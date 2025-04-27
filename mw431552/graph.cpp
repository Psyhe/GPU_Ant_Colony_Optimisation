#include "graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

struct City {
    float x;
    float y;
};

float euclidean_distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

float ceil_distance(float x1, float y1, float x2, float y2) {
    return ceil(euclidean_distance(x1, y1, x2, y2));
}

float geo_distance(float lat1, float lon1, float lat2, float lon2) {
    const float RRR = 6378.388;

    auto to_radians = [](float x) -> float {
        int deg = int(x);
        float min = x - deg;
        return M_PI * (deg + 5.0 * min / 3.0) / 180.0;
    };

    float latitude1 = to_radians(lat1);
    float longitude1 = to_radians(lon1);
    float latitude2 = to_radians(lat2);
    float longitude2 = to_radians(lon2);

    float q1 = cos(longitude1 - longitude2);
    float q2 = cos(latitude1 - latitude2);
    float q3 = cos(latitude1 + latitude2);

    float dij = int(RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
    return dij;
}

std::vector<std::vector<float>> read_input_file(const std::string& filename) {
    using namespace std;
    
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;
    int dimension = 0;
    string edge_weight_type;
    
    // Parse metadata
    for (int i = 0; i < 6; ++i) {
        getline(infile, line);
        if (line.find("DIMENSION") != string::npos) {
            size_t pos = line.find(":");
            if (pos != string::npos) {
                string dim_str = line.substr(pos + 1);
                dimension = stoi(dim_str);
            }
        }
        if (line.find("EDGE_WEIGHT_TYPE") != string::npos) {
            size_t pos = line.find(":");
            if (pos != string::npos) {
                edge_weight_type = line.substr(pos + 1);
                // Remove potential leading spaces
                edge_weight_type.erase(0, edge_weight_type.find_first_not_of(" \t"));
            }
            if (edge_weight_type != "EUC_2D" && edge_weight_type != "CEIL_2D" && edge_weight_type != "GEO") {
                cerr << edge_weight_type << endl;
                cerr << "Unsupported EDGE_WEIGHT_TYPE: " << edge_weight_type << endl;
                exit(1);
            }
        }
    }

    vector<City> cities(dimension);

    // Read city coordinates
    for (int i = 0; i < dimension; ++i) {
        getline(infile, line);
        istringstream iss(line);
        int index;
        float x, y;
        iss >> index >> x >> y;
        cities[index - 1] = {x, y};
    }

    infile.close();

    // Build distance matrix
    vector<vector<float>> distance_matrix(dimension, vector<float>(dimension, 0.0f));

    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            if (i == j) continue;

            float dist = 0.0;
            if (edge_weight_type == "EUC_2D") {
                dist = euclidean_distance(cities[i].x, cities[i].y, cities[j].x, cities[j].y);
            } else if (edge_weight_type == "CEIL_2D") {
                dist = ceil_distance(cities[i].x, cities[i].y, cities[j].x, cities[j].y);
            } else if (edge_weight_type == "GEO") {
                dist = geo_distance(cities[i].x, cities[i].y, cities[j].x, cities[j].y);
            }
            distance_matrix[i][j] = dist;
        }
    }

    return distance_matrix;
}

void print_graph(const std::vector<std::vector<float>>& graph) {
    using namespace std;

    int n = graph.size();
    cout << fixed << setprecision(2);

    cout << "Distance matrix (" << n << " x " << n << "):\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << graph[i][j] << "\t";
        }
        cout << endl;
    }
}
