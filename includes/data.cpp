#include "data.h"

void read_breastCancer(std::vector<std::vector<double>>& features, std::vector<double>& targets, std::string path) {
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> x;

        std::getline(ss, cell, ','); // ID
        
        std::getline(ss, cell, ','); // Malignant / benign
        targets.push_back((cell == "M"));

        for (int i=0; i<30; i++) {
            std::getline(ss, cell, ',');
            x.push_back(std::stod(cell));
        }
        features.push_back(x);
    }
    file.close();
}

void shuffle_data(std::vector<std::vector<double>>& X, std::vector<double>& y) {
    std::default_random_engine rng(static_cast<unsigned>(std::time(nullptr)));
    for (int i = X.size()-1; i>0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);

        std::swap(X[i], X[j]);
        std::swap(y[i], y[j]);
    }
}