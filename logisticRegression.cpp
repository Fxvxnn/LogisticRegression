#include <iostream>
#include <vector>
#include <cmath>
#include "includes/data.h"

double TEST_SPLIT = 0.2;

double operator*(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be the same size for dot product.");
    }

    double out {0.0};
    for (size_t i = 0; i < a.size(); i++) {
        out += a[i] * b[i];
    }
    return out;
}


class logisticRegression
{
private:
    std::vector<double> theta;
    double lr;
public:
    logisticRegression(int feature_size, double _lr) : lr(_lr) {
        theta.resize(feature_size);
    }
    double predict(std::vector<double>& x) {
        return 1/(1+std::exp(- (theta*x)));
    }
    void update(std::vector<double>& x, double y) {
        for (size_t i=0; i<theta.size(); i++) {
            theta[i] += lr * (y-predict(x))*x[i];
        }
    }
    void SGD(std::vector<std::vector<double>>& features, std::vector<double>& targets) {
        for (size_t i=0; i<features.size(); i++) {
            update(features[i], targets[i]);
        }
    }
    void BGD(std::vector<std::vector<double>>& features, std::vector<double>& targets) {
        for (size_t j=0; j<theta.size(); j++) {
            double sum {0};
            for (size_t i=0; i<features.size(); i++) {
                sum += (targets[i]-predict(features[i])) * features[i][j];
            }
            theta[j] += lr * sum;
        }
    }

    double acc(std::vector<std::vector<double>>& features, std::vector<double>& targets) {
        int correct {0};
        double out;
        for (size_t i=0; i<features.size(); i++) {
            out = (predict(features[i]) >= 0.5) ? 1 : 0;
            if (out == targets[i]) correct++;
        }
        return (double) correct / features.size();
    }
};


void read_test(std::vector<std::vector<double>>& features, std::vector<double>& targets, std::string featurePath, std::string targetPath) {
    std::ifstream file(featurePath);

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> x {1};

        std::getline(ss, cell, ','); // Passenger ID
        

        std::getline(ss, cell, ','); // P-class
        x.push_back(std::stod(cell));

        std::getline(ss, cell, ','); // Name
        std::getline(ss, cell, ','); // Name

        std::getline(ss, cell, ','); // sex
        x.push_back((cell=="male"));

        std::getline(ss, cell, ','); // Age
        (cell != "") ? x.push_back(std::stod(cell)) : x.push_back(29.7); // if age is empty append mean

        for (int i=0; i<2; i++) { //SibSp,Parch
            std::getline(ss, cell, ',');
            x.push_back(std::stod(cell));
        }

        std::getline(ss, cell, ','); // Ticket        
        
        std::getline(ss, cell, ','); // Fare
        (cell != "") ? x.push_back(std::stod(cell)) : x.push_back(32.2); // if Fare is empty append mean

        features.push_back(x);
        
    }
    file.close();

    std::ifstream solFile(targetPath);
    std::getline(solFile, line);
    while (std::getline(solFile, line)) {
        std::stringstream ss(line);
        std::string cell;

        std::getline(ss, cell, ','); // Passenger ID
        
        std::getline(ss, cell, ','); // Survived
        targets.push_back(std::stod(cell));
    }

}

int main() {

    std::vector<std::vector<double>> features;
    std::vector<double> targets;

    read_breastCancer(features, targets, "data/data.csv");
    shuffle_data(features, targets);

    std::vector<std::vector<double>> test_features(features.end()-TEST_SPLIT * features.size(), features.end());
    std::vector<double> test_targets(targets.end()-TEST_SPLIT * targets.size(), targets.end());

    features.resize((1-TEST_SPLIT)*features.size());
    targets.resize((1-TEST_SPLIT)*targets.size());


    logisticRegression model(features[0].size(), 1e-4);

    for (int i=0; i<=100; i++) {
        model.SGD(features, targets);
        std::cout << model.acc(test_features, test_targets) << "  -----------  Iteration: " << i << std::endl;
    }

    std::cout << model.acc(test_features, test_targets) << std::endl;
    return 0;
}