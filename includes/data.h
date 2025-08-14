#pragma once

#include <vector>
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>

void read_breastCancer(std::vector<std::vector<double>>& features, std::vector<double>& targets, std::string path);
void shuffle_data(std::vector<std::vector<double>>& X, std::vector<double>& y);