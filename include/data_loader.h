//
// Created by josei on 1/12/2025.
//

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <random>
#include "include/tensor.h"
#include <iostream>

namespace utec::data {

using Tensor2 = utec::algebra::Tensor<double, 2>;
using shape_type = typename Tensor2::shape_type;

struct DatasetStats {
    std::vector<double> min_vals;
    std::vector<double> max_vals;
    std::vector<double> mean_vals;
    std::vector<double> std_vals;
};

class HeartDiseaseLoader {
private:
    std::vector<std::vector<double>> X_data_;
    std::vector<double> y_data_;
    DatasetStats stats_;
    size_t n_features_;

public:
    HeartDiseaseLoader() : n_features_(13) {}

    void load_csv(const std::string& filepath, bool has_header = true) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        std::string line;

        if (has_header) {
            std::getline(file, line);
        }

        X_data_.clear();
        y_data_.clear();

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string value;
            std::vector<double> row;

            while (std::getline(ss, value, ',')) {
                try {
                    row.push_back(std::stod(value));
                } catch (...) {
                    row.push_back(0.0);
                }
            }

            if (row.size() < n_features_ + 1) {
                continue;
            }

            std::vector<double> features(row.begin(), row.begin() + n_features_);

            double target = row[n_features_];

            target = (target > 0) ? 1.0 : 0.0;

            X_data_.push_back(features);
            y_data_.push_back(target);
        }

        file.close();

        std::cout << "Loaded " << X_data_.size() << " samples with "
                  << n_features_ << " features" << std::endl;
    }

    void compute_stats() {
        if (X_data_.empty()) {
            throw std::runtime_error("No data loaded");
        }

        const size_t n_samples = X_data_.size();
        stats_.min_vals.resize(n_features_, std::numeric_limits<double>::max());
        stats_.max_vals.resize(n_features_, std::numeric_limits<double>::lowest());
        stats_.mean_vals.resize(n_features_, 0.0);
        stats_.std_vals.resize(n_features_, 0.0);

        for (const auto& sample : X_data_) {
            for (size_t j = 0; j < n_features_; ++j) {
                stats_.min_vals[j] = std::min(stats_.min_vals[j], sample[j]);
                stats_.max_vals[j] = std::max(stats_.max_vals[j], sample[j]);
                stats_.mean_vals[j] += sample[j];
            }
        }

        for (size_t j = 0; j < n_features_; ++j) {
            stats_.mean_vals[j] /= n_samples;
        }

        for (const auto& sample : X_data_) {
            for (size_t j = 0; j < n_features_; ++j) {
                double diff = sample[j] - stats_.mean_vals[j];
                stats_.std_vals[j] += diff * diff;
            }
        }

        for (size_t j = 0; j < n_features_; ++j) {
            stats_.std_vals[j] = std::sqrt(stats_.std_vals[j] / n_samples);
            if (stats_.std_vals[j] < 1e-8) {
                stats_.std_vals[j] = 1.0;
            }
        }
    }

    void normalize_minmax() {
        if (stats_.min_vals.empty()) {
            compute_stats();
        }

        for (auto& sample : X_data_) {
            for (size_t j = 0; j < n_features_; ++j) {
                double range = stats_.max_vals[j] - stats_.min_vals[j];
                if (range > 1e-8) {
                    sample[j] = (sample[j] - stats_.min_vals[j]) / range;
                } else {
                    sample[j] = 0.0;
                }
            }
        }

        std::cout << "Applied Min-Max normalization" << std::endl;
    }

    void normalize_zscore() {
        if (stats_.mean_vals.empty()) {
            compute_stats();
        }

        for (auto& sample : X_data_) {
            for (size_t j = 0; j < n_features_; ++j) {
                sample[j] = (sample[j] - stats_.mean_vals[j]) / stats_.std_vals[j];
            }
        }

        std::cout << "Applied Z-Score normalization" << std::endl;
    }


    void shuffle(const unsigned int seed = 42) {
        std::mt19937 rng(seed);


        std::vector<size_t> indices(X_data_.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<std::vector<double>> X_shuffled;
        std::vector<double> y_shuffled;

        for (size_t idx : indices) {
            X_shuffled.push_back(X_data_[idx]);
            y_shuffled.push_back(y_data_[idx]);
        }

        X_data_ = std::move(X_shuffled);
        y_data_ = std::move(y_shuffled);

        std::cout << "Data shuffled with seed " << seed << std::endl;
    }

    void train_test_split(Tensor2& X_train, Tensor2& y_train,
                          Tensor2& X_test, Tensor2& y_test,
                          double test_ratio = 0.3) const {
        if (X_data_.empty()) {
            throw std::runtime_error("No data loaded");
        }

        const size_t n_samples = X_data_.size();
        const auto n_test = static_cast<size_t>(n_samples * test_ratio);
        const size_t n_train = n_samples - n_test;

        X_train = Tensor2(shape_type{n_train, n_features_});
        y_train = Tensor2(shape_type{n_train, 1});
        X_test = Tensor2(shape_type{n_test, n_features_});
        y_test = Tensor2(shape_type{n_test, 1});

        for (size_t i = 0; i < n_train; ++i) {
            for (size_t j = 0; j < n_features_; ++j) {
                X_train(i, j) = X_data_[i][j];
            }
            y_train(i, 0) = y_data_[i];
        }

        for (size_t i = 0; i < n_test; ++i) {
            for (size_t j = 0; j < n_features_; ++j) {
                X_test(i, j) = X_data_[n_train + i][j];
            }
            y_test(i, 0) = y_data_[n_train + i];
        }

        std::cout << "Split: " << n_train << " train, " << n_test << " test samples" << std::endl;
    }

    [[nodiscard]] size_t get_n_samples() const { return X_data_.size(); }
    [[nodiscard]] size_t get_n_features() const { return n_features_; }

    void print_class_distribution() const {
        size_t n_class0 = 0, n_class1 = 0;
        for (double label : y_data_) {
            if (label < 0.5) n_class0++;
            else n_class1++;
        }

        std::cout << "Class distribution:" << std::endl;
        std::cout << "  Class 0 (No disease): " << n_class0
                  << " (" << (100.0 * n_class0 / y_data_.size()) << "%)" << std::endl;
        std::cout << "  Class 1 (Disease): " << n_class1
                  << " (" << (100.0 * n_class1 / y_data_.size()) << "%)" << std::endl;
    }

    [[nodiscard]] const DatasetStats& get_stats() const { return stats_; }
};

} // namespace utec::data

#endif //DATA_LOADER_H
