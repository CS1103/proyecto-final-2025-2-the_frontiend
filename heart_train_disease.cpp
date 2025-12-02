//
// Created by josei on 1/12/2025.
//

#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>
#include "include/tensor.h"
#include "include/neural_network.h"
#include "include/nn_activation.h"
#include "include/nn_loss.h"
#include "include/nn_dense.h"
#include "include/nn_optimizer.h"
#include "include/data_loader.h"
#include "include/evaluation.h"


template<typename T, size_t Rank>
using Tensor = utec::algebra::Tensor<T, Rank>;
using namespace utec::neural_network;
using namespace utec::data;
using namespace utec::evaluation;



inline void print_header(const std::string& text) {
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << "|  " << std::left << std::setw(52) << text << "  |" << std::endl;
    std::cout << "|-----------------------------------------------------------|" << std::endl;
}

inline int train(int argc, char* argv[]) {
    std::cout << "|-----------------------------------------------------------|" << std::endl;
    std::cout << "|       HEART DISEASE PREDICTION - NEURAL NETWORK           |" << std::endl;
    std::cout << "|                        UTEC                               |" << std::endl;
    std::cout << "|-----------------------------------------------------------|" << std::endl;


    const std::string dataset_path = (argc > 1) ? argv[1] : "C:/Users/josei/CLionProjects/proyecto-final-2025-2-the_frontiend/heart.csv";

    // Parámetros a ajustar en cada entrenamiento
    constexpr size_t EPOCHS = 5000;
    constexpr size_t BATCH_SIZE = 32;
    constexpr double LEARNING_RATE = 0.01;
    constexpr double TEST_RATIO = 0.3;
    constexpr bool USE_ADAM = false;

    print_header("CONFIGURATION");
    std::cout << "Dataset:       " << dataset_path << std::endl;
    std::cout << "Epochs:        " << EPOCHS << std::endl;
    std::cout << "Batch Size:    " << BATCH_SIZE << std::endl;
    std::cout << "Learning Rate: " << LEARNING_RATE << std::endl;
    std::cout << "Optimizer:     " << (USE_ADAM ? "Adam" : "SGD") << std::endl;
    std::cout << "Test Ratio:    " << (TEST_RATIO * 100) << "%" << std::endl;


    print_header("DATA LOADING & PREPROCESSING");

    HeartDiseaseLoader loader;

    try {
        loader.load_csv(dataset_path, true);
        loader.print_class_distribution();
        loader.shuffle(42);
        loader.normalize_minmax();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cerr << "\nPlease ensure heart.csv is in the current directory." << std::endl;
        std::cerr << "Download from: https://archive.ics.uci.edu/ml/datasets/heart+disease" << std::endl;
        return 1;
    }

    utec::algebra::Tensor<double, 2> X_train, y_train, X_test, y_test;
    loader.train_test_split(X_train, y_train, X_test, y_test, TEST_RATIO);


    print_header("NEURAL NETWORK ARCHITECTURE");

    NeuralNetwork<double> nn;


    auto init_weights = [](auto& W) {
        for (auto& w : W) {
            constexpr double scale = 0.5;
            w = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0 * scale;
        }
    };

    auto init_bias = [](auto& b) {
        std::fill(b.begin(), b.end(), 0.0);
    };

    nn.add_layer(std::make_unique<Dense<double>>(13, 32, init_weights, init_bias));
    nn.add_layer(std::make_unique<ReLU<double>>());
    nn.add_layer(std::make_unique<Dense<double>>(32, 16, init_weights, init_bias));
    nn.add_layer(std::make_unique<ReLU<double>>());
    nn.add_layer(std::make_unique<Dense<double>>(16, 1, init_weights, init_bias));
    nn.add_layer(std::make_unique<Sigmoid<double>>());

    std::cout << "Architecture: Input(13) -> Dense(32) -> ReLU -> Dense(16) -> ReLU -> Dense(1) -> Sigmoid" << std::endl;
    std::cout << "Total parameters: ~600" << std::endl;

    print_header("TRAINING");

    auto start_time = std::chrono::high_resolution_clock::now();

    if (USE_ADAM) {
        nn.train<BCELoss, Adam>(X_train, y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE);
    } else {
        nn.train<BCELoss, SGD>(X_train, y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\nTraining completed in " << duration.count() << " seconds" << std::endl;

    print_header("TRAINING SET EVALUATION");

    auto y_train_pred = nn.predict(X_train);
    Metrics train_metrics = Evaluator::evaluate(y_train, y_train_pred);
    train_metrics.print();

    print_header("TEST SET EVALUATION");

    auto y_test_pred = nn.predict(X_test);
    Metrics test_metrics = Evaluator::evaluate(y_test, y_test_pred);
    test_metrics.print();

    print_header("ADDITIONAL ANALYSIS");

    Evaluator::print_sample_predictions(y_test, y_test_pred, 15);


    double best_threshold = Evaluator::find_best_threshold(y_test, y_test_pred);

    if (best_threshold != 0.5) {
        std::cout << "\nRe-evaluating with best threshold..." << std::endl;
        Metrics optimized_metrics = Evaluator::evaluate(y_test, y_test_pred, best_threshold);
        optimized_metrics.print();
    }


    Evaluator::print_roc_points(y_test, y_test_pred);


    print_header("FINAL SUMMARY");

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Training Accuracy:   " << (train_metrics.accuracy * 100) << "%" << std::endl;
    std::cout << "Test Accuracy:       " << (test_metrics.accuracy * 100) << "%" << std::endl;
    std::cout << "Test Precision:      " << (test_metrics.precision * 100) << "%" << std::endl;
    std::cout << "Test Recall:         " << (test_metrics.recall * 100) << "%" << std::endl;
    std::cout << "Test F1-Score:       " << (test_metrics.f1_score * 100) << "%" << std::endl;
    std::cout << "\nLiterature Benchmark: 85-93% accuracy" << std::endl;

    if (test_metrics.accuracy >= 0.80) {
        std::cout << "\n Model performance is competitive with literature!" << std::endl;
    } else if (test_metrics.accuracy >= 0.70) {
        std::cout << "\n Model performance is reasonable but could be improved." << std::endl;
        std::cout << "  Try: more epochs, different architecture, or Adam optimizer" << std::endl;
    } else {
        std::cout << "\n Model performance is below expected." << std::endl;
        std::cout << "  Suggestions: check data preprocessing, increase epochs, tune learning rate" << std::endl;
    }

    std::cout << "\n|-----------------------------------------------------------|" << std::endl;
    std::cout << "|                    TRAINING COMPLETE                        |" << std::endl;
    std::cout << "|-------------------------------------------------------------|" << std::endl;
    return 0;
}
