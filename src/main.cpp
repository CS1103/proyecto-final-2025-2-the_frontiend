//
// Created by josei on 29/11/2025.
//

#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <random>
#include "layers/tensor.h"
#include "layers/neural_network.h"
#include "layers/nn_activation.h"
#include "layers/nn_loss.h"
#include "optimizer/nn_optimizer.h"
#include "layers/nn_dense.h"
#include "tests/heart_train_disease.cpp"

using utec::neural_network::NeuralNetwork;
using utec::neural_network::Dense;
using utec::neural_network::ReLU;
using utec::neural_network::Sigmoid;
using utec::neural_network::MSELoss;
using utec::neural_network::BCELoss;
using utec::neural_network::SGD;
using utec::neural_network::Adam;
using utec::algebra::transpose_2d;
using utec::algebra::matrix_product;


#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"

void print_test_header(const std::string& test_name) {
    std::cout << "\n" << YELLOW << "========================================" << RESET << std::endl;
    std::cout << YELLOW << "TEST: " << test_name << RESET << std::endl;
    std::cout << YELLOW << "========================================" << RESET << std::endl;
}

void print_result(bool passed, const std::string& message = "") {
    if (passed) {
        std::cout << GREEN << "✓ PASSED" << RESET;
        if (!message.empty()) std::cout << " - " << message;
        std::cout << std::endl;
    } else {
        std::cout << RED << "✗ FAILED" << RESET;
        if (!message.empty()) std::cout << " - " << message;
        std::cout << std::endl;
    }
}

void test_tensor_basic() {
    print_test_header("Tensor Básico - Creación e Indexación");

    try {
        utec::algebra::Tensor<double, 2> t(3, 4);
        assert(t.shape()[0] == 3);
        assert(t.shape()[1] == 4);
        assert(t.size() == 12);
        print_result(true, "Creación de tensor 2D");

        t(0, 0) = 1.5;
        t(2, 3) = 9.9;
        assert(std::abs(t(0, 0) - 1.5) < 1e-9);
        assert(std::abs(t(2, 3) - 9.9) < 1e-9);
        print_result(true, "Indexación no-const");

        const auto& t_const = t;
        assert(std::abs(t_const(0, 0) - 1.5) < 1e-9);
        assert(std::abs(t_const(2, 3) - 9.9) < 1e-9);
        print_result(true, "Indexación const (Bug #3 verificado)");

        utec::algebra::Tensor<double, 1> t1d(5);
        t1d(2) = 3.14;
        const auto& t1d_const = t1d;
        assert(std::abs(t1d_const(2) - 3.14) < 1e-9);
        print_result(true, "Tensor 1D const indexing (Bug #3 verificado)");

    } catch (const std::exception& e) {
        print_result(false, std::string("Exception: ") + e.what());
    }
}

void test_tensor_operations() {
    print_test_header("Operaciones de Tensor");

    try {
        utec::algebra::Tensor<double, 2> A(2, 2);
        A(0, 0) = 1.0; A(0, 1) = 2.0;
        A(1, 0) = 3.0; A(1, 1) = 4.0;

        utec::algebra::Tensor<double, 2> B(2, 2);
        B(0, 0) = 5.0; B(0, 1) = 6.0;
        B(1, 0) = 7.0; B(1, 1) = 8.0;

        auto C = A + B;
        assert(std::abs(C(0, 0) - 6.0) < 1e-9);
        assert(std::abs(C(1, 1) - 12.0) < 1e-9);
        print_result(true, "Suma de tensores");

        auto D = A * B;
        assert(std::abs(D(0, 0) - 5.0) < 1e-9);
        assert(std::abs(D(1, 1) - 32.0) < 1e-9);
        print_result(true, "Multiplicación elemento a elemento");

        auto E = A + 10.0;
        assert(std::abs(E(0, 0) - 11.0) < 1e-9);
        print_result(true, "Suma con escalar");

        auto F = matrix_product(A, B);
        assert(std::abs(F(0, 0) - 19.0) < 1e-9); // 1*5 + 2*7
        assert(std::abs(F(0, 1) - 22.0) < 1e-9); // 1*6 + 2*8
        assert(std::abs(F(1, 0) - 43.0) < 1e-9); // 3*5 + 4*7
        assert(std::abs(F(1, 1) - 50.0) < 1e-9); // 3*6 + 4*8
        print_result(true, "Multiplicación matricial");

        auto G = transpose_2d(A);
        assert(std::abs(G(0, 0) - 1.0) < 1e-9);
        assert(std::abs(G(0, 1) - 3.0) < 1e-9);
        assert(std::abs(G(1, 0) - 2.0) < 1e-9);
        assert(std::abs(G(1, 1) - 4.0) < 1e-9);
        print_result(true, "Transpose 2D");

    } catch (const std::exception& e) {
        print_result(false, std::string("Exception: ") + e.what());
    }
}

void test_activations() {
    print_test_header("Funciones de Activación");

    try {
        utec::algebra::Tensor<double, 2> input(2, 3);
        input(0, 0) = -1.0; input(0, 1) = 0.5; input(0, 2) = 2.0;
        input(1, 0) = -0.5; input(1, 1) = 0.0; input(1, 2) = 1.5;

        ReLU<double> relu;
        auto relu_out = relu.forward(input);

        assert(std::abs(relu_out(0, 0) - 0.0) < 1e-9);
        assert(std::abs(relu_out(0, 1) - 0.5) < 1e-9);
        assert(std::abs(relu_out(0, 2) - 2.0) < 1e-9);
        assert(std::abs(relu_out(1, 0) - 0.0) < 1e-9);
        assert(std::abs(relu_out(1, 2) - 1.5) < 1e-9);
        print_result(true, "ReLU forward");

        utec::algebra::Tensor<double, 2> grad(2, 3);
        grad.fill(1.0);

        auto relu_grad = relu.backward(grad);
        assert(std::abs(relu_grad(0, 0) - 0.0) < 1e-9); // input was -1.0
        assert(std::abs(relu_grad(0, 1) - 1.0) < 1e-9); // input was 0.5
        assert(std::abs(relu_grad(0, 2) - 1.0) < 1e-9); // input was 2.0
        assert(std::abs(relu_grad(1, 0) - 0.0) < 1e-9); // input was -0.5
        print_result(true, "ReLU backward (Bug #4 verificado)");

        utec::algebra::Tensor<double, 2> sigmoid_input(1, 3);
        sigmoid_input(0, 0) = 0.0;
        sigmoid_input(0, 1) = 1.0;
        sigmoid_input(0, 2) = -1.0;

        Sigmoid<double> sigmoid;
        auto sigmoid_out = sigmoid.forward(sigmoid_input);

        assert(std::abs(sigmoid_out(0, 0) - 0.5) < 1e-6);
        assert(std::abs(sigmoid_out(0, 1) - 0.7310586) < 1e-6);
        assert(std::abs(sigmoid_out(0, 2) - 0.2689414) < 1e-6);
        print_result(true, "Sigmoid forward");

        utec::algebra::Tensor<double, 2> sigmoid_grad(1, 3);
        sigmoid_grad.fill(1.0);

        auto sigmoid_grad_out = sigmoid.backward(sigmoid_grad);
        assert(std::abs(sigmoid_grad_out(0, 0) - 0.25) < 1e-6);
        print_result(true, "Sigmoid backward");

    } catch (const std::exception& e) {
        print_result(false, std::string("Exception: ") + e.what());
    }
}

void test_loss_functions() {
    print_test_header("Funciones de Pérdida");

    try {
        utec::algebra::Tensor<double, 2> y_pred(2, 2);
        y_pred(0, 0) = 1.0; y_pred(0, 1) = 2.0;
        y_pred(1, 0) = 3.0; y_pred(1, 1) = 4.0;

        utec::algebra::Tensor<double, 2> y_true(2, 2);
        y_true(0, 0) = 1.5; y_true(0, 1) = 2.5;
        y_true(1, 0) = 2.5; y_true(1, 1) = 3.5;

        MSELoss<double> mse(y_pred, y_true);
        const double loss = mse.loss();
        assert(std::abs(loss - 0.25) < 1e-9);
        print_result(true, "MSE Loss calculation");

        auto mse_grad = mse.loss_gradient();
        assert(std::abs(mse_grad(0, 0) - (-0.25)) < 1e-9);
        assert(std::abs(mse_grad(1, 1) - 0.25) < 1e-9);
        print_result(true, "MSE Loss gradient");

        utec::algebra::Tensor<double, 2> y_pred_bce(2, 1);
        y_pred_bce(0, 0) = 0.8;
        y_pred_bce(1, 0) = 0.3;

        utec::algebra::Tensor<double, 2> y_true_bce(2, 1);
        y_true_bce(0, 0) = 1.0;
        y_true_bce(1, 0) = 0.0;

        BCELoss<double> bce(y_pred_bce, y_true_bce);
        double bce_loss = bce.loss();
        assert(bce_loss > 0.0);
        print_result(true, "BCE Loss calculation");

        auto bce_grad = bce.loss_gradient();
        assert(bce_grad.size() == 2);
        print_result(true, "BCE Loss gradient");

    } catch (const std::exception& e) {
        print_result(false, std::string("Exception: ") + e.what());
    }
}

void test_optimizers() {
    print_test_header("Optimizadores");

    try {
        utec::algebra::Tensor<double, 2> params(2, 2);
        params(0, 0) = 1.0; params(0, 1) = 2.0;
        params(1, 0) = 3.0; params(1, 1) = 4.0;

        utec::algebra::Tensor<double, 2> grads(2, 2);
        grads.fill(0.1);

        SGD<double> sgd(0.1);
        sgd.update(params, grads);

        assert(std::abs(params(0, 0) - 0.99) < 1e-9);
        assert(std::abs(params(1, 1) - 3.99) < 1e-9);
        print_result(true, "SGD update");

        utec::algebra::Tensor<double, 2> params_adam(2, 2);
        params_adam.fill(1.0);

        utec::algebra::Tensor<double, 2> grads_adam(2, 2);
        grads_adam.fill(0.1);

        Adam<double> adam(0.001);
        adam.update(params_adam, grads_adam);
        adam.step();

        assert(std::abs(params_adam(0, 0) - 1.0) > 1e-9);
        print_result(true, "Adam update y step (Bug #2 verificado)");

        for (int i = 0; i < 5; ++i) {
            adam.update(params_adam, grads_adam);
            adam.step();
        }
        print_result(true, "Adam múltiples steps");

    } catch (const std::exception& e) {
        print_result(false, std::string("Exception: ") + e.what());
    }
}

void test_neural_network_batches() {
    print_test_header("Red Neuronal - Batch Dimensions");

    try {
        utec::algebra::Tensor<double, 2> X(4, 3);
        X.fill(0.5);

        utec::algebra::Tensor<double, 2> Y(4, 2);
        Y.fill(0.8);

        NeuralNetwork<double> nn;
        auto init_weights = [](auto& W) {
            std::fill(W.begin(), W.end(), 0.1);
        };
        auto init_bias = [](auto& b) {
            std::fill(b.begin(), b.end(), 0.0);
        };
        nn.add_layer(std::make_unique<Dense<double>>(3, 2, init_weights, init_bias));

        try {
            nn.train<MSELoss>(X, Y, 2, 2, 0.01);
            print_result(true, "Batch con input_dim != output_dim (Bug #1 verificado)");
        } catch (const std::exception& e) {
            print_result(false, std::string("Bug #1 no corregido: ") + e.what());
        }

        nn.train<MSELoss>(X, Y, 2, 10, 0.01);
        print_result(true, "Batch size mayor que n_samples");

        nn.train<MSELoss>(X, Y, 2, 0, 0.01);
        print_result(true, "Batch size = 0 (auto-batch)");

    } catch (const std::exception& e) {
        print_result(false, std::string("Exception: ") + e.what());
    }
}

void test_xor_problem() {
    print_test_header("Red Neuronal - Problema XOR (Integración)");

    try {
        utec::algebra::Tensor<double, 2> X(4, 2);
        X(0, 0) = 0.0; X(0, 1) = 0.0; // XOR(0,0) = 0
        X(1, 0) = 0.0; X(1, 1) = 1.0; // XOR(0,1) = 1
        X(2, 0) = 1.0; X(2, 1) = 0.0; // XOR(1,0) = 1
        X(3, 0) = 1.0; X(3, 1) = 1.0; // XOR(1,1) = 0

        utec::algebra::Tensor<double, 2> Y(4, 1);
        Y(0, 0) = 0.0;
        Y(1, 0) = 1.0;
        Y(2, 0) = 1.0;
        Y(3, 0) = 0.0;

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

        nn.add_layer(std::make_unique<Dense<double>>(2, 4, init_weights, init_bias));
        nn.add_layer(std::make_unique<ReLU<double>>());
        nn.add_layer(std::make_unique<Dense<double>>(4, 1, init_weights, init_bias));
        nn.add_layer(std::make_unique<Sigmoid<double>>());

        std::cout << "Entrenando XOR con SGD..." << std::endl;
        nn.train<BCELoss, SGD>(X, Y, 5000, 4, 0.5);

        // Predecir
        auto predictions = nn.predict(X);

        std::cout << "\nResultados:" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            std::cout << "Input: (" << X(i, 0) << ", " << X(i, 1) << ") -> "
                      << "Pred: " << std::fixed << std::setprecision(4)
                      << predictions(i, 0) << " | "
                      << "True: " << Y(i, 0) << std::endl;
        }

        bool converged = true;
        for (size_t i = 0; i < 4; ++i) {
            double pred = predictions(i, 0);
            double true_val = Y(i, 0);
            if (std::abs(pred - true_val) > 0.3) {
                converged = false;
                break;
            }
        }

        print_result(converged, converged ?
            "XOR converge razonablemente" :
            "XOR no converge (puede necesitar más epochs o ajuste de hiperparámetros)");

    } catch (const std::exception& e) {
        print_result(false, std::string("Exception: ") + e.what());
    }
}

void test_suite () {
    std::cout << GREEN << "╔════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << GREEN << "║  TEST SUITE - RED NEURONAL C++         ║" << RESET << std::endl;
    std::cout << GREEN << "║  Verificación de Correcciones de Bugs  ║" << RESET << std::endl;
    std::cout << GREEN << "╚════════════════════════════════════════╝" << RESET << std::endl;

    test_tensor_basic();
    test_tensor_operations();
    test_activations();
    test_loss_functions();
    test_optimizers();
    test_neural_network_batches();
    test_xor_problem();

    std::cout << "\n" << GREEN << "╔════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << GREEN << "║  TESTS COMPLETADOS                     ║" << RESET << std::endl;
    std::cout << GREEN << "╚════════════════════════════════════════╝" << RESET << std::endl;

}

int main() {
    train(1, ('1', '2'));
    return 0;
}