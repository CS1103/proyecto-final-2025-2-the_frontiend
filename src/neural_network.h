//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include "nn_loss.h"
#include "nn_interfaces.h"
#include "nn_optimizer.h"

namespace utec::neural_network {
    template <typename T>
class SGD;

template<typename T>
class NeuralNetwork {
public:
    using Tensor2 = Tensor<T, 2>;

    NeuralNetwork() = default;

    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    Tensor2 predict(const Tensor2& X) {
        Tensor2 out = X;
        for (auto& layer : layers_) {
            out = layer->forward(out);
        }
        return out;
    }

    template <
        template <typename...> class LossType,
        template <typename...> class OptimizerType = SGD
    >
    void train(
        const Tensor2& X,
        const Tensor2& Y,
        const size_t epochs,
        size_t batch_size,
        T learning_rate)
    {
        if (layers_.empty()) {
            throw std::runtime_error("NeuralNetwork::train - no layers added");
        }

        if (X.shape()[0] != Y.shape()[0]) {
            throw std::invalid_argument("X and Y must have same number of samples");
        }

        OptimizerType<T> optimizer(learning_rate);

        const size_t n_samples = X.shape()[0];
        if (batch_size == 0 || batch_size > n_samples) {
            batch_size = n_samples;
        }

        const size_t n_batches = (n_samples + batch_size - 1) / batch_size;

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            T epoch_loss = static_cast<T>(0);
            size_t total_samples = 0;

            for (size_t batch = 0; batch < n_batches; ++batch) {
                const size_t start = batch * batch_size;
                const size_t end = std::min(start + batch_size, n_samples);
                const size_t current_batch_size = end - start;
                epoch_loss += epoch_loss * current_batch_size;
                total_samples += current_batch_size;

                auto x_shape = typename Tensor2::shape_type{ current_batch_size, X.shape()[1] };
                Tensor2 X_batch(x_shape);

                auto y_shape = typename Tensor2::shape_type{ current_batch_size, Y.shape()[1] };
                Tensor2 Y_batch(y_shape);

                for (size_t i = 0; i < current_batch_size; ++i) {
                    for (size_t j = 0; j < X.shape()[1]; ++j)
                        X_batch(i, j) = X(start + i, j);
                    for (size_t j = 0; j < Y.shape()[1]; ++j)
                        Y_batch(i, j) = Y(start + i, j);
                }

                Tensor2 Y_pred = predict(X_batch);

                LossType<T> loss(Y_pred, Y_batch);
                epoch_loss += loss.loss();

                Tensor2 grad = loss.loss_gradient();

                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                    grad = (*it)->backward(grad);
                }

                for (auto& layer : layers_) {
                    layer->update_params(optimizer);
                }

                optimizer.step();
            }
            epoch_loss / static_cast<T>(total_samples);

            if (epoch % 500 == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << epoch + 1
                          << "/" << epochs
                          << " - Loss: " << epoch_loss / static_cast<T>(n_batches)
                          << std::endl;
            }
        }
    }

private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;
};
}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
