//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#pragma once
#include <stdexcept>
#include "nn_interfaces.h"

namespace utec::neural_network {

template<typename T>
class Dense final : public ILayer<T> {
public:
    using Tensor2 = Tensor<T, 2>;

private:
    Tensor2 W_;
    Tensor2 b_;
    Tensor2 X_;
    Tensor2 dW_;
    Tensor2 db_;

public:
    template<typename InitWFun, typename InitBFun>
Dense(const size_t in_features, const size_t out_features,
      InitWFun init_w_fun, InitBFun init_b_fun)
    {
        using shape_type = typename Tensor2::shape_type;

        shape_type w_shape = { static_cast<size_t>(in_features), static_cast<size_t>(out_features) };
        shape_type b_shape = { static_cast<size_t>(1),          static_cast<size_t>(out_features) };

        W_ = Tensor2(w_shape);
        b_ = Tensor2(b_shape);
        dW_ = Tensor2(w_shape);
        db_ = Tensor2(b_shape);

        init_w_fun(W_);
        init_b_fun(b_);
    }

    Tensor2 forward(const Tensor2& X) override {
        X_ = X;

        const auto shape_x = X.shape();
        const auto shape_w = W_.shape();
        const size_t batch = shape_x[0];
        const size_t in_features = shape_x[1];
        const size_t out_features = shape_w[1];

        if (shape_w[0] != in_features) {
            throw std::invalid_argument("Dense::forward - Shape mismatch between X and W");
        }

        Tensor2 Y(typename Tensor2::shape_type{ static_cast<size_t>(batch),
                                        static_cast<size_t>(out_features) });

        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < in_features; ++k) {
                    sum += X(i, k) * W_(k, j);
                }
                Y(i, j) = sum + b_(0, j);
            }
        }
        return Y;
    }

    Tensor2 backward(const Tensor2& dZ) override {
        const auto shape_x = X_.shape();
        const auto shape_w = W_.shape();
        const auto shape_dz = dZ.shape();

        const size_t batch = shape_x[0];
        const size_t in_features = shape_x[1];
        const size_t out_features = shape_w[1];

        if (shape_dz[0] != batch || shape_dz[1] != out_features) {
            throw std::invalid_argument("Dense::backward - Shape mismatch between dZ and layer output");
        }

        dW_.fill(static_cast<T>(0));
        for (size_t i = 0; i < in_features; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                T sum = 0;
                for (size_t b = 0; b < batch; ++b) {
                    sum += X_(b, i) * dZ(b, j);
                }
                dW_(i, j) = sum / static_cast<T>(batch);
            }
        }

        db_.fill(static_cast<T>(0));
        for (size_t j = 0; j < out_features; ++j) {
            T sum = 0;
            for (size_t b = 0; b < batch; ++b) {
                sum += dZ(b, j);
            }
            db_(0, j) = sum / static_cast<T>(batch);
        }

        Tensor2 dX(typename Tensor2::shape_type{ static_cast<size_t>(batch),
                                        static_cast<size_t>(in_features) });

        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < in_features; ++i) {
                T sum = 0;
                for (size_t j = 0; j < out_features; ++j) {
                    sum += dZ(b, j) * W_(i, j);
                }
                dX(b, i) = sum;
            }
        }

        return dX;
    }

    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(W_, dW_);
        optimizer.update(b_, db_);
    }

    const Tensor2& weights() const { return W_; }
    const Tensor2& biases()  const { return b_; }
    const Tensor2& grad_weights() const { return dW_; }
    const Tensor2& grad_biases()  const { return db_; }
};

}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H