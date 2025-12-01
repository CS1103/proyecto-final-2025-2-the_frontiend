//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#pragma once
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "nn_interfaces.h"

namespace utec::neural_network {

template<typename T>
class MSELoss final : public ILoss<T, 2> {
public:
    using Tensor2 = Tensor<T, 2>;

    MSELoss(const Tensor2& y_pred, const Tensor2& y_true)
        : y_pred_(y_pred), y_true_(y_true) {
        if (y_pred_.size() != y_true.size()) {
            throw std::invalid_argument("Incorrect Size");
        }
    }

    T loss() const override {
        const size_t N = y_pred_.size();
        if (N == 0) return static_cast<T>(0);
        auto itp = y_pred_.cbegin();
        auto itt = y_true_.cbegin();
        long double acc = 0.0L;
        for (size_t i = 0; i < N; ++i, ++itp, ++itt) {
            const long double diff = static_cast<long double>(*itp) - static_cast<long double>(*itt);
            acc += diff * diff;
        }
        return static_cast<T>(acc / static_cast<long double>(N));
    }

    Tensor2 loss_gradient() const override {
        const size_t N = y_pred_.size();
        Tensor2 grad(y_pred_.shape());
        if (N == 0) return grad;
        auto itp = y_pred_.cbegin();
        auto itt = y_true_.cbegin();
        auto ig = grad.begin();
        const long double invN2 = 2.0L / static_cast<long double>(N);
        for (size_t i = 0; i < N; ++i, ++itp, ++itt, ++ig) {
            long double d = static_cast<long double>(*itp) - static_cast<long double>(*itt);
            *ig = static_cast<T>(invN2 * d);
        }
        return grad;
    }

private:
    Tensor2 y_pred_;
    Tensor2 y_true_;
};

template<typename T>
class BCELoss final : public ILoss<T, 2> {
public:
    using Tensor2 = Tensor<T, 2>;

    BCELoss(const Tensor2& y_pred, const Tensor2& y_true, T eps = static_cast<T>(1e-12))
        : y_pred_(y_pred), y_true_(y_true), eps_(std::max(eps, static_cast<T>(1e-15))) {
        if (y_pred_.size() != y_true.size()) {
            throw std::invalid_argument("Incorrect Size");
        }
    }

    T loss() const override {
        size_t N = y_pred_.size();
        if (N == 0) return static_cast<T>(0);
        auto itp = y_pred_.cbegin();
        auto itt = y_true_.cbegin();
        long double acc = 0.0L;
        for (size_t i = 0; i < N; ++i, ++itp, ++itt) {
            long double p = std::clamp(static_cast<long double>(*itp), static_cast<long double>(eps_), static_cast<long double>(1.0L - eps_));
            const auto y = static_cast<long double>(*itt);
            acc += -(y * std::log(p) + (1.0L - y) * std::log(1.0L - p));
        }
        return static_cast<T>(acc / static_cast<long double>(N));
    }

    Tensor2 loss_gradient() const override {
        const size_t N = y_pred_.size();
        Tensor2 grad(y_pred_.shape());
        if (N == 0) return grad;
        auto itp = y_pred_.cbegin();
        auto itt = y_true_.cbegin();
        auto ig = grad.begin();
        const long double invN = 1.0L / static_cast<long double>(N);
        for (size_t i = 0; i < N; ++i, ++itp, ++itt, ++ig) {
            long double p = std::clamp(static_cast<long double>(*itp), static_cast<long double>(eps_), static_cast<long double>(1.0L - eps_));
            const long double y = static_cast<long double>(*itt);
            const long double d = ( - (y / p) + ((1.0L - y) / (1.0L - p)) );
            *ig = static_cast<T>(invN * d);
        }
        return grad;
    }

private:
    Tensor2 y_pred_;
    Tensor2 y_true_;
    T eps_;
};

}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
