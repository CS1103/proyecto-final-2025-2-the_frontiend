//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#pragma once
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include "layers/nn_interfaces.h"

namespace utec::neural_network {

template<typename T>
class SGD final : public IOptimizer<T> {
public:
    using Tensor2 = Tensor<T, 2>;

    explicit SGD(T learning_rate = static_cast<T>(0.01))
        : lr_(learning_rate)
    {}

    void update(Tensor2& params, const Tensor2& grads) override {
        if (params.shape() != grads.shape()) {
            throw std::invalid_argument("SGD::update - shape mismatch");
        }
        auto pit = params.begin();
        auto git = grads.cbegin();
        for (; pit != params.end(); ++pit, ++git) {
            *pit -= lr_ * (*git);
        }
    }

private:
    T lr_;
};

template<typename T>
class Adam final : public IOptimizer<T> {
public:
    using Tensor2 = Tensor<T, 2>;

    explicit Adam(T learning_rate = static_cast<T>(0.001),
                  T beta1 = static_cast<T>(0.9),
                  T beta2 = static_cast<T>(0.999),
                  T epsilon = static_cast<T>(1e-8))
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2),
          epsilon_(epsilon), t_(1)
    {}

    void update(Tensor2& params, const Tensor2& grads) override {
        if (params.shape() != grads.shape()) {
            throw std::invalid_argument("Adam::update - shape mismatch");
        }

        auto& m = m_[&params];
        auto& v = v_[&params];
        const auto shape = params.shape();

        if (m.size() == 0) m = Tensor2(shape);
        if (v.size() == 0) v = Tensor2(shape);
        const T one = static_cast<T>(1);
        const T lr_t = lr_ * std::sqrt(one - std::pow(beta2_, t_)) / (one - std::pow(beta1_, t_));

        auto pit = params.begin();
        auto git = grads.cbegin();
        auto mit = m.begin();
        auto vit = v.begin();

        for (; pit != params.end(); ++pit, ++git, ++mit, ++vit) {
            *mit = beta1_ * (*mit) + (one - beta1_) * (*git);
            *vit = beta2_ * (*vit) + (one - beta2_) * (*git) * (*git);
            *pit -= lr_t * (*mit) / (std::sqrt(*vit) + epsilon_);
        }
    }


    void step() override {
        t_ += 1;
    }

private:
    T lr_;
    T beta1_, beta2_, epsilon_;
    size_t t_;

    std::unordered_map<void*, Tensor2> m_;
    std::unordered_map<void*, Tensor2> v_;
};

}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
