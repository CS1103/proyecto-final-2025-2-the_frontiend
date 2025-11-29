//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#pragma once
#include <cmath>
#include "nn_interfaces.h"

namespace utec::neural_network {

  template<typename T>
  class ReLU final : public ILayer<T> {
  public:
    using Tensor2 = Tensor<T, 2>;
  private:
    Tensor2 last_input_;

  public:
    ReLU() = default;

    Tensor2 forward(const Tensor2& x) override {
      last_input_ = x;
      Tensor2 out = x;
      auto it_out = out.begin();
      auto it_x = x.cbegin();
      for (; it_out != out.end(); ++it_out, ++it_x) {
        *it_out = std::max(static_cast<T>(0), *it_x);
      }
      return out;
    }

    Tensor2 backward(const Tensor2& gradients) override {
      Tensor2 dx = gradients;
      auto it_dx = dx.begin();
      auto it_last = last_input_.begin();
      for (; it_dx != dx.end(); ++it_dx, ++it_last) {
        if (*it_last > static_cast<T>(0)) {
        } else {
          *it_dx = static_cast<T>(0);
        }
      }
      return dx;
    }


    void update_params(IOptimizer<T>&) override {
    }
  };

  template<typename T>
  class Sigmoid final : public ILayer<T> {
  public:
    using Tensor2 = Tensor<T, 2>;
  private:
    Tensor2 last_output_;

  public:
    Sigmoid() = default;

Tensor2 forward(const Tensor2& x) override {
    Tensor2 out = x;
    auto it_out = out.begin();
    auto it_x = x.cbegin();
    for (; it_out != out.end(); ++it_out, ++it_x) {
        *it_out = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-(*it_x)));
    }
    last_output_ = out;
    return out;
}

Tensor2 backward(const Tensor2& gradients) override {
    Tensor2 dx = gradients;
    auto it_dx = dx.begin();
    auto it_out = last_output_.begin();
    for (; it_dx != dx.end(); ++it_dx, ++it_out) {
        *it_dx *= (*it_out) * (static_cast<T>(1) - *it_out);
    }
    return dx;
}

    void update_params(IOptimizer<T>&) override {
    }
  };

}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
