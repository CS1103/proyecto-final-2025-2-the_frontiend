
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <functional>
#include <type_traits>

namespace utec::algebra {

template<typename T, size_t Rank>
class Tensor {
public:
    using shape_type = std::array<size_t, Rank>;

private:
    shape_type shape_;
    std::vector<T> data_;

public:
    Tensor() = default;

    template<typename... Dims,
             typename = std::enable_if_t<
                 (sizeof...(Dims) == Rank) &&
                 (std::is_integral_v<Dims> && ...)
             >>
    explicit Tensor(Dims... dims) {
        std::array<size_t, Rank> temp = { static_cast<size_t>(dims)... };
        for (size_t i = 0; i < Rank; ++i) shape_[i] = temp[i];

        size_t total = 1;
        for (auto d : shape_) total *= d;
        data_.resize(total);
    }

    explicit Tensor(const shape_type& shape) : shape_(shape) {
        size_t total = 1;
        for (auto d : shape_) total *= d;
        data_.resize(total);
    }

    shape_type shape() const { return shape_; }
    [[nodiscard]] size_t size() const { return data_.size(); }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    template<typename... Idx>
    T& operator()(Idx... idxs) {
        std::array<size_t, Rank> idx = { static_cast<size_t>(idxs)... };
        size_t pos = 0;
        for (size_t i = 0; i < Rank; ++i) {
            if (idx[i] >= shape_[i])
                throw std::out_of_range("Index out of range");
            pos = pos * shape_[i] + idx[i];
        }
        return data_[pos];
    }

    template<typename... Idx>
    const T& operator()(Idx... idxs) const {
        std::array<size_t, Rank> idx = { static_cast<size_t>(idxs)... };
        size_t pos = 0;
        for (size_t i = 0; i < Rank; ++i) {
            if (idx[i] >= shape_[i])
                throw std::out_of_range("Index out of range");
            pos = pos * shape_[i] + idx[i];
        }
        return data_[pos];
    }

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() noexcept { return data_.begin(); }
    iterator end() noexcept { return data_.end(); }

    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }

    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    Tensor& operator=(std::initializer_list<T> il) {
        if (il.size() != data_.size())
            throw std::invalid_argument("Data size does not match tensor size");
        std::copy(il.begin(), il.end(), data_.begin());
        return *this;
    }


    template<typename... Dims>
    void reshape(Dims... dims) {
        static_assert((std::is_integral_v<Dims> && ...), "reshape dimensions must be integral");
        constexpr size_t N = sizeof...(Dims);
        if (N != Rank) {
            throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
        }
        std::array<size_t, Rank> new_shape = { static_cast<size_t>(dims)... };
        size_t new_total = 1;
        for (auto d : new_shape) new_total *= d;

        if (new_total != data_.size()) {
            throw std::invalid_argument("reshape inválido: el tamaño total no coincide");
        }
        shape_ = new_shape;
    }

    Tensor operator+(const Tensor& other) const {
        Tensor result(broadcast_shape(*this, other));
        for (size_t i = 0; i < result.data_.size(); ++i)
            result.data_[i] = get_broadcast(i, *this, result.shape_) + get_broadcast(i, other, result.shape_);
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        Tensor result(broadcast_shape(*this, other));
        for (size_t i = 0; i < result.data_.size(); ++i)
            result.data_[i] = get_broadcast(i, *this, result.shape_) - get_broadcast(i, other, result.shape_);
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        Tensor result(broadcast_shape(*this, other));
        for (size_t i = 0; i < result.data_.size(); ++i)
            result.data_[i] = get_broadcast(i, *this, result.shape_) * get_broadcast(i, other, result.shape_);
        return result;
    }

    Tensor operator+(const T& s) const {
        Tensor result = *this;
        for (auto& x : result.data_) x += s;
        return result;
    }

    Tensor operator-(const T& s) const {
        Tensor result = *this;
        for (auto& x : result.data_) x -= s;
        return result;
    }

    Tensor operator*(const T& s) const {
        Tensor result = *this;
        for (auto& x : result.data_) x *= s;
        return result;
    }

    Tensor operator/(const T& s) const {
        Tensor result = *this;
        for (auto& x : result.data_) x /= s;
        return result;
    }

    friend Tensor operator+(const T& s, const Tensor& t) {
        return t + s;
    }

    friend Tensor operator-(const T& s, const Tensor& t) {
        Tensor result = t;
        for (auto& x : result.data_) x = s - x;
        return result;
    }

    friend Tensor operator*(const T& s, const Tensor& t) {
        return t * s;
    }

    friend Tensor operator/(const T& s, const Tensor& t) {
        Tensor result = t;
        for (auto& x : result.data_) x = s / x;
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        print_recursive(os, t.data_, t.shape_, 0, 0);
        return os;
    }

private:
    static void print_recursive(std::ostream& os,
                                const std::vector<T>& data,
                                const std::array<size_t, Rank>& shape,
                                size_t dim,
                                const size_t offset) {
        if (dim < Rank - 1) {
            os << "{\n";

            size_t stride = 1;
            for (size_t k = dim + 1; k < Rank; ++k) stride *= shape[k];

            for (size_t i = 0; i < shape[dim]; ++i) {
                print_recursive(os, data, shape, dim + 1, offset + i * stride);
                if (i + 1 < shape[dim]) os << "\n";
            }

            os << "\n}";
        } else {
            for (size_t i = 0; i < shape[dim]; ++i) {
                os << data[offset + i];
                if (i + 1 < shape[dim]) os << " ";
            }
        }
    }

    static shape_type broadcast_shape(const Tensor& A, const Tensor& B) {
        shape_type result{};
        for (size_t i = 0; i < Rank; ++i) {
            if (A.shape_[i] == B.shape_[i]) result[i] = A.shape_[i];
            else if (A.shape_[i] == 1) result[i] = B.shape_[i];
            else if (B.shape_[i] == 1) result[i] = A.shape_[i];
            else throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        return result;
    }

    static T get_broadcast(const size_t index, const Tensor& t, const shape_type& out_shape) {
        shape_type idx;
        size_t temp = index;
        for (int i = Rank - 1; i >= 0; --i) {
            idx[i] = temp % out_shape[i];
            temp /= out_shape[i];
        }
        size_t pos = 0, stride = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            size_t id = (t.shape_[i] == 1 ? 0 : idx[i]);
            pos += id * stride;
            stride *= t.shape_[i];
        }
        return t.data_[pos];
    }
};

template<typename T, size_t Rank>
Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& t) {
    static_assert(Rank == 2, "transpose_2d only works for 2D tensors");
    Tensor<T, Rank> result(t.shape()[1], t.shape()[0]);
    for (size_t i = 0; i < t.shape()[0]; ++i) {
        for (size_t j = 0; j < t.shape()[1]; ++j) {
            result(j, i) = t(i, j);
        }
    }
    return result;
}

template<typename T, size_t Rank>
Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B) {
    static_assert(Rank == 2, "matrix_product only works for 2D tensors");
    if (A.shape()[1] != B.shape()[0])
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");

    Tensor<T, Rank> result(A.shape()[0], B.shape()[1]);
    for (size_t i = 0; i < A.shape()[0]; ++i)
        for (size_t j = 0; j < B.shape()[1]; ++j) {
            T sum = 0;
            for (size_t k = 0; k < A.shape()[1]; ++k)
                sum += A(i, k) * B(k, j);
            result(i, j) = sum;
        }
    return result;
}

template<typename T, size_t Rank, typename Func>
Tensor<T, Rank> apply(const Tensor<T, Rank>& input, Func&& f) {
    Tensor<T, Rank> result(input.shape());
    auto it_in = input.cbegin();
    auto it_out = result.begin();
    for (; it_in != input.cend(); ++it_in, ++it_out) {
        *it_out = std::invoke(f, *it_in);
    }
    return result;
}

} // namespace utec::algebra

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
