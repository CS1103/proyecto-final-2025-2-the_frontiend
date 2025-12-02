//
// Created by josei on 1/12/2025.
//

#ifndef EVALUATION_H
#define EVALUATION_H

#pragma once
#include <iostream>
#include <iomanip>
#include <cmath>
#include "include/tensor.h"

namespace utec::evaluation {

using Tensor2 = utec::algebra::Tensor<double, 2>;

struct ConfusionMatrix {
    size_t true_positive = 0;
    size_t true_negative = 0;
    size_t false_positive = 0;
    size_t false_negative = 0;

    void print() const {
        std::cout << "\n╔------------------------------------╗" << std::endl;
        std::cout << "|      CONFUSION MATRIX              |" << std::endl;
        std::cout << "╚------------------------------------╝" << std::endl;
        std::cout << "                 Predicted" << std::endl;
        std::cout << "               Neg      Pos" << std::endl;
        std::cout << "Actual  Neg  " << std::setw(5) << true_negative
                  << "    " << std::setw(5) << false_positive << std::endl;
        std::cout << "        Pos  " << std::setw(5) << false_negative
                  << "    " << std::setw(5) << true_positive << std::endl;
        std::cout << std::endl;
    }
};

struct Metrics {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
    double specificity = 0.0;
    ConfusionMatrix cm;

    void print() const {
        std::cout << "\n╔------------------------------------╗" << std::endl;
        std::cout << "|      EVALUATION METRICS            |" << std::endl;
        std::cout << "╚------------------------------------╝" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Accuracy:    " << (accuracy * 100) << "%" << std::endl;
        std::cout << "Precision:   " << (precision * 100) << "%" << std::endl;
        std::cout << "Recall:      " << (recall * 100) << "%" << std::endl;
        std::cout << "F1-Score:    " << (f1_score * 100) << "%" << std::endl;
        std::cout << "Specificity: " << (specificity * 100) << "%" << std::endl;

        cm.print();
    }
};

class Evaluator {
public:
    static Metrics evaluate(const Tensor2& y_true, const Tensor2& y_pred, double threshold = 0.5) {
        if (y_true.shape()[0] != y_pred.shape()[0]) {
            throw std::invalid_argument("y_true and y_pred must have same number of samples");
        }

        Metrics metrics;
        const size_t n_samples = y_true.shape()[0];

        for (size_t i = 0; i < n_samples; ++i) {
            const double true_label = y_true(i, 0);
            const double pred_prob = y_pred(i, 0);
            const double pred_label = (pred_prob >= threshold) ? 1.0 : 0.0;

            if (true_label > 0.5 && pred_label > 0.5) {
                metrics.cm.true_positive++;
            } else if (true_label < 0.5 && pred_label < 0.5) {
                metrics.cm.true_negative++;
            } else if (true_label < 0.5 && pred_label > 0.5) {
                metrics.cm.false_positive++;
            } else {
                metrics.cm.false_negative++;
            }
        }

        const double tp = static_cast<double>(metrics.cm.true_positive);
        const double tn = static_cast<double>(metrics.cm.true_negative);
        const double fp = static_cast<double>(metrics.cm.false_positive);
        const double fn = static_cast<double>(metrics.cm.false_negative);

        metrics.accuracy = (tp + tn) / n_samples;

        if (tp + fp > 0) {
            metrics.precision = tp / (tp + fp);
        } else {
            metrics.precision = 0.0;
        }

        if (tp + fn > 0) {
            metrics.recall = tp / (tp + fn);
        } else {
            metrics.recall = 0.0;
        }

        if (metrics.precision + metrics.recall > 0) {
            metrics.f1_score = 2.0 * (metrics.precision * metrics.recall) /
                               (metrics.precision + metrics.recall);
        } else {
            metrics.f1_score = 0.0;
        }

        if (tn + fp > 0) {
            metrics.specificity = tn / (tn + fp);
        } else {
            metrics.specificity = 0.0;
        }

        return metrics;
    }

    static double accuracy(const Tensor2& y_true, const Tensor2& y_pred, double threshold = 0.5) {
        const size_t n_samples = y_true.shape()[0];
        size_t correct = 0;

        for (size_t i = 0; i < n_samples; ++i) {
            double true_label = y_true(i, 0);
            double pred_prob = y_pred(i, 0);
            double pred_label = (pred_prob >= threshold) ? 1.0 : 0.0;

            if (std::abs(true_label - pred_label) < 0.5) {
                correct++;
            }
        }

        return static_cast<double>(correct) / n_samples;
    }

    static void print_sample_predictions(const Tensor2& y_true, const Tensor2& y_pred,
                                        size_t n_samples = 10) {
        std::cout << "\n╔------------------------------------╗" << std::endl;
        std::cout << "|      SAMPLE PREDICTIONS            |" << std::endl;
        std::cout << "╚------------------------------------╝" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "True Label | Predicted Prob | Predicted Label | Correct?" << std::endl;
        std::cout << "-----------------------------------------------------------" << std::endl;

        const size_t total = std::min(n_samples, y_true.shape()[0]);
        for (size_t i = 0; i < total; ++i) {
            double true_label = y_true(i, 0);
            double pred_prob = y_pred(i, 0);
            double pred_label = (pred_prob >= 0.5) ? 1.0 : 0.0;
            bool correct = (std::abs(true_label - pred_label) < 0.5);

            std::cout << "    " << static_cast<int>(true_label)
                      << "      |      " << pred_prob
                      << "      |        " << static_cast<int>(pred_label)
                      << "        |    " << (correct ? "✓" : "✗")
                      << std::endl;
        }
        std::cout << std::endl;
    }

    static double find_best_threshold(const Tensor2& y_true, const Tensor2& y_pred) {
        double best_threshold = 0.5;
        double best_f1 = 0.0;

        std::cout << "\nSearching for best threshold..." << std::endl;

        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.05) {
            Metrics m = evaluate(y_true, y_pred, threshold);
            if (m.f1_score > best_f1) {
                best_f1 = m.f1_score;
                best_threshold = threshold;
            }
        }

        std::cout << "Best threshold: " << best_threshold
                  << " (F1-Score: " << best_f1 << ")" << std::endl;

        return best_threshold;
    }

    static void print_roc_points(const Tensor2& y_true, const Tensor2& y_pred) {
        std::cout << "\n╔------------------------------------╗" << std::endl;
        std::cout << "|      ROC CURVE POINTS              |" << std::endl;
        std::cout << "╚------------------------------------╝" << std::endl;
        std::cout << "Threshold | TPR (Recall) | FPR" << std::endl;
        std::cout << "---------------------------------------" << std::endl;

        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.1) {
            Metrics m = evaluate(y_true, y_pred, threshold);
            double fpr = 1.0 - m.specificity;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "   " << threshold << "    |    ";
            std::cout << std::setprecision(4);
            std::cout << m.recall << "   |  " << fpr << std::endl;
        }
        std::cout << std::endl;
    }
};

} // namespace utec::evaluation

#endif //EVALUATION_H
