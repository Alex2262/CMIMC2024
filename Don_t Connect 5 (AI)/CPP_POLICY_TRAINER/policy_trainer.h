//
// Created by Alexander Tian on 4/26/24.
//

#ifndef DC5_POLICY_TRAINER_H
#define DC5_POLICY_TRAINER_H

#endif //DC5_POLICY_TRAINER_H

#include <string>
#include <random>
#include "datagen.h"

constexpr int INPUT_SIZE = 3 * 96;
constexpr int OUTPUT_SIZE = 97;


constexpr int EPOCHS = 40;

constexpr double DEFAULT_LR = 0.001;


inline double ReLU(double x) {
    return std::max(x, 0.0);
}

inline double ReLU_derivative(double x) {
    return x > 0 ? 1 : 0;
}

inline double dot_product(std::vector<double>& a, std::vector<double>& b) {

    // std::cout << "DOT PRODUCT!" << std::endl;
    double sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
        // std::cout << a[i] << " " << b[i] << " " << sum << std::endl;
    }

    return sum;
}

inline std::vector<double> softmax(std::vector<double>& logits) {

    // std::cout << "SOFTMAXXING" << std::endl;

    double max_logit = 0;
    for (double& c : logits) max_logit = std::max(max_logit, c);

    std::vector<double> out(logits.size());

    double sum_e_x = 0.0;

    for (int i = 0; i < logits.size(); i++) {
        out[i] = exp(logits[i] - max_logit);
        sum_e_x += out[i];
    }

    for (int i = 0; i < logits.size(); i++) {
        out[i] /= sum_e_x;
    }

    return out;
}


inline double cross_entropy_loss(std::vector<double>& predicted, std::vector<double>& actual) {
    // std::cout << "CROSS ENTROPY LOSS" << std::endl;

    double epsilon = 0.000000001;
    double loss = 0;

    for (int i = 0; i < actual.size(); i++) {
        // std::cout << actual[i] << " " << predicted[i] << std::endl;
        loss -= actual[i] * std::log(std::max(predicted[i], epsilon));
    }

    return loss;
}

inline int quantize(double x) {
    return std::clamp<int>(static_cast<int>(round(x * 64)), -128, 127);
}

inline std::string get_weight_str(double x) {
    int a = quantize(x) + 128;
    if (a >= 100) return std::to_string(a);

    if (a >= 10) return "0" + std::to_string(a);

    return "00" + std::to_string(a);
}

inline std::vector<double> get_inputs(PolicyData& policy_data) {
    std::vector<double> inputs(96 * 3, 0);
    for (int i = 0; i < 96 * 3; i++) {
        inputs[i] = static_cast<double>(policy_data.get_bit(i));
    }

    return inputs;
}

struct Gradient {
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    inline Gradient() {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            weights.emplace_back();
            for (int j = 0; j < INPUT_SIZE; j++) {
                weights[i].push_back(0);
            }
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            bias.push_back(0);
        }
    }
};


class NeuralNetwork {
public:

    inline NeuralNetwork() {

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(lower_bound, upper_bound);

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            weights.emplace_back();
            for (int j = 0; j < INPUT_SIZE; j++) {
                weights[i].push_back(dist(e2));
                // std::cout << "WEIGHTS: " << weights[i][j] << std::endl;
            }
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            bias.push_back(dist(e2));
            // std::cout << "BIAS: " << bias[i] << std::endl;
        }
    }

    double lower_bound = -1.98;
    double upper_bound =  1.98;

    std::vector<std::vector<double>> weights;

    std::vector<double> bias;

    std::vector<double> activation;

    std::vector<double> feedforward(std::vector<double>& inputs);

    void update_weights(Gradient& gradient, int batch_size);

    double back_propagation(Gradient& gradient, std::vector<double>& inputs, std::vector<double>& actual, double lr);

    void train();

    void save_net();
};