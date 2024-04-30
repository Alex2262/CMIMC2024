//
// Created by Alexander Tian on 4/26/24.
//

#include <iostream>
#include <fstream>
#include "policy_trainer.h"


std::vector<double> NeuralNetwork::feedforward(std::vector<double>& inputs) {
    activation.clear();
    activation.reserve(OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (i < 96 && (inputs[i] == 1 || inputs[96 + i] == 1 || inputs[96 * 2 + i] == 1)) {
            activation.push_back(-100'000'000);
            continue;
        }

        activation.push_back(ReLU(dot_product(inputs, weights[i]) + bias[i]));
    }

    return softmax(activation);
}


void NeuralNetwork::update_weights(Gradient& gradient, int batch_size) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights[i][j] -= gradient.weights[i][j] / batch_size;
            weights[i][j] = std::clamp(weights[i][j], lower_bound, upper_bound);
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias[i] -= gradient.bias[i] / batch_size;
        bias[i] = std::clamp(bias[i], lower_bound, upper_bound);
    }
}

double NeuralNetwork::back_propagation(Gradient& gradient, std::vector<double>& inputs, std::vector<double>& actual, double lr) {

    // std::cout << "LR: " << lr << std::endl;
    auto predicted = feedforward(inputs);

    double cel_error = cross_entropy_loss(predicted, actual);
    std::vector<double> d_wrt_logit(OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) d_wrt_logit[i] = predicted[i] - actual[i];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            gradient.weights[i][j] += lr * inputs[j] * d_wrt_logit[i];
        }

        gradient.bias[i] += lr * d_wrt_logit[i];
    }

    return cel_error;
}

void NeuralNetwork::save_net() {

    std::string save_code;

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            save_code += get_weight_str(weights[i][j]);
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) save_code += get_weight_str(bias[i]);

    std::ofstream file;
    file.open("nn-weights.txt");

    file << save_code;

    file.close();

    std::cout << "NET SAVED" << std::endl;
}

void NeuralNetwork::train() {
    CombinedData combined_data{};

    gen_data(combined_data);

    std::vector<PolicyData> all_data{};

    for (int i = 0; i < THREADS; i++) {
        for (auto& pd : combined_data.thread_data[i]) {
            all_data.push_back(pd);
        }
    }

    std::cout << "SHUFFLING DATA" << std::endl;

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    std::shuffle(all_data.begin(), all_data.end(), rng);

    std::cout << "ALL DATA SHUFFLED" << std::endl;
    std::cout << "TOTAL " << all_data.size() << " DATAS" << std::endl;

    double lr = DEFAULT_LR;

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {

        if (epoch % 10 == 0) {  // Reduce learning rate every 10 epochs
            lr *= 0.9;  // Reduce learning rate by 10%
        }

        double loss = 0;
        Gradient accum_gradient = Gradient();

        for (auto& policy_data : all_data) {
            auto inputs = get_inputs(policy_data);

            // auto pb = policy_data.get_board();

            std::vector<double> actual(97);
            double total_visits = 0;
            for (int i = 0; i < 97; i++) {
                actual[i] = policy_data.visits[i];
                total_visits += actual[i];
            }

            for (double& e : actual) e /= total_visits;

            /*
            for (int i = 0; i < 96; i++) {
                Coord coord = all_coordinates[i];
                print_board(pb, {coord, false});
                std::cout << actual[i] << std::endl;
            }
            */

            loss += back_propagation(accum_gradient, inputs, actual, lr);
        }

        update_weights(accum_gradient, all_data.size());

        std::cout << "EPOCH " << epoch << " completed!" << std::endl;
        std::cout << "Loss: " << loss / all_data.size() << std::endl;
    }

    save_net();
}