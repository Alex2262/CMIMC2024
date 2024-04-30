#include <iostream>
#include "policy_trainer.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    NeuralNetwork nn = NeuralNetwork();

    nn.train();
}
