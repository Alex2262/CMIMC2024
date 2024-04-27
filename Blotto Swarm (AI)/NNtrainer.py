import random
import math

from grader import BlottoSwarmGame
from strategy import strategy, offset_strategy, random_strategy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(input_nodes)] for _ in range(hidden_nodes)]
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(hidden_nodes)] for _ in range(output_nodes)]

        self.biases_hidden = [random.uniform(-1, 1) for _ in range(hidden_nodes)]
        self.biases_output = [random.uniform(-1, 1) for _ in range(output_nodes)]

        self.hidden_activations = None
        self.output_activations = None

    def feedforward(self, inputs):
        # Hidden layer
        self.hidden_activations = [sigmoid(dot_product(inputs, self.weights_input_hidden[i]) + self.biases_hidden[i]) for i in
                  range(self.hidden_nodes)]

        # Output layer
        self.output_activations = [sigmoid(dot_product(self.hidden_activations, self.weights_hidden_output[i]) + self.biases_output[i]) for i in
                  range(self.output_nodes)]

        return self.output_activations

    def softmax(self, x):
        e_x = [math.exp(i) for i in x]
        sum_e_x = sum(e_x)
        return [i / sum_e_x for i in e_x]

    def choose_action(self, output):
        probabilities = self.softmax(output)
        return random.choices([-1, 0, 1], weights=probabilities, k=1)[0]

    def get_inputs(self, ally, enemy, offset):
        inputs = []

        for e in ally:
            inputs.append(e / 60)

        for e in enemy:
            inputs.append(e / 60)

        offset_one_hot = [0, 0, 0]
        offset_one_hot[offset + 1] = 1

        inputs += offset_one_hot
        return inputs

    def get_move(self, ally, enemy, offset):
        inputs = self.get_inputs(ally, enemy, offset)
        output = self.feedforward(inputs)

        action = self.choose_action(output)
        return action


class RL:
    def __init__(self, nn):
        self.nn = nn

    def update_weights(self, inputs, chosen_action, reward, lr=0.01):
        predicted_output = self.nn.feedforward(inputs)

        expected_output = [0, 0, 0]
        expected_output[chosen_action + 1] = reward  # adjusting index for action -1, 0, 1 to 0, 1, 2

        # Calculate errors for output layer
        mse_error = [(expected_output[i] - predicted_output[i]) ** 2 for i in range(self.nn.output_nodes)]
        mse_derivative = [2 * (predicted_output[i] - expected_output[i]) for i in range(self.nn.output_nodes)]

        # BACK PROPAGATION

        # Update weights and biases from hidden to output
        for i in range(self.nn.output_nodes):
            for j in range(self.nn.hidden_nodes):

                # gradient = lr * loss derivative * h * sigmoid'(x) --> sigmoid(x) * (1-sigmoid(x))
                self.nn.weights_hidden_output[i][j] -= lr * self.nn.hidden_activations[j] \
                                                       * predicted_output[i] * (1 - predicted_output[i]) \
                                                       * mse_derivative[i]

            # similar to weights but no need to consider activations
            self.nn.biases_output[i] -= lr * predicted_output[i] * (1 - predicted_output[i]) * mse_derivative[i]

        # Update weights and biases from input to hidden
        for i in range(self.nn.hidden_nodes):

            # Error Contribution of ith hidden neuron =
            # Sum of output error * Weight from i-th hidden neuron to k-th output neuron
            # Whole sum * derivative of activation function with respect to this hidden neuron's activation

            error_contribution = sum(
                mse_derivative[k] * self.nn.weights_hidden_output[k][i] for k in range(self.nn.output_nodes)
            ) * self.nn.hidden_activations[i] * (1 - self.nn.hidden_activations[i])

            for j in range(self.nn.input_nodes):
                self.nn.weights_input_hidden[i][j] -= lr * error_contribution * inputs[j]

            self.nn.biases_hidden[i] -= lr * error_contribution

        return mse_error

    def train(self, games, epochs):

        NUM_DAYS = 100

        two_strategies = [self.nn.get_move, offset_strategy]

        for epoch in range(epochs):

            scores = [0, 0]

            game_hist = []
            results = []

            for game_number in range(games):

                game_hist.append([])

                game = BlottoSwarmGame()

                for day in range(1, NUM_DAYS + 1):

                    moves = [[], []]

                    for team, func in enumerate(two_strategies):
                        for i in range(BlottoSwarmGame.NUM_SOLDIERS):
                            ally, enemy, offset = game.state(team, i)

                            move = func(ally, enemy, offset)

                            if team == 0:
                                game_hist[game_number].append(((ally, enemy, offset), move))

                            assert move in [-1, 0, 1]

                            moves[team].append(move)

                    for team in range(2):
                        for i, move in enumerate(moves[team]):
                            game.move(team, i, move)

                    game.calc_score()

                game_score = game.scores()
                # print(game_score)
                reward = 0
                if game_score[0] > game_score[1]:
                    reward = 1
                    scores[0] += 1
                elif game_score[1] > game_score[0]:
                    scores[1] += 1
                    reward = -1
                else:
                    reward = -0.5
                    scores[0] += 0.5
                    scores[1] += 0.5

                results.append(reward)

            loss = 0
            for game_number in range(games):
                for day in range(NUM_DAYS):
                    inputs = self.nn.get_inputs(*game_hist[game_number][day][0])
                    chosen = game_hist[game_number][day][1]

                    reward = results[game_number]

                    error = self.update_weights(inputs, chosen, reward)
                    loss += sum(error)

            print(scores)
            print(f"EPOCH {epoch} completed!")
            print(f"Loss of: {loss}")


def main():
    cool_network = NeuralNetwork(17, 100, 3)
    trainer = RL(cool_network)

    trainer.train(games=10, epochs=200)

    print(cool_network.weights_input_hidden, cool_network.weights_hidden_output, cool_network.biases_hidden,
          cool_network.biases_output)


if __name__ == "__main__":
    main()
