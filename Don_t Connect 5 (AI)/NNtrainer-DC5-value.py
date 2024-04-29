import random
import threading

import math

from hex_match import Results, run_thread
from hex_func import get_coordinates
from bot.strategy import strategy


real_coordinates = get_coordinates()

THREAD_COUNT = 8


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def ReLU(x):
    return max(x, 0)


def ReLU_derivative(x):
    return 1 if x > 0 else 0


def dot_product(v1, v2):

    return sum(v1[i] * v2[i] for i in range(len(v1)))


def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]


def cross_entropy_loss(predicted, actual):
    # Adding a small constant to avoid log(0)
    epsilon = 1e-15
    loss = 0

    for i in range(len(actual)):
        loss -= actual[i] * math.log(predicted[i] + epsilon)

    return loss / len(actual)


def clamp(x, lower_bound, upper_bound):
    return max(min(x, upper_bound), lower_bound)


def quantize(x):
    return clamp(round(x * 64), -128, 127)


def get_weight(ws, idx):
    s = ws[idx] + ws[idx + 1] + ws[idx + 2]
    return (int(s) - 128.0) / 64.0


def get_char(x):
    a = quantize(x) + 128
    if a >= 100:
        return str(a)

    if a >= 10:
        return "0" + str(a)

    return "00" + str(a)


def get_inputs(board, player_to_move):
    inputs = [0] * (96 * 3)

    player_map = [0, 1, 2]

    if player_to_move == 1:
        player_map = [2, 0, 1]
    elif player_to_move == 2:
        player_map = [1, 2, 0]

    for i in range(96):
        if real_coordinates[i] in board:
            player = player_map[board[real_coordinates[i]]]
            inputs[player * 96 + i] = 1

    return inputs


def get_wdl(scores):
    big = max(scores)
    sma = min(scores)

    wins = [0, 0, 0]
    if scores[0] == scores[1] and scores[1] == scores[2]:
        wins[0] = 1.0 / 3.0
        wins[1] = 1.0 / 3.0
        wins[2] = 1.0 / 3.0
    elif scores[0] == big and scores[1] == big:
        wins[0] = 0.5
        wins[1] = 0.5
    elif scores[0] == big and scores[2] == big:
        wins[0] = 0.5
        wins[2] = 0.5
    elif scores[1] == big and scores[2] == big:
        wins[1] = 0.5
        wins[2] = 0.5
    else:
        for i in range(3):
            if scores[i] == big:
                wins[i] += 2.0 / 3.0
                break

        small_count = 0
        for i in range(3):
            if scores[i] == sma:
                small_count += 1

        if small_count == 1:
            for i in range(3):
                if scores[i] != sma:
                    wins[i] += 1.0 / 3.0
                    break
        elif small_count == 2:
            for i in range(3):
                if scores[i] == sma:
                    wins[i] += 1.0 / 6.0

    # print(scores, wins)
    return wins


class NeuralNetwork:
    def __init__(self, input_size, hl_size, output_size):
        self.input_size = input_size
        self.hl_size = hl_size
        self.output_size = output_size

        self.lower_bound = -1.98
        self.upper_bound = 1.98

        self.weights_feature = [[random.uniform(self.lower_bound, self.upper_bound) for _ in range(input_size)]
                                for _ in range(hl_size)]
        self.weights_output = [[random.uniform(self.lower_bound, self.upper_bound) for _ in range(hl_size)]
                               for _ in range(output_size)]

        # print(self.weights_output)
        # print(self.output_size)

        self.bias_hidden = [random.uniform(self.lower_bound, self.upper_bound) for _ in range(hl_size)]
        self.bias_output = [random.uniform(self.lower_bound, self.upper_bound) for _ in range(output_size)]

        # print(self.weights_feature, self.weights_output, self.bias_hidden, self.bias_output)

        self.hidden_activation = None
        self.output_activation = None

    def feedforward(self, inputs):
        # Hidden layer
        self.hidden_activation = [ReLU(dot_product(inputs, self.weights_feature[i]) + self.bias_hidden[i])
                                   for i in range(self.hl_size)]

        # Output layer
        self.output_activation = [dot_product(self.hidden_activation, self.weights_output[i]) + self.bias_output[i]
                                  for i in range(self.output_size)]

        return softmax(self.output_activation)

    def update_weights(self, gradient):
        for i in range(self.hl_size):
            for j in range(self.input_size):
                self.weights_feature[i][j] -= gradient.weights_feature[i][j]
                self.weights_feature[i][j] = clamp(self.weights_feature[i][j], self.lower_bound, self.upper_bound)

        for i in range(self.output_size):
            for j in range(self.hl_size):
                self.weights_output[i][j] -= gradient.weights_output[i][j]
                self.weights_output[i][j] = clamp(self.weights_output[i][j], self.lower_bound, self.upper_bound)

        for i in range(self.hl_size):
            self.bias_hidden[i] -= gradient.bias_hidden[i]
            self.bias_hidden[i] = clamp(self.bias_hidden[i], self.lower_bound, self.upper_bound)

        for i in range(self.output_size):
            self.bias_output[i] -= gradient.bias_output[i]
            self.bias_output[i] = clamp(self.bias_output[i], self.lower_bound, self.upper_bound)

    def init_weights(self, ws):
        idx = 0

        for i in range(self.hl_size):
            for j in range(self.input_size):
                self.weights_feature[i][j] = get_weight(ws, idx)
                idx += 3

        for i in range(self.output_size):
            for j in range(self.hl_size):
                self.weights_output[i][j] = get_weight(ws, idx)
                idx += 3

        for i in range(self.hl_size):
            self.bias_hidden[i] = get_weight(ws, idx)
            idx += 3

        for i in range(self.output_size):
            self.bias_output[i] = get_weight(ws, idx)
            idx += 3


class Gradient:
    def __init__(self, input_size, hl_size, output_size):
        self.input_size = input_size
        self.hl_size = hl_size
        self.output_size = output_size

        self.weights_feature = [[0 for _ in range(input_size)] for _ in range(hl_size)]
        self.weights_output = [[0 for _ in range(hl_size)] for _ in range(output_size)]

        self.bias_hidden = [0 for _ in range(hl_size)]
        self.bias_output = [0 for _ in range(output_size)]


class RL:
    def __init__(self, nn):
        self.nn = nn
        self.all_data = []

    def backprop(self, gradient, inputs, actual, lr):
        predicted = self.nn.feedforward(inputs)

        cel_error = cross_entropy_loss(predicted, actual)
        d_wrt_logit = [(predicted[i] - actual[i]) for i in range(self.nn.output_size)]

        # print(mse_error)

        # BACK PROPAGATION

        # Update weights and biases from hidden to output
        for i in range(self.nn.output_size):
            for j in range(self.nn.hl_size):
                # gradient = lr * loss derivative * h
                gradient.weights_output[i][j] += lr * self.nn.hidden_activation[j] \
                                                * d_wrt_logit[i]

            # similar to weights but no need to consider activations
            gradient.bias_output[i] += lr * d_wrt_logit[i]

        # Update weights and biases from input to hidden
        for i in range(self.nn.hl_size):

            error_contribution = sum([d_wrt_logit[j] * self.nn.weights_output[j][i]
                                      for j in range(self.nn.output_size)]) \
                                 * ReLU_derivative(self.nn.hidden_activation[i])

            for j in range(self.nn.input_size):
                gradient.weights_feature[i][j] += lr * error_contribution * inputs[j]

            # similar to weights but no need to consider activations
            gradient.bias_hidden[i] += lr * error_contribution

        return cel_error

    def gen_data(self, games):

        bot_list_1 = {"strategy1": strategy, "strategy2": strategy, "strategy3": strategy}
        bot_list_2 = {"strategy2": strategy, "strategy1": strategy, "strategy3": strategy}

        game_per_thread = int(games / THREAD_COUNT)

        print(game_per_thread)

        results = Results()
        threads = []

        for thread_id in range(THREAD_COUNT):
            print(f"RUNNING THREAD {thread_id + 1}")
            threads.append(threading.Thread(target=run_thread,
                                            args=(thread_id, game_per_thread, results, bot_list_1, bot_list_2)))
            threads[-1].start()

        for thread_id in range(THREAD_COUNT):
            threads[thread_id].join()

        self.all_data = []
        for g_hist in results.game_hist:
            for res in g_hist:
                game = res["game"]
                # all_res.append(res)

                wdl = get_wdl(res["scores"])

                actual = [wdl, [wdl[1], wdl[2], wdl[0]], [wdl[2], wdl[0], wdl[1]]]

                # print(actual)

                board = {}
                player = 0

                for pair in game:
                    inputs = get_inputs(board, player)
                    self.all_data.append((inputs, actual[player]))

                    move = pair[1]
                    if move is not None:
                        board[move] = player

                    player = (player + 1) % 3

        print(f"DATA SIZE: {len(self.all_data)}")
        random.shuffle(self.all_data)

        print("DATA SHUFFLED")
        self.save_data()

    def save_data(self):
        print("SAVING DATA")
        f = open("new_data.txt", "w")
        for res in self.all_data:
            # print(res[1])

            string = "".join(list(map(str, res[0]))) + "|" \
                     + str(res[1])

            f.write(string + "\n")

        f.close()
        print("SAVED DATA")

    def train(self, epochs):

        for epoch in range(epochs):

            loss = 0
            accum_gradient = Gradient(INPUT_SIZE, HL_SIZE, OUTPUT_SIZE)

            for res in self.all_data:
                error = self.backprop(accum_gradient, res[0], res[1], DEFAULT_LR / len(self.all_data))
                loss += error

            self.nn.update_weights(accum_gradient)

            print(f"EPOCH {epoch + 1} completed!")
            print(f"Loss of: {loss}")

        self.save_net()

    def save_net(self):

        print("SAVING NET")

        weights_feature_string = ""
        weights_output_string = ""
        bias_hidden_string = ""
        bias_output_string = ""

        for i in range(self.nn.hl_size):
            for j in range(self.nn.input_size):
                weights_feature_string += get_char(self.nn.weights_feature[i][j])

        for i in range(self.nn.output_size):
            for j in range(self.nn.hl_size):
                weights_output_string += get_char(self.nn.weights_output[i][j])

        for i in range(self.nn.hl_size):
            bias_hidden_string += get_char(self.nn.bias_hidden[i])

        for i in range(self.nn.output_size):
            bias_output_string += get_char(self.nn.bias_output[i])

        f = open("nn-weights.txt", "w")
        f.write(weights_feature_string + weights_output_string + bias_hidden_string + bias_output_string + "\n")

        f.close()

        print("SAVED NET")


def main():
    cool_network = NeuralNetwork(INPUT_SIZE, HL_SIZE, OUTPUT_SIZE)
    trainer = RL(cool_network)

    trainer.gen_data(games=THREAD_COUNT * 1800)
    trainer.train(epochs=40)

    # print(cool_network.weights_feature, cool_network.weights_output, cool_network.bias_hidden,
    #       cool_network.bias_output)


def gen_data():
    cool_network = NeuralNetwork(INPUT_SIZE, HL_SIZE, OUTPUT_SIZE)
    trainer = RL(cool_network)

    trainer.gen_data(games=THREAD_COUNT * 500)


DEFAULT_LR = 0.01
INPUT_SIZE = 96 * 3
HL_SIZE = 8
OUTPUT_SIZE = 3


if __name__ == "__main__":
    main()
