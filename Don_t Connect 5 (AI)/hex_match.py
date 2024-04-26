import threading
from multiprocessing import Process, Queue

import math

from bot.strategy import strategy
from bot.strategy_dev import strategy_dev
from bot.strategy_dev3 import strategy_dev3
from bot.greedy_bot_move import greedy_bot_move
from bot.random_bot_move import random_bot_move

from hex_func import run_game_process


MAX_GAMES = 10000

THREADS = 8


class Results:
    def __init__(self):

        self.tot_wins = [[0, 0, 0] for _ in range(THREADS)]
        self.tot_games = [0 for _ in range(THREADS)]

        self.game_hist = [[] for _ in range(THREADS)]

    def get_combined_games(self):
        return sum(self.tot_games)

    def get_wins(self, player):
        return sum([self.tot_wins[i][player] for i in range(THREADS)])

    def get_combined_wins(self):
        return [self.get_wins(player) for player in range(3)]


def run_thread(thread_id, max_games, results, bot_list_1, bot_list_2, full_print=False):

    for match in range(math.ceil(max_games / 2)):
        for rep in range(2):

            if rep == 0:
                bot_list = bot_list_1
            else:
                bot_list = bot_list_2

            res_queue = Queue()
            process = Process(target=run_game_process, args=(bot_list, res_queue))
            process.start()
            process.join()

            res = res_queue.get()
            scores = res["scores"]

            if full_print:
                if rep == 0:
                    print(f"Game #{results.get_combined_games() + 1} result: {scores} with Dev vs Master")
                else:
                    print(f"Game #{results.get_combined_games() + 1} result: {scores} with Master vs Dev")
            else:
                print(f"Game #{results.get_combined_games() + 1} completed")

            big = max(scores)
            sma = min(scores)

            wins = [0, 0, 0]
            if scores[0] == scores[1] and scores[1] == scores[2]:
                wins[0] = 0.5
                wins[1] = 0.5
                wins[2] = 0.5
            elif scores[0] == big and scores[1] == big:
                wins[0] = 0.75
                wins[1] = 0.75
            elif scores[0] == big and scores[2] == big:
                wins[0] = 0.75
                wins[2] = 0.75
            elif scores[1] == big and scores[2] == big:
                wins[1] = 0.75
                wins[2] = 0.75
            else:
                for i in range(3):
                    if scores[i] == big:
                        wins[i] += 1
                        break

                small_count = 0
                for i in range(3):
                    if scores[i] == sma:
                        small_count += 1

                for i in range(3):
                    if scores[i] == sma:
                        wins[i] += 0.5 - 0.5 / small_count

            if rep == 0:
                results.tot_wins[thread_id][0] += wins[0]
                results.tot_wins[thread_id][1] += wins[1]
                results.tot_wins[thread_id][2] += wins[2]
            else:
                results.tot_wins[thread_id][0] += wins[1]
                results.tot_wins[thread_id][1] += wins[0]
                results.tot_wins[thread_id][2] += wins[2]

            results.tot_games[thread_id] += 1
            results.game_hist[thread_id].append(res)
            # print(thread_id, results.tot_wins, results.tot_games)
            cg = results.get_combined_games()

            if full_print:
                print(f"RESULTS: {results.get_combined_wins()}, {results.get_wins(0) / cg, results.get_wins(1) / cg, results.get_wins(2) / cg}")

                print(f"DEV RATIO: {results.get_wins(0) / max(float(results.get_wins(1)), 0.001)}")


def main():
    results = Results()

    bot_list_1 = {"strategy_dev": strategy_dev, "strategy": strategy, "third_player": strategy_dev3}
    bot_list_2 = {"strategy": strategy, "strategy_dev": strategy_dev, "third_player": strategy_dev3}

    threads = []

    for thread_id in range(THREADS):
        print(f"RUNNING THREAD {thread_id + 1}")
        threads.append(threading.Thread(target=run_thread,
                                        args=(thread_id, MAX_GAMES, results, bot_list_1, bot_list_2, True)))
        threads[-1].start()


if __name__ == "__main__":
    main()
