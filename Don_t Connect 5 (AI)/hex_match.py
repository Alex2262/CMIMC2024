

from bot.strategy import strategy
from bot.strategy_dev import strategy_dev
from bot.strategy_dev3 import strategy_dev3
from bot.greedy_bot_move import greedy_bot_move
from bot.random_bot_move import random_bot_move

from hex_func import run_game


MAX_GAMES = 1000


def main():
    bot_list = {"strategy_dev":strategy, "strategy":strategy, "random_bot_move":strategy_dev}

    wins = [0, 0, 0]
    tot_score = 0

    for game in range(MAX_GAMES):
        scores = run_game(bot_list)["scores"]
        print(f"Game #{game} result: {scores}")

        big = max(scores)
        sma = min(scores)

        if scores[0] == scores[1] and scores[1] == scores[2]:
            wins[0] += 0.5
            wins[1] += 0.5
            wins[2] += 0.5
        elif scores[0] == big and scores[1] == big:
            wins[0] += 0.75
            wins[1] += 0.75
        elif scores[0] == big and scores[2] == big:
            wins[0] += 0.75
            wins[2] += 0.75
        elif scores[1] == big and scores[2] == big:
            wins[1] += 0.75
            wins[2] += 0.75
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

        tot_score += 1
        print(f"RESULTS: {wins}, {wins[0] / tot_score, wins[1] / tot_score, wins[2] / tot_score}")

        print(f"DEV RATIO: {wins[0] / max(float(wins[1]), 0.001)}")


if __name__ == "__main__":
    main()
