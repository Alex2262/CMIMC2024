"""
Edit this file! This is the file you will submit.
"""

import random

"""
NOTE: Each soldier's memory in the final runner will be separate from the others.

WARNING: Do not print anything to stdout. It will break the grading script!
"""

LOWER_BOUND = -12
UPPER_BOUND = 12


def strategy(ally: list, enemy: list, offset: int) -> int:

    # ON A CASTLE
    if offset % 3 == 0:

        current = ally[3] - enemy[3]

        # DETERMINE A RANGE TO STAY ON THIS CASTLE
        if LOWER_BOUND <= current <= UPPER_BOUND:
            return 0

        left = ally[0] - enemy[0]
        right = ally[6] - enemy[6]

        if LOWER_BOUND <= left <= UPPER_BOUND:
            return -1

        if LOWER_BOUND <= right <= UPPER_BOUND:
            return 1

        return 0

    # CASTLE IS TO THE LEFT
    if offset % 3 == 1:
        left = ally[2] - enemy[2]
        right = ally[5] - enemy[5]

        if LOWER_BOUND <= left <= UPPER_BOUND:
            return -1

        if LOWER_BOUND <= right <= UPPER_BOUND:
            return 1

        return -1

    # CASTLE IS TO THE RIGHT
    if offset % 3 == 2:
        left = ally[1] - enemy[1]
        right = ally[4] - enemy[4]

        if LOWER_BOUND <= right <= UPPER_BOUND:
            return 1

        if LOWER_BOUND <= left <= UPPER_BOUND:
            return -1

        return 1


def random_strategy(ally: list, enemy: list, offset: int) -> int:
    # A random strategy to use in your game
    return random.randint(-1, 1)


def get_strategies():
    """
    Returns a list of strategies to play against each other.

    In the local tester, all of the strategies will be used as separate players, and the 
    pairwise winrate will be calculated for each strategy.

    In the official grader, only the first element of the list will be used as your strategy.
    """
    strategies = [strategy, random_strategy, random_strategy]

    return strategies
