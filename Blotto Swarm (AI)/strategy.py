"""
Edit this file! This is the file you will submit.
"""

import random

import math

"""
NOTE: Each soldier's memory in the final runner will be separate from the others.

WARNING: Do not print anything to stdout. It will break the grading script!
"""

LOWER_BOUND = -5
UPPER_BOUND = 5


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

def strategy2(ally: list, enemy: list, offset: int) -> int:
    THRESHOLD = 2
    D_PROB = 5  # D_PROB/100 -- Castle Dispersion Probability

    # Strategy 1: 0.6785
    # Strategy 2: 0.3215

    # ON A CASTLE
    if offset % 3 == 0:

        current = ally[3] - enemy[3]

        # disperse allies
        if current >= THRESHOLD:
            p = random.randint(0, 100)

            x = current - THRESHOLD

            dispersion_probability = 22 * math.tanh(1/6 * (x - 5)) + 21

            if 0 <= p < dispersion_probability:
                return -1

            # Disperse Right
            if dispersion_probability <= p < 2 * dispersion_probability:
                return 1

        # Stay on the castle to reinforce it
        return 0

    left = 0
    right = 0

    # CASTLE IS TO THE LEFT
    if offset % 3 == 1:
        left = ally[2] - enemy[2]
        right = ally[5] - enemy[5]

    # CASTLE IS TO THE RIGHT
    if offset % 3 == 2:
        left = ally[1] - enemy[4]
        right = ally[1] - enemy[4]

    diff = abs(left - right)
    if left < right:
        left_prob = 50 + diff * 7
        r = random.randint(0, 100)

        if r < left_prob:
            return -1
        else:
            return 1

    else:
        right_prob = 50 + diff * 7
        r = random.randint(0, 100)

        if r < right_prob:
            return 1
        else:
            return -1


def offset_strategy(alli: list, enemy: list, offset: int) -> int:
    return offset


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
    strategies = [strategy2, offset_strategy]

    return strategies
