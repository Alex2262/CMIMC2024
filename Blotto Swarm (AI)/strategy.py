"""
Edit this file! This is the file you will submit.
"""

import random

import math

"""
NOTE: Each soldier's memory in the final runner will be separate from the others.

WARNING: Do not print anything to stdout. It will break the grading script!
"""

day = 0

def left_or_right(left, right):
    GIVE_UP_MARGIN = -5

    if left <= GIVE_UP_MARGIN and right <= GIVE_UP_MARGIN:
        return -1 if left > right else 1

    if left <= GIVE_UP_MARGIN:
        return 1

    if right <= GIVE_UP_MARGIN:
        return -1

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


def strategy(ally: list, enemy: list, offset: int) -> int:
    # global day
    # day += 1

    # print(day)

    THRESHOLD = 2
    ALLY_THRESHOLD = 0

    # Strategy 1: 0.6785
    # Strategy 2: 0.3215

    # ON A CASTLE
    if offset == 0:

        # Close to the end of the game, stay where you are
        if day >= 98:
            return 0

        current = ally[3] - enemy[3]
        left = ally[0] - enemy[0]
        right = ally[6] - enemy[6]

        need_left = max(1 - left, 0)
        need_right = max(1 - right, 0)

        # GIVE UP
        if ally[3] < need_left + need_right + 3 and ally[3] <= enemy[3] and enemy[3] >= 6:

            if need_left + need_right == 0:
                p_left = 50
            else:
                p_left = (100 * need_left) / (need_left + need_right)

            if random.randint(0, 100) < p_left:
                return -1

            return 1

        # disperse allies
        if current >= THRESHOLD:

            p = random.randint(0, 100)

            x = current - THRESHOLD

            day_scaler = 50 * math.tanh(-1/5 * (day - 100)) + 50
            if day <= 10:
                day_scaler = -day + 110

            dispersion_probability = (22 * math.tanh(1/6 * (x - 5)) + 21) * (0.01 * day_scaler)

            if p >= 2.1 * dispersion_probability:
                return 0

            if left <= ALLY_THRESHOLD and right <= ALLY_THRESHOLD:
                return left_or_right(left, right)

            if left <= ALLY_THRESHOLD:
                return -1

            if right <= ALLY_THRESHOLD:
                return 1

            if p >= dispersion_probability * 1.3:
                return 0

            largest  = max(current, left, right)
            smallest = min(current, left, right)

            # Just stay.
            if largest == smallest or current == smallest:
                return 0

            # Not worth it
            if left <= -5 or right <= -5:
                return 0

            if current == largest:
                return left_or_right(left, right)

            # Left is smallest, then comes middle castle
            if left == smallest:
                lr = left_or_right(left, current)
                if lr == -1:
                    return -1
                return 0

            # Right is smallest, then comes middle castle
            lr = left_or_right(left, current)
            if lr == 1:
                return 1
            return 0

        # Stay on the castle to reinforce it
        return 0

    left = 0
    right = 0

    if day >= 98:
        if offset == 1:
            return -1

        return 1

    # CASTLE IS TO THE LEFT
    if offset == 1:
        left = ally[2] - enemy[2]
        right = ally[5] - enemy[5]

    # CASTLE IS TO THE RIGHT
    if offset == -1:
        left = ally[1] - enemy[4]
        right = ally[1] - enemy[4]

    return left_or_right(left, right)


def left_or_right2(left, right):
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


def strategy2(ally: list, enemy: list, offset: int) -> int:
    # global day
    # day += 1

    # print(day)

    THRESHOLD = 2
    ALLY_THRESHOLD = 0

    # Strategy 1: 0.6785
    # Strategy 2: 0.3215

    # ON A CASTLE
    if offset == 0:

        # Close to the end of the game, stay where you are
        if day >= 98:
            return 0

        current = ally[3] - enemy[3]

        # disperse allies
        if current >= THRESHOLD:
            p = random.randint(0, 100)

            x = current - THRESHOLD

            day_scaler = 50 * math.tanh(-1/5 * (day - 100)) + 50
            if day <= 10:
                day_scaler = -day + 110

            dispersion_probability = (22 * math.tanh(1/6 * (x - 5)) + 21) * (0.01 * day_scaler)

            if p >= 2.1 * dispersion_probability:
                return 0

            left = ally[0] - enemy[0]
            right = ally[6] - enemy[6]

            if left <= ALLY_THRESHOLD and right <= ALLY_THRESHOLD:
                return left_or_right2(left, right)

            if left <= ALLY_THRESHOLD:
                return -1

            if right <= ALLY_THRESHOLD:
                return 1

            if p >= dispersion_probability * 1.3:
                return 0

            largest  = max(current, left, right)
            smallest = min(current, left, right)

            # Just stay.
            if largest == smallest or current == smallest:
                return 0

            if current == largest:
                return left_or_right2(left, right)

            # Left is smallest, then comes middle castle
            if left == smallest:
                lr = left_or_right2(left, current)
                if lr == -1:
                    return -1
                return 0

            # Right is smallest, then comes middle castle
            lr = left_or_right2(left, current)
            if lr == 1:
                return 1
            return 0

        # Stay on the castle to reinforce it
        return 0

    left = 0
    right = 0

    if day >= 98:
        if offset == 1:
            return -1

        return 1

    # CASTLE IS TO THE LEFT
    if offset == 1:
        left = ally[2] - enemy[2]
        right = ally[5] - enemy[5]

    # CASTLE IS TO THE RIGHT
    if offset == -1:
        left = ally[1] - enemy[4]
        right = ally[1] - enemy[4]

    return left_or_right2(left, right)


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
    strategies = [strategy2, strategy, offset_strategy]

    return strategies
