import random
import time

import math

GRID_RADIUS = 3
coordinates = []

ALL_NEIGHBOR = lambda x1, y1, z1: (
    (x1 + 1, y1, z1), (x1 - 1, y1, z1), (x1, y1 + 1, z1), (x1, y1 - 1, z1), (x1, y1, z1 + 1),
    (x1, y1, z1 - 1))  # make this more efficient?

SELECT_VALID = lambda lis: [(x1, y1, z1) for (x1, y1, z1) in lis if
                            1 <= x1 + y1 + z1 <= 2 and -GRID_RADIUS + 1 <= x1 <= GRID_RADIUS and -GRID_RADIUS + 1 <= y1 <= GRID_RADIUS and -GRID_RADIUS + 1 <= z1 <= GRID_RADIUS]  # keep those within-bound

TABLE = {1: 0, 2: 0, 3: 1, 4: 3, 5: 0}


def set_node_coordinates():
    global coordinates

    for x in range(-GRID_RADIUS + 1, GRID_RADIUS + 1):
        for y in range(-GRID_RADIUS + 1, GRID_RADIUS + 1):
            for z in range(-GRID_RADIUS + 1, GRID_RADIUS + 1):
                # Check if node is valid
                if 1 <= x + y + z <= 2:
                    coordinates.append((x, y, z))


set_node_coordinates()
NEIGHBOR_LIST = dict(zip(coordinates, [SELECT_VALID(ALL_NEIGHBOR(*(node))) for node in coordinates]))


def get_diameter(board, start_node, visit: dict, use_visit):
    def neighbors(node):
        # return SELECT_VALID(ALL_NEIGHBOR(*(node)))
        return NEIGHBOR_LIST[node]

    def con(node):  # Find connected component and respective degrees
        # print(node)

        if use_visit:
            visit[node] = 1

        connected[node] = -1
        cnt = 0
        for neighbor in neighbors(node):
            if neighbor in board and board[neighbor] == player:
                cnt += 1
                if neighbor not in connected:
                    con(neighbor)
        connected[node] = cnt

    def dfs(node, visited=set()):
        visited.add(node)
        max_path_length = 0
        for neighbor in neighbors(node):
            if neighbor in board and board[neighbor] == player and neighbor not in visited:
                path_length = dfs(neighbor, visited.copy())
                max_path_length = max(max_path_length, path_length)
        return max_path_length + 1

    player = board[start_node]

    '''
    try:
        player = board[start_node]
    except Exception as exc:
        print("node empty?")
        print(exc)
        return 0
    '''

    connected = dict()
    con(start_node)

    # print(connected)
    if len(connected) <= 3:  # must be a line
        return len(connected)
    if 4 <= len(connected) <= 5:  # a star if we have a deg-3 node, a line otherwise
        if 3 in connected.values():  # It's a star!
            return len(connected) - 1
        return len(connected)
    if 6 == len(connected):
        three = list(connected.values())
        if 3 in connected.values():
            three.remove(3)
            if 3 in connected.values():
                return 4  # this is a shape x - x - x - x
                #                     x   x
        return 5  # diameter is 5 otherwise

    # For the larger(>6) ones, diameter must be larger than 5 so we just return 5
    return 5

    # maxl = 0

    # for node in connected:
    #     maxl = max(maxl, dfs(node))
    # return maxl


def score(board):  # return current score for each player
    visit = {pos: 0 for pos in coordinates}
    scores = {0: 0, 1: 0, 2: 0}
    for pos in board.keys():
        if not visit[pos]:
            d = get_diameter(board, pos, visit, True)
            if d:
                scores[board[pos]] += TABLE[d]
    return scores


class Move:
    def __init__(self, x, y, z, passing):
        self.x = x
        self.y = y
        self.z = z
        self.p = passing

    def get_key(self):
        return (self.x, self.y, self.z)


class Node:
    def __init__(self, parent, last_move, pass_length):
        self.parent = parent

        self.children_start = 0
        self.children_end = 0

        self.win_count = 0
        self.visits = 1

        self.last_move = last_move

        self.pass_length = pass_length


class Tree:
    def __init__(self):
        self.graph = []


EXPLORATION_CONSTANT = 0.5
MAX_DEPTH = 35
MAX_ITERATIONS = 100_000


def hex_to_pixel(x, y, z):
    xc = round(2 * (x / 2 + y / 2 - z))
    yc = round(-(2 * math.sqrt(3)) * (x * math.sqrt(3) / 2 - y * math.sqrt(3) / 2))

    # print(x, y, z, (xc, yc))

    return xc, yc


RESET_COLOR = "\033[0m"
BLACK_COLOR = "\033[30m"
RED_COLOR = "\033[31m"
WHITE_COLOR = "\033[97m"
GREEN_COLOR = "\033[32m"


def print_board(board, move):

    mat = [[-2 for _ in range(17)] for _ in range(31)]

    for coord in coordinates:
        xc, yc = hex_to_pixel(*coord)

        # print(coord, yc, xc)
        mat[round(yc) + 15][round(xc) + 8] = -1

    for coord in board:
        xc, yc = hex_to_pixel(*coord)
        mat[round(yc) + 15][round(xc) + 8] = board[coord]

    if move is not None:
        xc, yc = hex_to_pixel(*move.get_key())
        mat[round(yc) + 15][round(xc) + 8] = 3

    for row in range(31):
        s = ""
        for col in range(17):
            if mat[row][col] == -2:
                s += " "

            elif mat[row][col] == -1:
                s += "_"

            elif mat[row][col] == 0:
                s += WHITE_COLOR + "W" + RESET_COLOR

            elif mat[row][col] == 1:
                s += BLACK_COLOR + "B" + RESET_COLOR

            elif mat[row][col] == 2:
                s += RED_COLOR + "R" + RESET_COLOR

            else:
                s += GREEN_COLOR + "G" + RESET_COLOR

        if s.isspace():
            continue

        print(s)


class MCTS:

    def __init__(self, board, player, allocated_time):

        self.board = board
        self.player = player

        self.start_time = 0
        self.iterations = 0
        self.sel_depth = 0

        self.root_node_index = 0
        self.max_time = allocated_time

        self.tree = Tree()

        # Add the root node, its parent and last move do not matter
        self.tree.graph.append(Node(-1, Move(0, 0, 0, True), 0))

    def get_result(self, current_board):
        score_dict = score(current_board)
        result = [-1, -1, -1]

        small = min(score_dict.values())
        large = max(score_dict.values())

        count_large = 0
        count_small = 0

        for i in range(3):
            if score_dict[i] == large:
                count_large += 1
            if score_dict[i] == small:
                count_small += 1

        # First place score
        large_score = 1 / count_large

        # Last place score
        # Two players are tied at last place receive a very small bonus
        # Three players tying for last place will not occur, since the first place code will run instead
        small_score = 0.15 if count_small == 2 else 0

        for i in range(3):
            if score_dict[i] == large:
                result[i] = large_score
            elif score_dict[i] == small:
                result[i] = small_score
            else:
                result[i] = 0.3

        # result has corresponding rankings

        return result

    def descend_to_root(self, node_index):
        while node_index != self.root_node_index:
            last_move = self.tree.graph[node_index].last_move

            if not last_move.p:
                del self.board[last_move.get_key()]

            node_index = self.tree.graph[node_index].parent
            self.player = ((self.player - 1) + 3) % 3

            # print(node_index, self.player, self.tree.graph[node_index].player)

    def selection(self):

        leaf_node_index = self.root_node_index

        depth = 0

        while True:

            # print(leaf_node_index)
            leaf_node = self.tree.graph[leaf_node_index]

            # print(self.player, leaf_node.player)
            # assert(self.player == leaf_node.player)

            n_children = leaf_node.children_end - leaf_node.children_start
            if n_children <= 0:
                break

            best_uct = -100000.0

            # SELECTING BEST CHILD
            for i in range(n_children):
                child_node_index = leaf_node.children_start + i
                child_node = self.tree.graph[child_node_index]

                # UCT ALGORITHM
                exploitation_value = child_node.win_count / child_node.visits
                exploration_value = math.sqrt(math.log(leaf_node.visits) / child_node.visits)

                move = child_node.last_move

                if not move.p:
                    self.board[move.get_key()] = self.player
                    d = get_diameter(self.board, move.get_key(), {}, False)
                    del self.board[move.get_key()]

                    coef = 0.2 if d >= 5 else 1
                else:
                    coef = 0.6

                uct_value = coef * (exploitation_value + EXPLORATION_CONSTANT * exploration_value)

                if uct_value > best_uct:
                    leaf_node_index = child_node_index
                    best_uct = uct_value

            last_move = self.tree.graph[leaf_node_index].last_move

            if not last_move.p:
                self.board[last_move.get_key()] = self.player

            self.player = (self.player + 1) % 3
            depth += 1

        # print("depth:", depth)
        self.sel_depth = max(self.sel_depth, depth)
        return leaf_node_index

    def expansion(self, node_index):

        node = self.tree.graph[node_index]
        node.children_start = len(self.tree.graph)

        for coord in coordinates:
            if coord in self.board:
                continue

            self.tree.graph.append(Node(node_index, Move(coord[0], coord[1], coord[2], False), 0))
            # self.board[node] = self.player
            # s = score(self.board)
            # del self.board[node]

        # Pass move
        if node.pass_length < 3:
            self.tree.graph.append(Node(node_index, Move(0, 0, 0, True), node.pass_length + 1))

        node.children_end = len(self.tree.graph)

    def simulation(self, node_index):

        # node = self.tree.graph[node_index]

        current_player = self.player
        current_board = self.board.copy()

        pass_length = self.tree.graph[node_index].pass_length

        # print("SIMULATION:")
        for depth in range(MAX_DEPTH):

            if pass_length >= 3:
                break

            moves = []
            for coord in coordinates:
                if coord in current_board:
                    continue

                moves.append(Move(coord[0], coord[1], coord[2], False))

            # Can only keep passing from here, so just break anyway
            if len(moves) == 0:
                break

            good_moves = [Move(0, 0, 0, True)]
            for move in moves:
                current_board[move.get_key()] = current_player

                d = get_diameter(current_board, move.get_key(), {}, False)

                if d < 5:
                    good_moves.append(move)

                del current_board[move.get_key()]

            random_move = random.choice(good_moves)
            if random_move.p:
                pass_length += 1
            else:
                pass_length = 0
                current_board[random_move.get_key()] = current_player

            current_player = (current_player + 1) % 3

        return self.get_result(current_board)

    def back_propagation(self, node_index, result):
        current_node_index = node_index
        current_player = self.player

        while True:
            if current_node_index == -1:
                break

            current_node = self.tree.graph[current_node_index]

            # last_move = current_node.last_move

            current_node.visits += 1

            # Take the previous player, since the current node's value is based on the previous player's move
            previous_player = ((current_player - 1) + 3) % 3
            current_node.win_count += result[previous_player]

            current_node_index = current_node.parent

            # rotate backwards in back_propagation
            current_player = previous_player

    def search(self):
        self.sel_depth = 0
        self.start_time = time.time()

        selected_node_index = self.root_node_index

        for iteration in range(MAX_ITERATIONS):
            # print("iteration:", iteration)
            # print(self.player)

            self.descend_to_root(selected_node_index)
            selected_node_index = self.selection()
            selected_node = self.tree.graph[selected_node_index]

            # print("Current Node: ")
            # print("Score:", selected_node.win_count)
            # print("Visits:", selected_node.visits)

            n_children = self.tree.graph[selected_node_index].children_end - \
                         self.tree.graph[selected_node_index].children_start

            if n_children <= 0:
                self.expansion(selected_node_index)

            # New leaf selected node index, make the corresponding move
            if n_children > 0:
                selected_node_index = random.randint(selected_node.children_start, selected_node.children_end - 1)
                selected_node = self.tree.graph[selected_node_index]

                last_move = selected_node.last_move

                if not last_move.p:
                    self.board[last_move.get_key()] = self.player

                self.player = (self.player + 1) % 3

            if selected_node.pass_length >= 3:
                simulation_result = self.get_result(self.board)
            else:
                simulation_result = self.simulation(selected_node_index)
            # print("Simulation Result:", simulation_result)

            # print("Back Propagation")
            self.back_propagation(selected_node_index, simulation_result)

            if iteration % 256 == 0:
                if time.time() - self.start_time >= self.max_time:
                    break

                # print(time.time() - self.start_time)
                # print(iteration, self.sel_depth, len(self.tree.graph))

        self.descend_to_root(selected_node_index)

        # Return the best child of root
        n_root_children = self.tree.graph[self.root_node_index].children_end - \
                          self.tree.graph[self.root_node_index].children_start

        best_index = self.tree.graph[self.root_node_index].children_start
        best_visits = self.tree.graph[best_index].visits

        for it in range(n_root_children):
            root_child_index = self.tree.graph[self.root_node_index].children_start + it
            rc_visits = self.tree.graph[root_child_index].visits

            if rc_visits > best_visits:
                best_index = root_child_index
                best_visits = rc_visits

        return best_index


def mcts2_bot_move(board_copy, player):

    n = len(board_copy)
    allocated_time = max(-1/700 * (n * n) + 2.6, 0.2)

    mcts_engine = MCTS(board_copy, player, allocated_time)

    print_board(board_copy, None)
    print(score(board_copy))
    print(mcts_engine.get_result(board_copy))

    best_index = mcts_engine.search()

    best_node = mcts_engine.tree.graph[best_index]

    move = best_node.last_move

    print("SEL_DEPTH:", mcts_engine.sel_depth)
    print(best_node.win_count, best_node.visits, best_node.win_count / best_node.visits)

    if move.p:
        return None

    return move.get_key()
