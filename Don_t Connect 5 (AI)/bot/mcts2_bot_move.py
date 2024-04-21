import random
import time

import math
from collections import deque

GRID_RADIUS = 4
coordinates = []


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

    def copy(self):
        new_node = Node(self.parent, self.last_move, self.pass_length)
        new_node.children_start = self.children_start
        new_node.children_end = self.children_end
        new_node.win_count = self.win_count
        new_node.visits = self.visits

        return new_node


class Tree:
    def __init__(self):
        self.graph = []


def ALL_NEIGHBOR(x1, y1, z1):
    return (x1 + 1, y1, z1), (x1 - 1, y1, z1), (x1, y1 + 1, z1), (x1, y1 - 1, z1), (x1, y1, z1 + 1), (x1, y1, z1 - 1)


def SELECT_VALID(lis):
    return [(x1, y1, z1) for (x1, y1, z1) in lis
            if 1 <= x1 + y1 + z1 <= 2
            and -GRID_RADIUS + 1 <= x1 <= GRID_RADIUS
            and -GRID_RADIUS + 1 <= y1 <= GRID_RADIUS
            and -GRID_RADIUS + 1 <= z1 <= GRID_RADIUS]


def set_node_coordinates():
    global coordinates

    for x in range(-GRID_RADIUS + 1, GRID_RADIUS + 1):
        for y in range(-GRID_RADIUS + 1, GRID_RADIUS + 1):
            for z in range(-GRID_RADIUS + 1, GRID_RADIUS + 1):
                # Check if node is valid
                if 1 <= x + y + z <= 2:
                    coordinates.append((x, y, z))


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

    mat = [[-2 for _ in range(23)] for _ in range(43)]

    for coord in coordinates:
        xc, yc = hex_to_pixel(*coord)

        # print(coord, yc, xc)
        mat[round(yc) + 21][round(xc) + 11] = -1

    for coord in board:
        xc, yc = hex_to_pixel(*coord)
        mat[round(yc) + 21][round(xc) + 11] = board[coord]

    if move is not None:
        xc, yc = hex_to_pixel(*move.get_key())
        mat[round(yc) + 21][round(xc) + 11] = 3

    for row in range(43):
        s = ""
        for col in range(23):
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

    def __init__(self):

        self.board = {}
        self.player = 0

        self.start_time = 0
        self.iterations = 0
        self.sel_depth = 0

        self.root_node_index = 0
        self.max_time = 1

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

        return result

    def descend_to_root(self, node_index):
        while node_index != self.root_node_index:
            last_move = tree.graph[node_index].last_move

            if not last_move.p:
                del self.board[last_move.get_key()]

            node_index = tree.graph[node_index].parent
            self.player = ((self.player - 1) + 3) % 3

    def selection(self):

        leaf_node_index = self.root_node_index

        depth = 0

        while True:

            leaf_node = tree.graph[leaf_node_index]

            n_children = leaf_node.children_end - leaf_node.children_start
            if n_children <= 0:
                break

            best_uct = -100000.0

            # SELECTING BEST CHILD
            for i in range(n_children):
                child_node_index = leaf_node.children_start + i
                child_node = tree.graph[child_node_index]

                # SILLY BONUSES / COEFFICIENTS FOR MOVE TYPES
                move = child_node.last_move

                neighbors = SELECT_VALID(ALL_NEIGHBOR(*move.get_key()))
                empty_count = 0

                adjacent_friend = None
                enemies = []

                for neighbor in neighbors:
                    if neighbor not in self.board:
                        empty_count += 1
                        continue

                    if self.board[neighbor] == self.player:
                        adjacent_friend = neighbor

                    else:
                        enemies.append(neighbor)

                coef = 0.6
                bonus = 0

                if not move.p:
                    self.board[move.get_key()] = self.player
                    d = get_diameter(self.board, move.get_key(), {}, False)
                    del self.board[move.get_key()]

                    coef = 1

                    bonus = 0.025 * empty_count

                    if d >= 5:
                        coef = 0.01
                        bonus = -1

                    elif adjacent_friend is not None:
                        if get_diameter(self.board, adjacent_friend, {}, False) <= 3:
                            bonus += 0.22
                        else:
                            bonus -= 0.05

                    if d <= 2:
                        bonus += 0.01 * len(enemies)

                # UCT ALGORITHM
                exploitation_value = child_node.win_count / child_node.visits
                exploration_value = math.sqrt(math.log(leaf_node.visits) / child_node.visits)

                uct_value = coef * (exploitation_value + EXPLORATION_CONSTANT * exploration_value) + bonus

                if uct_value > best_uct:
                    leaf_node_index = child_node_index
                    best_uct = uct_value

            last_move = tree.graph[leaf_node_index].last_move

            if not last_move.p:
                self.board[last_move.get_key()] = self.player

            self.player = (self.player + 1) % 3
            depth += 1

        self.sel_depth = max(self.sel_depth, depth)
        return leaf_node_index

    def expansion(self, node_index):

        node = tree.graph[node_index]
        node.children_start = len(tree.graph)

        for coord in coordinates:
            if coord in self.board:
                continue

            tree.graph.append(Node(node_index, Move(coord[0], coord[1], coord[2], False), 0))

        # Pass move
        if node.pass_length < 3:
            tree.graph.append(Node(node_index, Move(0, 0, 0, True), node.pass_length + 1))

        node.children_end = len(tree.graph)

    def simulation(self, node_index):

        # node = self.tree.graph[node_index]

        current_player = self.player
        current_board = self.board.copy()

        pass_length = tree.graph[node_index].pass_length

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
                del current_board[move.get_key()]

                if d < 5:
                    good_moves.append(move)

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

            current_node = tree.graph[current_node_index]

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

        self.iterations = 0
        for self.iterations in range(MAX_ITERATIONS):
            # print("iteration:", iteration)
            # print(self.player)

            self.descend_to_root(selected_node_index)
            selected_node_index = self.selection()
            selected_node = tree.graph[selected_node_index]

            # print("Current Node: ")
            # print("Score:", selected_node.win_count)
            # print("Visits:", selected_node.visits)

            n_children = selected_node.children_end - \
                         selected_node.children_start

            if n_children <= 0 and selected_node.visits >= 2:
                self.expansion(selected_node_index)

            # New leaf selected node index, make the corresponding move
            if n_children > 0:
                selected_node_index = random.randint(selected_node.children_start, selected_node.children_end - 1)
                selected_node = tree.graph[selected_node_index]

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

            if self.iterations % 20 == 0:
                if time.time() - self.start_time >= self.max_time:
                    break

                # print(time.time() - self.start_time)
                # print(self.iterations, self.sel_depth, len(self.tree.graph))

        # self.descend_to_root(selected_node_index)

        # Return the best child of root
        n_root_children = tree.graph[self.root_node_index].children_end - \
                          tree.graph[self.root_node_index].children_start

        best_index = tree.graph[self.root_node_index].children_start
        best_visits = tree.graph[best_index].visits

        for it in range(n_root_children):
            root_child_index = tree.graph[self.root_node_index].children_start + it
            rc_visits = tree.graph[root_child_index].visits

            if rc_visits > best_visits:
                best_index = root_child_index
                best_visits = rc_visits

        return best_index

    def flatten_tree(self):

        copy_graph = [x.copy() for x in tree.graph]

        start_size = len(tree.graph)
        tree.graph = []

        next_nodes_index = deque()
        next_nodes_index.append((self.root_node_index, 0))

        tree.graph.append(copy_graph[self.root_node_index].copy())

        self.root_node_index = 0

        tree.graph[0].parent = -1

        while len(next_nodes_index) != 0:
            current_node_index = next_nodes_index.popleft()
            old_node_index = current_node_index[0]
            new_node_index = current_node_index[1]

            current_old_node = copy_graph[old_node_index]
            current_new_node = tree.graph[new_node_index]

            current_new_node.children_start = len(tree.graph)

            for i in range(current_old_node.children_end - current_old_node.children_start):
                tree.graph.append(copy_graph[current_old_node.children_start + i].copy())
                tree.graph[-1].parent = new_node_index

                next_nodes_index.append((current_old_node.children_start + i, len(tree.graph) - 1))

            current_new_node.children_end = len(tree.graph)

        # print("Tree flattened from", start_size, "to", len(tree.graph))

    def update_tree(self, board, player):
        real_keys = set(real_board.keys())
        given_keys = set(board.keys())

        new_keys = given_keys - real_keys

        for i in range(1, 3):
            check_player = (player + i) % 3

            anything = False
            for key in new_keys:
                if board[key] != check_player:
                    continue

                anything = True

                real_board[key] = check_player

                root_node = tree.graph[self.root_node_index]
                found = False

                for j in range(root_node.children_end - root_node.children_start):
                    current_child_index = tree.graph[self.root_node_index].children_start + j

                    last_move = tree.graph[current_child_index].last_move

                    if last_move.x == key[0] and last_move.y == key[1] and last_move.z == key[2]:
                        found = True
                        self.root_node_index = current_child_index
                        break

                if found:
                    break

                tree.graph.append(Node(-1, Move(0, 0, 0, True), 0))
                self.root_node_index = len(tree.graph) - 1

            if anything:
                continue

            # PASS MOVE
            found = False
            root_node = tree.graph[self.root_node_index]
            for j in range(root_node.children_end - root_node.children_start):
                current_child_index = tree.graph[self.root_node_index].children_start + j

                last_move = tree.graph[current_child_index].last_move

                if last_move.p:
                    found = True
                    self.root_node_index = current_child_index
                    break

            if found:
                continue

            tree.graph.append(Node(-1, Move(0, 0, 0, True), 0))
            self.root_node_index = len(tree.graph) - 1


TABLE = {1: 0, 2: 0, 3: 1, 4: 3, 5: 0}

EXPLORATION_CONSTANT = 0.5
MAX_DEPTH = 60
MAX_ITERATIONS = 100_000

set_node_coordinates()
NEIGHBOR_LIST = dict(zip(coordinates, [SELECT_VALID(ALL_NEIGHBOR(*node)) for node in coordinates]))


tree = Tree()
tree.graph.append(Node(-1, Move(0, 0, 0, True), 0))  # Root node

real_board = {}

mcts_engine = MCTS()


def mcts2_bot_move(board_copy, player):

    mcts_engine.board = board_copy
    mcts_engine.player = player

    n = len(board_copy)

    allocated_time = max(-1/2500 * (n * n) + 1.3, 0.3)
    mcts_engine.max_time = allocated_time

    mcts_engine.update_tree(board_copy, player)
    mcts_engine.flatten_tree()

    '''
    print_board(board_copy, None)
    print(score(board_copy))
    print(mcts_engine.get_result(board_copy))

    print("IPS: ", mcts_engine.iterations / allocated_time, mcts_engine.iterations, allocated_time)
    '''

    best_index = mcts_engine.search()

    best_node = tree.graph[best_index]

    move = best_node.last_move
    mcts_engine.root_node_index = best_index

    '''
    print(mcts_engine.root_node_index, tree.graph[mcts_engine.root_node_index].children_start, 
          tree.graph[mcts_engine.root_node_index].children_end)
    '''
    # print("SEL_DEPTH:", mcts_engine.sel_depth)

    # print(best_node.win_count, best_node.visits, best_node.win_count / best_node.visits)

    if move.p:
        real_board[move.get_key] = player
        return None

    return move.get_key()
