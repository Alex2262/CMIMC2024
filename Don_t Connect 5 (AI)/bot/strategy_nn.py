import random
import time

import math

GAME_TIME = 30

init_time = time.time()

GRID_RADIUS = 4
coordinates = []


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
        if coordinates[i] in board:
            player = player_map[board[coordinates[i]]]
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
            if 3 in three:
                return 4  # this is a shape x - x - x - x
                #                     x   x
        return 5  # diameter is 5 otherwise
    # For the larger(>6) ones, diameter must be larger than 5 so we just return 5
    return 5


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

        self.iterations = 0
        self.sel_depth = 0

        self.root_node_index = 0
        self.max_time = 1

    def get_evaluation(self, pass_length):

        if pass_length >= 3 or len(self.board) == 96:
            return get_wdl(score(self.board))

        perspective_evaluation = value_net.feedforward(get_inputs(self.board, self.player))

        # print(self.player, perspective_evaluation)
        # print_board(self.board, None)
        # time.sleep(1)
        if self.player == 0:
            return perspective_evaluation

        if self.player == 1:
            return [perspective_evaluation[2], perspective_evaluation[0], perspective_evaluation[1]]

        return [perspective_evaluation[1], perspective_evaluation[2], perspective_evaluation[0]]


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

                coef = 0.8
                bonus = -0.3

                if not move.p:
                    self.board[move.get_key()] = self.player
                    d = get_diameter(self.board, move.get_key(), {}, False)
                    del self.board[move.get_key()]

                    coef = 1

                    bonus = 0.05 * empty_count

                    if d >= 5:
                        coef = 0.5
                        bonus = -10

                    elif adjacent_friend is not None:

                        old_diameter = get_diameter(self.board, adjacent_friend, {}, False)
                        if old_diameter <= 3:
                            if d > old_diameter:
                                bonus += 0.1
                                if d == 4 or empty_count >= 1:
                                    bonus += 0.2

                            elif d <= old_diameter:
                                bonus -= 0.1
                        else:
                            bonus -= min(0.2 + 0.01 * empty_count - 0.03 * len(enemies), 0)

                    else:
                        bonus += 0.02 * len(enemies)

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

            if (n_children <= 0 and selected_node.visits >= 2) or selected_node_index == self.root_node_index:
                self.expansion(selected_node_index)

            # New leaf selected node index, make the corresponding move
            if n_children > 0:
                selected_node_index = random.randint(selected_node.children_start, selected_node.children_end - 1)
                selected_node = tree.graph[selected_node_index]

                last_move = selected_node.last_move

                if not last_move.p:
                    self.board[last_move.get_key()] = self.player

                self.player = (self.player + 1) % 3

            evaluation = self.get_evaluation(selected_node.pass_length)

            # print("Back Propagation")
            self.back_propagation(selected_node_index, evaluation)

            if self.iterations >= 30 and time.time() - start_time >= self.max_time:
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

    '''
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
    '''


TABLE = {1: 0, 2: 0, 3: 1, 4: 3, 5: 0}

EXPLORATION_CONSTANT = 0.3
MAX_ITERATIONS = 10000

set_node_coordinates()
NEIGHBOR_LIST = dict(zip(coordinates, [SELECT_VALID(ALL_NEIGHBOR(*node)) for node in coordinates]))

tree = Tree()
tree.graph.append(Node(-1, Move(0, 0, 0, True), 0))  # Root node

real_board = {}

mcts_engine = MCTS()

current_time = GAME_TIME - 6
move_count = 0

predicted_total_moves = 45
start_time = 0


DEFAULT_LR = 0.01
INPUT_SIZE = 96 * 3
HL_SIZE = 8
OUTPUT_SIZE = 3
value_net = NeuralNetwork(input_size=INPUT_SIZE, hl_size=HL_SIZE, output_size=OUTPUT_SIZE)

weights_string = "235176041247101049061169192137049169218090218191015078143127180121039003141083229094001033237082022165149209146104223073225108167151067114189224251032212040025157215232213122235184104074139174103253001031001126054128133121125091005118011242201132040138062011174170190075135180212135119076052183180044131114208191245215113015152198250207069105076055233015240252245214155202231071210043078165101144075190185145008150029150235037210022242199042190221250034208172050184174174036129125165009218130076021019128163157151251093084201239075205173009238216122117041014092044088159213005122008188092143016155132227131071148120049022107011017178087189211252125123146126193067096221075010133095152247246022129015252027198052205043070098118240066010070002207007191103108021118029162106144246218044059228109012148102069085016067125013052235021179125133155063154014163091151229058219082060020031069131163135150125025226017129047182004084140129052196088201034020029202102248253142041097020100054247151023051094179156028013010007179229144170056232234004191065183026034193130237189091154121171233164141020160051006019020238187118123097077012242142079198103157030126217071194201042213230059190148021115119199057049048159219034088102149206137214224086019007060171107224179049048023146217161186156146117128001037111033067067116162204167001223029071173059040072217030077004052251167249145101020078098067050095083185162250043049241096212159088218027005077143057132163205020032247092031106208113007158138036009222039175121084053035051168180174008170161012115132108038137163067180238154216188198210137181163080047208220252233202178055010178082026137190254211016185004178023060019165076197080165201143009112150224216088198198234254079108069070188102060174111045080087253141226103145094091041034195221002059206221216227033082113181017214219098051054110124064078121145068027196121146072062206098218176034240134191151198080051177090011091065245002189230057178151043210253044051010201220239145245249114142202143199216145247150234170119195013189188140213176067004150239019020095080075114013148154217238049055088179213129203230099097031076108090064009243134143215181050203148173035221045220152032126166254169154081132099063186081126236054018041197039013126244077126098113118091030184061080211131047026196090063150225252022061005069066057085238094017248030243176031253099229030009240159092165102081201109171055227189098176069251088080203025174018241105179071156209229178192058076118094057136041053124004165223068092150144039252178091027158240190148235030121051207218209112070023210207212236005212173051237090094025038120046054051190159006002002060017232196168199047093046125004070248038083030135098118205015078209149225110230011161101244204242130046202248187126141114234031003185215015017040113095137076125115015248178190182023191090105026251059082216161098106115136035030209041184041098115108046189124052123234094158082095065056073117102128218020029191176199228073191058118123108020178245086253169123060046201124216036024025045106155118164121101154207108111214198188066067075249236068219143214032230203229031102071140073236146103182237183022026191133008052003220011057033091078065166223227034228085254091238093035161051017220069247016191154234201115067183029096070066201036036150005092112196220068163172232072113117061092211093033061220155223221153167167123082219227235010250033228138208242019228031196132022241177070225175023028179144216110137018169067104043144206248115004105165041003028158211155156088201161202088215186210185029225044035143163041035047013206065002157196055090151231212014074006134019085240225114234082132011238018200056136127185064195077148194125168127070134069076104156042081233063077241128187241180022103058230210250232033212063240197020102131019014149139234173063136229133082158033251065056033212030144136176166075215121134212077128194199210226075250141212095120029248017167189108106127144152118127187045020056146019173062227106102078218099067213223161101176008084136223062115191243049194120105242116045087103165091162080098087154221227150048068023158163060222044232090040164033086110240213114133171181218155147078134084045212143048032248092078169246149082011082059245073121246126251170248122131122024053036020185033093216145136026182135119043196031061231005185245207234213105186165150181186070128019063093019246147219072142150069069212109042075091197159227043098110195222248152076108227143064206090165046125231244145094110145036084028098064042118024162007104139089149171241229111246190119220109024007028127022220198044064004049125123077077066123132025031050057204123165089073032079213209235055139157151133052230057219078115145018122072120167082251070145028112245014021126173026200012089207099049102114064022155185253137189243061210136105100245152023012250219159043097096228013179145141141049246112082226147105073179209230081233203107077048186085049084066075077151062052073046083182096235027125194066179030138212198077246094231032172216095194092246170085138095082029247220198201160117032142184009005055192079202083208039047184163052090245094191157157202014243065044154052078050027120142048059189037049225009167106074108199100131014180048240172168065044131063176108091119149189151022059202204180217012219173249250198142005012046124128051016090024242125090124248014076219032079216143140044148178098138124214243081123176224244038212235173226178129030073015063178069049233046010141134159014206119238122030071130150093088083131167095087192241220168100186206138051104089031200083039158209192197240248055022233220131011025248192010124238013085158187165121138187172107016085239231248014078248111187067122019189073061130078159227102113122114163107240211051005175164188002162068116164241052132223107006072241060047042246207029010030070110238086092062082129129088167026123114162194139147229139203015038233218032181028169219012094220212152076154088021020078042092116041036150145120021088228111006161009095214103199005079235147235231164073021032227150077165141218122114091094081218130203040133061055252089057066072185174185141050056152100208031101084078180053125105073103209221221049114240099198096001204001244064133151150035062213188163115041044075044187008093158171216157194226063162212159094238140138223120236224089005215112004218193169133191063035162151223075048236016223192136062046118182217003167019031115234188171160070168177244119241186003194002029184154093225047242219091205221150097110035222124204074081109062213188052005111195149043070212108193145013037175069234099204113035106036168069108216106080149238096020016064158215144014231130225121001237191081168151201084108206143158055196069017173044054240001194182230072081187185134152176001163075158098163081067197080097035062100099248144001096144181020156142177016070187235135136227119134124161153241202074064097104161128105010176096095074016173250029130185094111236058039003196171059137039024081093127159075242147054008077027009008084070006160224129124003144058054200022232099052105110112160105077041220050107055053015187095215208026021148030203194135176045017045105212114132216074"
value_net.init_weights(weights_string)


def strategy_nn(board_copy, player):
    global current_time, move_count, predicted_total_moves, start_time

    start_time = time.time()

    tree.graph = []
    tree.graph.append(Node(-1, Move(0, 0, 0, True), 0))  # Root node

    mcts_engine.root_node_index = 0

    mcts_engine.board = board_copy
    mcts_engine.player = player

    allocated_time = max(current_time / max(12, (predicted_total_moves - move_count)) - 0.05, 0.1)
    # print(allocated_time, current_time)

    mcts_engine.max_time = allocated_time

    # print_board(board_copy, None)
    # print(score(board_copy))
    # print(mcts_engine.get_evaluation(0))

    # print("IPS: ", mcts_engine.iterations / allocated_time, mcts_engine.iterations, allocated_time)

    best_index = mcts_engine.search()

    # print("ITERATIONS: ", mcts_engine.iterations)

    best_node = tree.graph[best_index]
    move = best_node.last_move

    # mcts_engine.root_node_index = best_index

    '''
    print(mcts_engine.root_node_index, tree.graph[mcts_engine.root_node_index].children_start, 
          tree.graph[mcts_engine.root_node_index].children_end)
    '''
    # print("SEL_DEPTH:", mcts_engine.sel_depth)
    # print(best_node.win_count, best_node.visits, best_node.win_count / best_node.visits)
    # print(move.get_key(), move.p)

    move_count += 1

    current_time -= (time.time() - start_time)

    if move.p:
        # real_board[move.get_key] = player
        return None

    return move.get_key()
