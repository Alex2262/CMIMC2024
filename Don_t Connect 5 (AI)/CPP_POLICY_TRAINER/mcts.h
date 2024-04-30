//
// Created by Alexander Tian on 4/26/24.
//

#ifndef DC5_MCTS_H
#define DC5_MCTS_H

#include "util.h"

const int MAX_TIME = 10;
const int MAX_DEPTH = 25;
const double EXPLORATION_CONSTANT = 0.3;
const int MAX_ITERATIONS = 1000;


struct Node {
    uint64_t parent;
    int children_start = 0;
    int children_end = 0;

    double win_count = 0;
    int visits = 0;

    Move last_move;
    int pass_length;

    Node(uint64_t c_parent, Move c_last_move, int pl) {
        parent = c_parent;
        last_move = c_last_move;
        pass_length = pl;
    }
};

struct Tree {
    std::vector<Node> graph;
};

class MCTS {
public:
    Tree tree = Tree();

    std::map<Coord, int> board;
    int player = 0;
    int iterations = 0;
    int sel_depth = 0;

    int root_node_index = 0;

    uint64_t start_time = 0;
    // uint64_t max_time = 1000;

    std::array<double, 3> get_result();

    void descend_to_root(uint64_t node_index);

    uint64_t selection();
    void expansion(uint64_t node_index);

    std::array<double, 3> simulation(uint64_t node_index);
    void back_propagation(uint64_t node_index, std::array<double, 3>& result);

    uint64_t search();
};

#endif //DC5_MCTS_H
