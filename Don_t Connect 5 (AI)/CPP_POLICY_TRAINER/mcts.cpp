//
// Created by Alexander Tian on 4/26/24.
//

#include <iostream>
#include "mcts.h"

std::array<double, 3> MCTS::get_result() {
    auto scores = score(board);
    std::array<int, 3> score_copy = scores;

    std::array<double, 3> result{};

    std::sort(std::begin(score_copy), std::end(score_copy));

    int small = score_copy[0];
    int large = score_copy[2];

    int count_large = 0;
    int count_small = 0;

    for (int i = 0; i < 3; i++) {
        if (scores[i] == large) count_large++;
        if (scores[i] == small) count_small++;
    }

    double large_score = 1.0 / count_large;
    double small_score = count_small == 2 ? 0.15 : 0;

    for (int i = 0; i < 3; i++) {
        if (scores[i] == large) result[i] = large_score;
        else if (scores[i] == small) result[i] = small_score;
        else result[i] = 0.3;
    }

    return result;
}

void MCTS::descend_to_root(uint64_t node_index) {
    while (node_index != root_node_index) {
        Move last_move = tree.graph[node_index].last_move;

        if (!last_move.pass) board.erase(last_move.coord);

        node_index = tree.graph[node_index].parent;
        player = ((player - 1) + 3) % 3;
    }
}

uint64_t MCTS::selection() {
    uint64_t leaf_node_index = root_node_index;
    int depth = 0;
    std::set<Coord> visited;

    while (true) {
        Node& leaf_node = tree.graph[leaf_node_index];

        int n_children = leaf_node.children_end - leaf_node.children_start;

        if (n_children <= 0) break;

        double best_uct = -100'000.0;

        for (int i = 0; i < n_children; i++) {
            uint64_t child_node_index = leaf_node.children_start + i;
            Node& child_node = tree.graph[child_node_index];

            Move move = child_node.last_move;

            int empty_count = 0;
            int enemy_count = 0;

            Coord adjacent_friend;
            bool has_af = false;

            for (auto neighbor : all_valid_neighbors[move.coord]) {
                if (board.find(neighbor) == board.end()) {
                    empty_count++;
                    continue;
                }

                if (board[neighbor] == player) {
                    has_af = true;
                    adjacent_friend = neighbor;
                }

                else enemy_count++;
            }

            double coef = 0.8;
            double bonus = -0.3;

            if (!move.pass) {
                visited.clear();
                board[move.coord] = player;
                int d = get_diameter(board, move.coord, visited);
                board.erase(move.coord);

                coef = 1.0;
                bonus = 0.05 * empty_count;

                if (d >= 5) {
                    coef = 0.5;
                    bonus = -10;
                }

                else if (has_af) {
                    visited.clear();
                    int old_diameter = get_diameter(board, adjacent_friend, visited);

                    if (old_diameter <= 3) {
                        if (d > old_diameter) {
                            bonus += 0.1;
                            if (d == 4 || empty_count >= 1) bonus += 0.2;
                        }
                    } else bonus -= std::min(0.2 + 0.01 * empty_count - 0.03 * enemy_count, 0.0);
                }

                else bonus += 0.02 * enemy_count;
            }

            double exploitation_value = child_node.win_count / (child_node.visits + 1.0);
            double exploration_value  = std::sqrt(std::log(leaf_node.visits) / (child_node.visits + 1.0));

            double uct_value = coef * (exploitation_value + EXPLORATION_CONSTANT * exploration_value) + bonus;
            if (uct_value > best_uct) {
                leaf_node_index = child_node_index;
                best_uct = uct_value;
            }
        }

        Move last_move = tree.graph[leaf_node_index].last_move;
        if (!last_move.pass) board[last_move.coord] = player;

        player = (player + 1) % 3;

        depth++;
    }

    sel_depth = std::max(sel_depth, depth);
    return leaf_node_index;
}

void MCTS::expansion(uint64_t node_index) {
    Node& node = tree.graph[node_index];
    node.children_start = tree.graph.size();

    for (Coord coord : all_coordinates) {
        if (board.find(coord) != board.end()) continue;

        tree.graph.push_back(Node(node_index, {coord, false}, 0));
    }

    if (node.pass_length < 3) tree.graph.push_back(Node(node_index, pass_move, node.pass_length + 1));

    node.children_end = tree.graph.size();
}

std::array<double, 3> MCTS::simulation(uint64_t node_index) {

    int pass_length = tree.graph[node_index].pass_length;

    std::vector<Move> moves_made;

    std::vector<Move> moves;

    for (int depth = 0; depth < MAX_DEPTH; depth++) {

        if (pass_length >= 3) break;
        if (board.size() >= 96) break;

        moves.clear();

        for (Coord coord : all_coordinates) {
            if (board.find(coord) != board.end()) continue;
            moves.push_back({coord, false});
        }

        std::vector<bool> works(moves.size(), true);
        bool possible = true;

        int attempts = 0;

        std::set<Coord> visited;

        int max_attempts = std::min<int>(attempts, moves.size());

        while (true) {
            int r_int = rand() % moves.size();
            if (!works[r_int]) {
                attempts++;
                if (attempts >= max_attempts) {
                    possible = false;
                    break;
                }
                continue;
            }

            visited.clear();

            Move random_move = moves[r_int];
            board[random_move.coord] = player;
            int d = get_diameter(board, random_move.coord, visited);

            if (d >= 5) {
                board.erase(random_move.coord);
                works[r_int] = false;
                attempts++;

                if (attempts < max_attempts) continue;
                possible = false;
            } else {
                pass_length = 0;
                moves_made.push_back(random_move);
            }

            break;
        }

        if (!possible) {
            pass_length++;
            moves_made.push_back(pass_move);
        }

        player = (player + 1) % 3;
    }

    for (int i = moves_made.size() - 1; i >= 0; i--) {
        if (!moves_made[i].pass) {
            board.erase(moves_made[i].coord);
        }

        player = ((player - 1) + 3) % 3;
    }

    auto result = get_result();
    return result;

}

void MCTS::back_propagation(uint64_t node_index, std::array<double, 3>& result) {
    uint64_t current_node_index = node_index;
    int current_player = player;

    while (true) {
        if (current_node_index == -1) break;

        Node& current_node = tree.graph[current_node_index];

        current_node.visits++;

        int previous_player = ((current_player - 1) + 3) % 3;
        current_node.win_count += result[previous_player];

        current_node_index = current_node.parent;
        current_player = previous_player;
    }
}

uint64_t MCTS::search() {

    for (int i = 0; i < 96; i++) {
        std::cout << i << " " << all_coordinates[i].x << " " << all_coordinates[i].y << " " << all_coordinates[i].z << std::endl;
    }


    tree.graph.clear();
    tree.graph.emplace_back(-1, pass_move, 0);

    root_node_index = 0;
    sel_depth = 0;
    uint64_t selected_node_index = root_node_index;

    // print_board(board, pass_move);

    /*
    auto time = std::chrono::high_resolution_clock::now();
    start_time = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::time_point_cast<std::chrono::milliseconds>(time).time_since_epoch()).count();
    */
    // std::cout << "PLAYER: " << player << std::endl;

    for (iterations = 0; iterations < MAX_ITERATIONS; iterations++) {

        // std::cout << "ITERATION: " << iterations << std::endl;
        descend_to_root(selected_node_index);
        selected_node_index = selection();
        Node& selected_node = tree.graph[selected_node_index];

        int n_children = selected_node.children_end - selected_node.children_start;

        if (n_children <= 0 && selected_node.visits > 0) expansion(selected_node_index);

        if (n_children > 0) {
            selected_node_index = selected_node.children_start + rand() % n_children;
            selected_node = tree.graph[selected_node_index];

            Move last_move = selected_node.last_move;

            if (!last_move.pass) board[last_move.coord] = player;
            player = (player + 1) % 3;
        }

        auto simulation_result = simulation(selected_node_index);

        back_propagation(selected_node_index, simulation_result);

        if (iterations >= MAX_ITERATIONS) break;
    }

    /*
    time = std::chrono::high_resolution_clock::now();
    uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::time_point_cast<std::chrono::milliseconds>(time).time_since_epoch()).count();

    std::cout << "TIME TAKEN: " << current_time - start_time << std::endl;

    std::cout << "IPS: " << static_cast<int>(static_cast<double>(iterations) /
                                            (static_cast<double>(current_time - start_time) / 1000.0)) << std::endl;
    */

    descend_to_root(selected_node_index);

    int n_root_children = tree.graph[root_node_index].children_end - tree.graph[root_node_index].children_start;

    int best_index = tree.graph[root_node_index].children_start;
    int best_visits = tree.graph[best_index].visits;

    for (int i = 0; i < n_root_children; i++) {
        uint64_t root_child_index = tree.graph[root_node_index].children_start + i;
        int rc_visits = tree.graph[root_child_index].visits;

        if (rc_visits > best_visits) {
            best_index = root_child_index;
            best_visits = rc_visits;
        }
    }

    // std::cout << ":C" << std::endl;
    // print_board(board, pass_move);

    return best_index;
}