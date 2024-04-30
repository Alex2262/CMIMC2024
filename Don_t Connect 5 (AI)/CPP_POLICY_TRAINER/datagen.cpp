//
// Created by Alexander Tian on 4/26/24.
//

#include <thread>
#include <iostream>
#include <future>
#include "datagen.h"
#include "mcts.h"


void gen_data_thread(CombinedData& combined_data, int thread_id) {
    MCTS mcts = MCTS();

    for (int game = 0; game < GAME_PER_THREAD; game++) {

        if (combined_data.stopped) break;

        std::cout << "GAME #" << game + 1 << std::endl;

        DataGame data_game = DataGame();

        mcts.board.clear();
        mcts.player = 0;

        for (int i = 0; i < RANDOM_MOVES; i++) {
            std::vector<Move> moves;
            for (Coord coord : all_coordinates) {

                if (data_game.board.find(coord) == data_game.board.end()) {
                    moves.push_back({coord, false});
                }
            }

            Move random_move = moves[rand() % moves.size()];
            mcts.board[random_move.coord] = mcts.player;
            mcts.player = (mcts.player + 1) % 3;
            data_game.make_move(random_move);
        }

        // print_board(mcts.board, pass_move);
        // print_board(data_game.board, pass_move);

        while (!data_game.is_game_over()) {
            // std::cout << "thread " << thread_id << " is searching" << std::endl;
            uint64_t best_index = mcts.search();
            // std::cout << "thread " << thread_id << " is done searching" << std::endl;
            Move best_move = mcts.tree.graph[best_index].last_move;

            PolicyData policy_data{};

            for (int i = 0; i < 96; i++) {
                Coord coord = all_coordinates[i];
                if (data_game.board.find(coord) == data_game.board.end()) continue;

                int c_player = data_game.board[coord];
                int p_player = PLAYER_TO_PERSPECTIVE[data_game.player][c_player];

                int index = p_player * 96 + i;
                policy_data.set_bit(index);
            }

            // std::cout << "PLAYER: " << data_game.player << std::endl;
            // auto pb = policy_data.get_board();
            // print_board(data_game.board, pass_move);
            // print_board(pb, pass_move);


            std::map<Coord, int> move_visits;

            for (int child_node_index = mcts.tree.graph[mcts.root_node_index].children_start;
                     child_node_index < mcts.tree.graph[mcts.root_node_index].children_end; child_node_index++) {
                Move last_move = mcts.tree.graph[child_node_index].last_move;

                if (last_move.pass) policy_data.visits[96] = mcts.tree.graph[child_node_index].visits;
                else move_visits[last_move.coord] = mcts.tree.graph[child_node_index].visits;
            }

            for (int i = 0; i < 96; i++) {
                Coord coord = all_coordinates[i];
                policy_data.visits[i] = move_visits[coord];
            }

            combined_data.thread_data[thread_id].push_back(policy_data);

            if (!best_move.pass) mcts.board[best_move.coord] = mcts.player;
            mcts.player = (mcts.player + 1) % 3;

            data_game.make_move(best_move);
        }

        combined_data.num_games[thread_id]++;

        std::cout << "GAME COMPLETED" << std::endl;
    }
}


void input_thread(CombinedData& combined_data) {


    while (!combined_data.stopped) {

        std::string msg;

        std::chrono::seconds timeout(1);

        // std::cout << "WHAT!" << std::endl;

        if (msg == "stop") {
            combined_data.stopped = true;
            break;
        }

        if (msg == "show") {
            int num_data = 0;
            for (int i = 0; i < THREADS; i++) num_data += combined_data.thread_data[i].size();

            int num_games = 0;
            for (int i = 0; i < THREADS; i++) num_data += combined_data.num_games[i];

            std::cout << "NUM DATA: " << num_data << " NUM GAMES: " << num_games << std::endl;
        }
    }
}

void gen_data(CombinedData& combined_data) {
    std::vector<std::thread> threads;

    threads.reserve(THREADS + 1);

    for (int thread_id = 0; thread_id < THREADS; thread_id++) {
        std::cout << "THREAD " << thread_id << " started!" << std::endl;
        threads.emplace_back(gen_data_thread, std::ref(combined_data), thread_id);
    }

    // threads.emplace_back(input_thread, std::ref(combined_data));

    for (int thread_id = 0; thread_id < THREADS; thread_id++) {
        threads[thread_id].join();
    }

    combined_data.stopped = true;

    // threads[THREADS].join();

    std::cout << "All threads joined" << std::endl;
}