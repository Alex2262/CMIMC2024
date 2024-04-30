//
// Created by Alexander Tian on 4/26/24.
//

#ifndef DC5_DATAGEN_H
#define DC5_DATAGEN_H

#include <cstdint>
#include <vector>
#include <array>

#include "mcts.h"

constexpr int THREADS = 1;

constexpr int GAME_PER_THREAD = 1;

constexpr int RANDOM_MOVES = 4;

constexpr int PLAYER_TO_PERSPECTIVE[3][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}};


class DataGame {
public:
    int player = 0;
    int pass_length = 0;

    std::map<Coord, int> board;
    // uint64_t bitboards[5]{};

    inline bool is_game_over() {
        if (pass_length >= 3) return true;

        for (Coord coord : all_coordinates) {
            if (board.find(coord) == board.end()) return false;
        }

        return true;
    }

    inline void make_move(Move move) {

        if (!move.pass) {
            pass_length = 0;

            /*
            for (int i = 0; i < 96; i++) {
                Coord coord = all_coordinates[i];
                if (move.coord.x == coord.x && move.coord.y == coord.y && move.coord.z == coord.z) {
                    set_bit(player * 96 + i);
                    break;
                }
            }
             */

            // std::cout << "SET MOVE!!!!!" << std::endl;
            board[move.coord] = player;
        } else pass_length++;

        player = (player + 1) % 3;
    }
};

struct PolicyData {
    uint64_t bitboards[5];
    std::array<int, 97> visits;

    inline void set_bit(int index) {
        if (index < 64) bitboards[0] |= (1ULL << index);
        else if (index < 128) bitboards[1] |= (1ULL << (index - 64));
        else if (index < 192) bitboards[2] |= (1ULL << (index - 128));
        else if (index < 256) bitboards[3] |= (1ULL << (index - 192));
        else bitboards[4] |= (1ULL << (index - 256));
    }

    inline uint64_t get_bit(int index) {
        if (index < 64) return (bitboards[0] >> index) & 1ULL;
        else if (index < 128) return (bitboards[1] >> (index - 64)) & 1ULL;
        else if (index < 192) return (bitboards[2] >> (index - 128)) & 1ULL;
        else if (index < 256) return (bitboards[3] >> (index - 192)) & 1ULL;
        else return (bitboards[4] >> (index - 256)) & 1ULL;
    }

    inline std::map<Coord, int> get_board() {
        std::map<Coord, int> board;
        for (int i = 0; i < 96; i++) {
            Coord coord = all_coordinates[i];

            for (int player = 0; player < 3; player++) {
                if (get_bit(player * 96 + i)) {
                    board[coord] = player;
                    break;
                }
            }
        }

        return board;
    }
};

struct CombinedData {
    std::array<int, THREADS> num_games;
    std::array<std::vector<PolicyData>, THREADS> thread_data;
    bool stopped = false;
};

void gen_data_thread(CombinedData& combined_data, int thread_id);

std::string get_input();

void input_thread(CombinedData& combined_data);

void gen_data(CombinedData& combined_data);


#endif //DC5_DATAGEN_H
