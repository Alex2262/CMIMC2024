//
// Created by Alexander Tian on 4/26/24.
//

#ifndef DC5_UTIL_H
#define DC5_UTIL_H

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <string>

constexpr int GRID_RADIUS = 4;

struct Coord {
    int x;
    int y;
    int z;

    bool operator<(const Coord coord2) const {
        if (x < coord2.x) return true;
        if (x == coord2.x) {
            if (y < coord2.y) return true;
            if (y == coord2.y) {
                if (z < coord2.z) return true;
            }
        }

        return false;
    }
};

const Coord no_coord = {0, 0, 0};

std::vector<Coord> get_neighbors(Coord coord);
std::vector<Coord> select_valid(std::vector<Coord>& inp);

const int COMPONENT_TABLE[5] = {0, 0, 1, 3, 0};

std::vector<Coord> get_all_coordinates();

std::map<Coord, std::vector<Coord>> get_all_valid_neighbors();

int get_diameter(std::map<Coord, int>& board, Coord start_coord, std::set<Coord>& visited);

std::array<int, 3> score(std::map<Coord, int>& board);

const auto all_coordinates = get_all_coordinates();
inline std::map<Coord, std::vector<Coord>> all_valid_neighbors = get_all_valid_neighbors();

struct Move {
    Coord coord;
    bool pass;
};

const Move pass_move = {no_coord, true};

const std::string WHITE_COLOR = "\033[1;37m";
const std::string BLACK_COLOR = "\033[1;30m";
const std::string RED_COLOR = "\033[1;31m";
const std::string GREEN_COLOR = "\033[1;32m";
const std::string RESET_COLOR = "\033[0m";

inline std::pair<int, int> hex_to_pixel(Coord coord) {
    int x = coord.x, y = coord.y, z = coord.z;
    int xc = std::round(2.0 * (x / 2.0 + y / 2.0 - z));
    int yc = std::round(-2.0 * std::sqrt(3.0) * (x * std::sqrt(3.0) / 2.0 - y * std::sqrt(3.0) / 2.0));
    return std::make_pair(xc, yc);
}

inline void print_board(std::map<Coord, int>& board, Move move) {
    std::vector<std::vector<int>> mat(43, std::vector<int>(23, -2));

    for (const auto coord : all_coordinates) {
        auto [xc, yc] = hex_to_pixel(coord);
        mat[yc + 21][xc + 11] = -1;
    }

    for (const auto& it : board) {
        auto [xc, yc] = hex_to_pixel(it.first);
        mat[yc + 21][xc + 11] = it.second;
    }

    if (!move.pass) {
        auto [xc, yc] = hex_to_pixel(move.coord);
        mat[yc + 21][xc + 11] = 3;
    }

    for (int row = 0; row < 43; row++) {
        std::string s;
        for (int col = 0; col < 23; col++) {
            switch (mat[row][col]) {
                case -2:
                    s += " ";
                    break;
                case -1:
                    s += "_";
                    break;
                case 0:
                    s += WHITE_COLOR + "W" + RESET_COLOR;
                    break;
                case 1:
                    s += BLACK_COLOR + "B" + RESET_COLOR;
                    break;
                case 2:
                    s += RED_COLOR + "R" + RESET_COLOR;
                    break;
                case 3:
                    s += GREEN_COLOR + "G" + RESET_COLOR;
                    break;
            }
        }

        if (s.find_first_not_of(' ') != std::string::npos) {
            std::cout << s << std::endl;
        }
    }
}



#endif //DC5_UTIL_H
