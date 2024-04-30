//
// Created by Alexander Tian on 4/26/24.
//

#include "util.h"

std::vector<Coord> get_neighbors(Coord coord) {
    std::vector<Coord> neighbors;

    neighbors.push_back({coord.x + 1, coord.y, coord.z});
    neighbors.push_back({coord.x - 1, coord.y, coord.z});
    neighbors.push_back({coord.x, coord.y + 1, coord.z});
    neighbors.push_back({coord.x, coord.y - 1, coord.z});
    neighbors.push_back({coord.x, coord.y, coord.z + 1});
    neighbors.push_back({coord.x, coord.y, coord.z - 1});

    return neighbors;
}

std::vector<Coord> select_valid(std::vector<Coord>& inp) {
    std::vector<Coord> valid;

    for (auto coord : inp) {
        if (coord.x + coord.y + coord.z < 1 or coord.x + coord.y + coord.z > 2) continue;

        if (coord.x > GRID_RADIUS or coord.x < -GRID_RADIUS + 1) continue;
        if (coord.y > GRID_RADIUS or coord.y < -GRID_RADIUS + 1) continue;
        if (coord.z > GRID_RADIUS or coord.z < -GRID_RADIUS + 1) continue;

        valid.push_back(coord);
    }

    return valid;
}

std::vector<Coord> get_all_coordinates() {
    std::vector<Coord> coordinates;

    for (int x = -GRID_RADIUS + 1; x <= GRID_RADIUS; x++) {
        for (int y = -GRID_RADIUS + 1; y <= GRID_RADIUS; y++) {
            for (int z = -GRID_RADIUS + 1; z <= GRID_RADIUS; z++) {
                if (x + y + z < 1 or x + y + z > 2) continue;
                coordinates.push_back({x, y, z});
            }
        }
    }

    return coordinates;
}

std::map<Coord, std::vector<Coord>> get_all_valid_neighbors() {
    std::vector<Coord> coordinates = get_all_coordinates();
    std::map<Coord, std::vector<Coord>> m;

    for (Coord coord : coordinates) {
        std::vector<Coord> neighbors = get_neighbors(coord);
        m[coord] = select_valid(neighbors);
    }

    return m;
}


int get_diameter(std::map<Coord, int>& board, Coord start_coord, std::set<Coord>& visited) {

    std::map<Coord, int> connected;

    // std::queue<Coord> con_queue;

    std::function<void(Coord coord)> con;

    int player = board[start_coord];

    con = [&](Coord coord){
        visited.insert(coord);
        connected[coord] = -1;

        int cnt = 0;

        for (auto neighbor : all_valid_neighbors[coord]) {
            if (board.find(neighbor) == board.end() || board[neighbor] != player) continue;

            cnt++;
            if (connected.find(neighbor) == connected.end()) con(neighbor);
        }

        connected[coord] = cnt;
    };

    /*
    std::function<int(Coord coord, std::set<Coord>& new_visited)> dfs;
    dfs = [&](Coord coord, std::set<Coord>& new_visited) {
        int max_path_length = 0;

        for (auto neighbor : get_neighbors(coord)) {
            if (board.find(neighbor) == board.end()
                || board[neighbor] != player
                || new_visited.find(neighbor) == new_visited.end()) continue;

            new_visited.insert(neighbor);
            int path_length = dfs(neighbor, new_visited);
            new_visited.erase(new_visited.find(neighbor));

            max_path_length = std::max(max_path_length, path_length);
        }

        return max_path_length + 1;
    };
    */

    con(start_coord);

    int s = static_cast<int>(connected.size());

    if (s <= 3) return s;

    if (s >= 4 && s <= 5) {
        for (auto& it : connected) {
            if (it.second == 3) return s - 1;
        }

        return s;
    }

    if (s == 6) {
        bool encountered = false;
        for (auto& it : connected) {
            if (!encountered && it.second == 3) encountered = true;
            if (encountered && it.second == 3) return 4;
        }

        return 5;
    }

    return 5;
}

std::array<int, 3> score(std::map<Coord, int>& board) {
    std::array<int, 3> scores{};
    std::set<Coord> visited;

    for (auto& it : board) {
        if (visited.find(it.first) == visited.end()) {
            int d = get_diameter(board, it.first, visited);

            if (d) scores[it.second] += COMPONENT_TABLE[d - 1];
        }
    }

    return scores;
}
