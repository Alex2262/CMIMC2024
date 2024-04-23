import networkx as nx
import random


class BaseStudent:
    def __init__(
        self, edge_list: list[tuple[int, int, int]], begin: int, ends: list[int]
    ) -> None:
        """
        :param edge_list: A list of tuples representing the edge list of the graph. Tuples are of the
        form (u, v, w), where (u, v) specifies that an edge between vertices u and v exist, and w is the
        weight of that edge.
        :param begin: The label of the vertex which students begin on.
        :param ends: A list of labels of vertices that students may end on (i.e. count as a valid exit).
        """

        self.edge_list = edge_list
        self.begin = begin
        self.ends = ends

        self.graph = nx.DiGraph()
        self.graph.add_weighted_edges_from(edge_list)

    def strategy(
        self,
        edge_updates: dict[tuple[int, int], int],
        vertex_count: dict[int, int],
        current_vertex: int,
    ) -> int:
        """
        :param edge_updates: A dictionary where the key is an edge (u, v) and the value is how much that edge's weight increased in the current round.
        Note that this only contains information about edge updates in the current round, and not previous rounds.
        :param vertex_count: A dictionary where the key is a vertex and the value is how many students are currently on that vertex.
        :param current_vertex: The vertex that you are currently on.
        :return: The label of the vertex to move to. The edge (current_vertex, next_vertex) must exist.
        """

        k_shortest_paths = 10

        for edge in list(self.graph.edges):
            edge_key = (edge[0], edge[1])

            if edge_key in edge_updates:
                self.graph[edge[0]][edge[1]]["weight"] += edge_updates[edge_key]

        best_weight = 0
        best_path = None

        short_paths = []

        # print("CALLED")
        for end in self.ends:

            if not nx.has_path(self.graph, current_vertex, end):
                continue

            # print("WOAH", best_path, best_weight, end)
            # print(current_vertex, self.graph)
            paths = nx.shortest_simple_paths(self.graph, current_vertex, end, weight="weight")
            # print("GOT PATHS")
            path_to_end = None
            curr_weight = 0

            paths_checked = 0

            for path in paths:

                if paths_checked > k_shortest_paths:
                    break

                short_paths.append(path)

                # print("CHECKING PATHS???", path)
                works = True

                for i in range(1, len(path)):
                    if self.graph.out_degree(path[i]) < 2:
                        works = False
                        break

                if not works:
                    paths_checked += 1
                    continue

                path_to_end = path
                break

            if path_to_end is None:
                continue

            for i in range(1, len(path_to_end)):
                curr_weight += self.graph[path_to_end[i - 1]][path_to_end[i]]["weight"]

            if best_path is None or curr_weight < best_weight:
                best_path = path_to_end
                best_weight = curr_weight

        # print("HI", best_weight, best_path)

        # We fucked up!!!!
        # We will simply choose the shortest path now.
        if best_path is None:
            return short_paths[0][1]

        return best_path[1]


# Starter strategy
class RandomStudent(BaseStudent):
    def __init__(self, edge_list, begin, ends):
        self.edge_list = edge_list
        self.begin = begin
        self.ends = ends

    def strategy(self, edge_updates, vertex_count, current_vertex):
        # Take a random out-edge
        return random.choice(
            [
                x
                for (_, x, _) in filter(
                    lambda z: z[0] == current_vertex, self.edge_list
                )
            ]
        )
