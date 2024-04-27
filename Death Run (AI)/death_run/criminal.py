import networkx as nx
import random


class BaseCriminal:
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
        budget: int,
    ) -> tuple[int, int, int]:
        """
        :param edge_updates: A dictionary where the key is an edge (u, v) and the value is how much that edge's weight increased in the previous round.
        Note that this only contains information about edge updates in the previous round, and not rounds before that.
        :param vertex_count: A dictionary where the key is a vertex and the value is how many students are currently on that vertex.
        :param budget: The remaining budget
        :return: Which edge to attack and by how much. Must be a tuple of the form (u, v, w) where (u, v) represents the edge endpoints
        and w is the increase in edge weight. w must be in the range [0, budget].
        """

        for edge in list(self.graph.edges):
            edge_key = (edge[0], edge[1])

            if edge_key in edge_updates:
                self.graph[edge[0]][edge[1]]["weight"] += edge_updates[edge_key]

        best_value = 0
        best_action = None

        all_can_finish = True
        num_students_left = 0

        for vertex in vertex_count.keys():
            num_students = vertex_count[vertex]

            if num_students == 0:
                continue

            num_students_left += num_students

            for adj in self.graph.adj[vertex]:
                if adj not in self.ends:
                    all_can_finish = False
                    break

            if not all_can_finish:
                break

        for vertex in vertex_count.keys():
            num_students = vertex_count[vertex]

            if num_students == 0:
                continue

            out_degree = self.graph.out_degree(vertex)

            if out_degree == 0:
                assert (vertex in self.ends)
                continue

            weights = {}

            reachable_end = []

            for adj in self.graph.adj[vertex]:
                weights[adj] = self.graph[vertex][adj]["weight"]

                if adj in self.ends:
                    reachable_end.append(adj)

            smallest_weight = min(weights.values())
            smallest_key = list(weights.keys())[list(weights.values()).index(smallest_weight)]

            smallest_weight2 = 0
            smallest_key2 = None

            for adj in self.graph.adj[vertex]:
                if adj == smallest_key:
                    continue

                if smallest_key2 is None or weights[adj] < smallest_weight2:
                    smallest_weight2 = weights[adj]
                    smallest_key2 = adj

            if all_can_finish:
                # We must spend the rest here, since game is over after this

                opt = smallest_weight2 - smallest_weight - 1
                opt = min(opt, budget)

                possible_value = opt * num_students

                if possible_value > best_value:
                    best_value = possible_value
                    best_action = (vertex, smallest_key, opt)

                continue

            if out_degree == 1:
                # We can spend all our budget on this, given that enough students pass over this one edge
                opt = min(round(num_students / 6.0 * budget), budget)
                possible_value = opt * num_students

                if possible_value > best_value:
                    best_value = possible_value
                    best_action = (vertex, list(self.graph.adj[vertex])[0], opt)

            elif out_degree == 2:
                # Let's say we are willing to spend up to 1/3 of our budget here
                # We have to ensure the smaller value stays as the smallest value

                if smallest_weight == smallest_weight2:
                    continue

                opt = smallest_weight2 - smallest_weight - 1
                opt = min(opt, min(round(num_students / 12.0 * budget), budget // 3))

                possible_value = opt * num_students

                # print(smallest_weight, smallest_weight2, opt, budget, num_students)

                if possible_value > best_value:
                    best_value = possible_value
                    best_action = (vertex, smallest_key, opt)

            else:
                # Spend like max 1/8th of our money otherwise

                if smallest_weight == smallest_weight2:
                    continue

                opt = smallest_weight2 - smallest_weight - 1
                opt = min(opt, min(round(num_students / 24.0 * budget), budget // 8))

                possible_value = opt * num_students

                if possible_value > best_value:
                    best_value = possible_value
                    best_action = (vertex, smallest_key, opt)

        if best_action is not None:
            return best_action

        # SINCE THERE IS NOTHING DECENT TO PLAY TO BLOCK THE STUDENTS, INVEST IN A LONG TERM STRATEGY
        # OF ADDING WEIGHTS TO THE ENDS

        # We will now attempt to minimize best_value, as a function of a few different things.
        # The number of in degrees of an endpoint, the number of students at the vertex that can reach the end, and
        # the shortest path length

        best_value = 0

        for vertex in vertex_count.keys():
            num_students = vertex_count[vertex]

            if num_students == 0:
                continue

            for end in self.ends:
                if not nx.has_path(self.graph, vertex, end):
                    continue

                in_degree = self.graph.in_degree(end)

                spl = nx.shortest_path_length(self.graph, vertex, end, weight="weight")

                #print(spl, in_degree, num_students)
                #print(vertex, end)

                value = 5 * in_degree + 10 * (8 - num_students) + spl

                if best_action is None or value < best_value:
                    #

                    sp = nx.shortest_path(self.graph, vertex, end, weight="weight")
                    opt = min(3, budget // 6)
                    best_action = (vertex, sp[-2], opt)

        return best_action




# Starter strategy
class RandomCriminal(BaseCriminal):
    def __init__(self, edge_list, begin, ends):
        self.edge_list = edge_list
        self.begin = begin
        self.ends = ends

    def strategy(self, edge_updates, vertex_count, budget):
        # Find a random populated vertex
        populated_vertices = list(
            filter(lambda z: vertex_count[z], vertex_count.keys())
        )
        vertex = random.choice(populated_vertices)
        # Fill in random out-edge with random weight
        return (
            vertex,
            random.choice(
                [x for (_, x, _) in filter(lambda z: z[0] == vertex, self.edge_list)]
            ),
            random.randint(0, budget),
        )
