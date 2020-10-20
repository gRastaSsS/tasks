import numpy as np
import random
import heapdict
import time

from networkx import dense_gnm_random_graph


def generate_graph(v=100, e=500):
    graph = dense_gnm_random_graph(v, e)

    adj_matrix = np.zeros((v, v))

    for edge in graph.edges:
        weight = random.randrange(1, 50)
        adj_matrix[edge[0], edge[1]] = weight
        adj_matrix[edge[1], edge[0]] = weight

    return adj_matrix


def generate_grid(linear_size=10, obstacles=30):
    grid = np.full((linear_size, linear_size), False, dtype=bool)
    free_cells = set()

    for i in range(linear_size):
        for j in range(linear_size):
            if i == 0 and j == 0:
                continue
            if i == linear_size - 1 and j == linear_size - 1:
                continue

            free_cells.add((i, j))

    while obstacles > 0:
        coordinates = random.sample(free_cells, 1)[0]
        free_cells.remove(coordinates)
        obstacles -= 1
        i, j = coordinates
        grid[i, j] = True

    return grid


def retrieve_edges(matrix):
    result = []
    for i in range(matrix.shape[0]):
        for j in range(i):
            if matrix[i, j] > 0:
                result.append((i, j))
                result.append((j, i))

    return result


def djikstra(adj_matrix, v0):
    distance = np.full(adj_matrix.shape[0], 0, dtype=np.int64)
    prev = np.zeros(adj_matrix.shape[0])
    Q = heapdict.heapdict()
    Q[v0] = 0

    for vertex in range(adj_matrix.shape[0]):
        if vertex != v0:
            distance[vertex] = np.iinfo(np.int64).max
            prev[vertex] = None

        Q[vertex] = distance[vertex]

    while Q:
        u = Q.popitem()[0]
        for v in range(adj_matrix.shape[0]):
            weight = adj_matrix[u, v]
            if weight != 0:
                alt = distance[u] + weight
                if alt < distance[v]:
                    distance[v] = alt
                    prev[v] = u
                    Q[v] = alt

    def translate(true_val):
        if true_val == np.iinfo(np.int64).max:
            return None
        return int(true_val)

    return [translate(x) for x in distance], prev


def bellman_ford(adj_matrix, v0):
    edges = retrieve_edges(adj_matrix)
    distance = np.full(adj_matrix.shape[0], 0, dtype=np.int64)
    prev = np.zeros(adj_matrix.shape[0])

    for vertex in range(adj_matrix.shape[0]):
        distance[vertex] = np.iinfo(np.int64).max
        prev[vertex] = None

    distance[v0] = 0

    for _ in range(adj_matrix.shape[0]):
        for (u, v) in edges:
            weight = adj_matrix[u, v]
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                prev[v] = u

    for (u, v) in edges:
        weight = adj_matrix[u, v]
        if distance[u] + weight < distance[v]:
            raise ValueError("Graph contains a negative-weight cycle")

    def translate(true_val):
        if true_val == np.iinfo(np.int64).max:
            return None
        return int(true_val)

    return [translate(x) for x in distance], prev


def a_star(grid, start, goal, h):
    def reconstruct_path(path_dict, cur):
        total_path = [cur]
        while cur != start:
            cur = path_dict[cur]
            total_path.append(cur)

        total_path.reverse()
        return total_path

    def get_neighbors(grid, p):
        def has_cell(i, j):
            if i < 0 or i >= grid.shape[0]:
                return False
            if j < 0 or j >= grid.shape[1]:
                return False
            return not grid[i, j]

        neighbors = []
        if has_cell(p[0] + 1, p[1]):
            neighbors.append((p[0] + 1, p[1]))
        if has_cell(p[0] - 1, p[1]):
            neighbors.append((p[0] - 1, p[1]))
        if has_cell(p[0], p[1] + 1):
            neighbors.append((p[0], p[1] + 1))
        if has_cell(p[0], p[1] - 1):
            neighbors.append((p[0], p[1] - 1))

        return neighbors

    frontier = heapdict.heapdict()
    frontier[start] = 0
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current = frontier.popitem()[0]

        if current == goal:
            return reconstruct_path(came_from, current)

        for next in get_neighbors(grid, current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + h(next, goal)
                frontier[next] = priority
                came_from[next] = current

    return None


def black_hole(arg):
    return


def dummy_benchmark(fun, data, runs):
    t1 = time.perf_counter_ns()
    for _ in range(runs):
        black_hole(fun(**data))
    t2 = time.perf_counter_ns()
    return (t2 - t1) / runs


if __name__ == '__main__':
    g_1 = generate_graph()

    print("djikstra_average_time_ns", dummy_benchmark(djikstra, {'adj_matrix': g_1, 'v0': 0}, 10))
    print("bellman_average_time_ns", dummy_benchmark(bellman_ford, {'adj_matrix': g_1, 'v0': 0}, 10))

    g_2 = generate_grid()
    for i in range(5):
        params = {
            'grid': g_2,
            'start': (i, 0),
            'goal': (g_2.shape[0] - 1, g_2.shape[1] - 1),
            'h': lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
        }

        print("a_star_average_time_ns", i, dummy_benchmark(a_star, params, 20))

    # print(g_2)
    # print(a_star(g_2, (0, 0), (g_2.shape[0] - 1, g_2.shape[1] - 1), lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])))
