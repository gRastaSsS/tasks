import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys

from networkx.generators.random_graphs import dense_gnm_random_graph


def generate_graph(v=100, e=200):
    graph = dense_gnm_random_graph(v, e)

    adj_matrix = np.zeros((v, v))
    for edge in graph.edges:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1

    adj_list = [[] for _ in range(v)]
    for edge in graph.edges:
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])

    return graph, adj_matrix, adj_list


def find_components(adj_list):
    discovered = {}
    components = []

    def dfs(v, component):
        nonlocal discovered, adj_list
        discovered[v] = True
        component.append(v)
        for w in adj_list[v]:
            if w not in discovered:
                dfs(w, component)

    for vertex, _ in enumerate(adj_list):
        if vertex not in discovered:
            comp = []
            dfs(vertex, comp)
            components.append(comp)

    return components


def shortest_path(adj_list, v0, v1):
    def bfs():
        nonlocal v0, v1, adj_list
        discovered = {v0}
        parents = {v0: None}
        q = [v0]
        while q:
            v = q.pop()

            if v == v1:
                return parents

            for w in adj_list[v]:
                if w not in discovered:
                    discovered.add(w)
                    parents[w] = v
                    q.append(w)

        return parents

    parents = bfs()
    path = []
    parent = v1

    while parent is not None:
        path.append(parent)
        parent = parents.get(parent, None)

    path.reverse()
    return path


if __name__ == '__main__':
    graph, adj_matrix, adj_list = generate_graph()
    nx.draw_networkx(graph, node_size=10, with_labels=True, font_size=10, node_color='#000000')
    plt.show()

    np.set_printoptions(threshold=sys.maxsize)

    print(adj_matrix)
    print(adj_list)

    comps = find_components(adj_list)
    shortest = shortest_path(adj_list, 0, 6)
    print("Components", comps)
    print("Shortest", shortest)
