import time
import subprocess
import os
import matplotlib.pyplot as plt
import re
import math
from collections import defaultdict
import heapdict
import random
import numpy as np
from scipy.optimize import curve_fit


def prims(graph, starting_vertex):
    mst = defaultdict(set)
    visited = {starting_vertex}
    edges = heapdict.heapdict()

    for to, cost in graph[starting_vertex].items():
        edges[(starting_vertex, to)] = cost

    while edges:
        (frm, to), cost = edges.popitem()

        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            for to_next, cost in graph[to].items():
                if to_next not in visited:
                    edges[(to, to_next)] = cost

    return mst


def run_benchmark(fun, args, runs):
    stats = []
    results = [''] * len(args)

    for i in range(len(args)):
        t1 = time.perf_counter_ns()

        for j in range(runs):
            result = fun(args[i])
            if j == runs - 1:
                results[i] = result

        t2 = time.perf_counter_ns()

        stats.append((t2 - t1) / runs)

        print(f'Step {i + 1} of {len(args)} completed!')

    return stats, results


def generate_complete_graphs(path, n):
    if n < 2:
        raise ValueError('n must be >= 2')

    if path is None:
        graphs = []

        for i in range(2, n + 1):
            graph = dict()

            for v0 in range(i):
                for v1 in range(v0):
                    if v0 != v1:
                        w = random.randint(1, 10)
                        if v0 + 1 not in graph:
                            graph[v0 + 1] = dict()
                        if v1 + 1 not in graph:
                            graph[v1 + 1] = dict()

                        graph[v0 + 1][v1 + 1] = w
                        graph[v1 + 1][v0 + 1] = w

            graphs.append(graph)

        return graphs

    else:
        files = []

        for i in range(2, n + 1):
            with open(os.path.join(path, f'complete_graph_{i}.txt'), 'w+') as file:
                for v0 in range(i):
                    for v1 in range(v0):
                        if v0 != v1:
                            file.write(f'     {v0 + 1}     {v1 + 1}\n')

                files.append(file.name)

        return files


def delete_files(files):
    for file in files:
        os.remove(file)


def save_to_file(path, stats, results):
    with open(path, 'w+') as file:
        for i in range(len(stats)):
            file.write(f'{stats[i]} {results[i]}\n')


def read_from_file(path):
    stats = []
    results = []

    with open(path) as file:
        for line in file:
            lines = line.split()
            stats.append(float(lines[0]))
            results.append(int(lines[1]))

    return stats, results


def run_first_part():
    graphs = generate_complete_graphs(None, 60)

    def run(graph):
        return prims(graph, 1)

    stats, results = run_benchmark(run, graphs, 10)

    def fit(n, a):
        return a * (n + n*(n-1)/2) * np.log(n)

    x_range = np.linspace(2, 60, num=59)
    y_range = stats

    pars, _ = curve_fit(f=fit,
                        xdata=x_range, ydata=stats, maxfev=5000,
                        p0=[1]
                        )

    plt.plot(x_range, fit(x_range, *pars), label="Theoretical")
    plt.plot(x_range, y_range, label="Runtime")
    plt.legend()
    plt.show()


def run_second_part():
    files = generate_complete_graphs('graphs', 60)

    def run(file_name):
        out = subprocess.run(['QC', '-f', file_name],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT
                             )
        return out.stdout

    stats, results = run_benchmark(run, files, 1)
    delete_files(files)

    extracted_results = []
    for res in results:
        regex_digits = re.findall(r'\d+', res.decode("utf-8"))[0]
        extracted_results.append(int(regex_digits))

    save_to_file(f'graphs\\results-{time.time_ns()}.txt', stats, extracted_results)

    plt.plot([x + 2 for x in range(len(files))], stats, label="Time consumption")
    plt.legend()
    plt.show()

    plt.plot([x + 2 for x in range(len(files))], extracted_results, label="Output")
    plt.plot([x + 2 for x in range(len(files))], [(x + 2) ** 4 / 64 for x in range(len(files))], label="Conjecture")
    plt.legend()
    plt.show()


def approx_results():
    stats, results = read_from_file("graphs\\results-1604061069406330700.txt")

    x_range = np.linspace(2, 60, num=59)
    y_range = []
    for i, x in enumerate(x_range):
        y_range.append((x + results[i]) * results[i] * (x*(x-1)/2))

    def fit(n, a):
        return a * (results + n) * results * (n*(n-1)/2)

    pars, _ = curve_fit(f=fit,
                        xdata=x_range, ydata=stats, maxfev=5000,
                        p0=[1]
                        )

    print(pars)

    plt.plot(x_range, fit(x_range, *pars), label="Theoretical")
    plt.plot(x_range, stats, label="Runtime")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_first_part()
    #approx_results()
