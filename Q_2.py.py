import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def birkhoff_decomposition(A, tol=1e-8):
    """
    Performs Birkhoff–von Neumann decomposition of a doubly-stochastic matrix A.
    Yields tuples (iteration, matching, weight, residual_matrix).
    """
    A_current = A.copy().astype(float)
    n = A_current.shape[0]
    if A_current.shape[0] != A_current.shape[1]:
        raise ValueError("Input matrix must be square.")
    row_sums = A_current.sum(axis=1)
    col_sums = A_current.sum(axis=0)
    if not (np.allclose(row_sums, 1, atol=tol) and np.allclose(col_sums, 1, atol=tol)):
        print("\033[1;31mWarning: input matrix will not produce a balanced graph, algo might fail.\033[0m")
    iteration = 1
    left  = [("l", i) for i in range(n)]
    right = [("r", j) for j in range(n)]

    while True:
        B = nx.Graph()
        B.add_nodes_from(left,  bipartite=0)
        B.add_nodes_from(right, bipartite=1)
        for i in range(n):
            for j in range(n):
                if A_current[i, j] > tol:
                    B.add_edge(("l", i), ("r", j))

        matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(B, left)
        M = [(node[1], mate[1]) for node, mate in matching.items() if node[0] == 'l']
        if len(M) < n:
            raise ValueError(f"No perfect matching found at iteration {iteration}.")

        p_j = min(A_current[i, j] for (i, j) in M)
        yield iteration, M, p_j, A_current.copy()

        for (i, j) in M:
            A_current[i, j] -= p_j
            if A_current[i, j] < tol:
                A_current[i, j] = 0.0

        if np.allclose(A_current, 0, atol=tol):
            break
        iteration += 1


def plot_iteration(iteration, matching, p_j, A, tol=1e-8):
    """
    Plots the bipartite graph of the residual matrix A,
    highlighting the extracted matching and placing edge labels with offsets to avoid overlap.
    """
    n = A.shape[0]
    B = nx.Graph()
    pos = {}
    for i in range(n):
        B.add_node(("l", i)); pos[("l", i)] = (0, n - 1 - i)
        B.add_node(("r", i)); pos[("r", i)] = (2, n - 1 - i)

    edges = [(("l", i), ("r", j)) for i in range(n) for j in range(n) if A[i, j] > tol]
    B.add_edges_from(edges)

    fig, ax = plt.subplots()
    # draw all edges lightly
    nx.draw_networkx_edges(B, pos, edgelist=edges, edge_color='gray', alpha=0.5, ax=ax)
    # highlight matching edges
    match_edges = [(("l", i), ("r", j)) for (i, j) in matching]
    nx.draw_networkx_edges(B, pos, edgelist=match_edges, width=2, edge_color='red', ax=ax)
    # draw nodes
    nx.draw_networkx_nodes(B, pos, nodelist=[("l", i) for i in range(n)], node_color='skyblue', node_size=600, ax=ax)
    nx.draw_networkx_nodes(B, pos, nodelist=[("r", i) for i in range(n)], node_color='lightgreen', node_size=600, ax=ax)
    labels = { ("l", i): f"X{i}" for i in range(n) }
    labels.update({ ("r", i): f"Y{i}" for i in range(n) })
    nx.draw_networkx_labels(B, pos, labels, font_size=10, ax=ax)

    # manually place edge weight labels with slight offset to avoid collisions
    for (u, v) in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # perpendicular direction for offset
        dx, dy = x2 - x1, y2 - y1
        norm = np.array([-dy, dx])
        if np.linalg.norm(norm) != 0:
            norm = norm / np.linalg.norm(norm)
        offset = 0.1  # adjust as needed
        lx, ly = mx + offset * norm[0], my + offset * norm[1]
        weight = A[u[1], v[1]]
        ax.text(lx, ly, f"{weight:.2f}", fontsize=8,
                ha='center', va='center', backgroundcolor='white')

    ax.set_title(f"Iteration {iteration}: p_{iteration} = {p_j:.2f}")
    ax.axis('off')
    plt.show()


def demonstrate_birkhoff(A):
    """
    Runs the full decomposition and plots each step.
    """
    for iteration, M, p_j, A_step in birkhoff_decomposition(A):
        plot_iteration(iteration, M, p_j, A_step)


if __name__ == '__main__':
    # Like the example from the first clause of question 2 that i submitted.
    A = np.array([
        [0.4, 0  , 0.6,  0],
        [0.2, 0.6, 0.2,  0],
        [0  , 0.4, 0  ,0.6],
        [0.4, 0  , 0.2,0.4]
    ])
    
    # Example for a matrix that will not produce a balanced graph

    # A = np.array([
    #     [0.4, 0  , 0.6,  0],
    #     [0.2, 0.1, 0.2,  0],
    #     [0  , 0.4, 0  ,0.1],
    #     [0.4, 0  , 0.2,0.4]
    # ])
    demonstrate_birkhoff(A)
