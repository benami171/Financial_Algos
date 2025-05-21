# Birkhoff Decomposition Demo

A small Python module to perform and visualize the Birkhoff decomposition of any doubly‑stochastic matrix.

## Requirements

* Python 3.7+
* NumPy
* NetworkX
* Matplotlib

You can install dependencies with:

```bash
pip install numpy networkx matplotlib
```

## Usage

1. Clone or download this repository.

2. Open a terminal in the project folder.

3. Run the demo script:

   ```bash
   python q_2.py
   ```

4. The script will pop up a series of plots—one per iteration—showing the residual bipartite graph, the extracted matching (in red), and edge weights.

## Custom Matrix

To decompose your own matrix:

* Edit the `A` variable at the bottom of `birkhoff.py` with your $n \times n$ doubly‑stochastic matrix (rows and columns sum to 1).
* Rerun the script.

```python
# example 4×4 matrix
a = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.40, 0.10, 0.30, 0.20],
    [0.20, 0.30, 0.40, 0.10],
    [0.15, 0.35, 0.25, 0.25]
])
demonstrate_birkhoff(a)
```

## Notes

* If your matrix isn’t exactly doubly‑stochastic (within a small tolerance), the script will print a **bold red** warning.

---
