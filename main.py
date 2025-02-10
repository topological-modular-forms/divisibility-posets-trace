import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # progress bar

def divisibility_pairs(n):
    """Return a list of pairs (k, l) with 1 <= k, l <= n such that k divides l."""
    return [(k, l) for k in range(1, n+1) for l in range(1, n+1) if l % k == 0]

def is_monotone(f, n, div_pairs):
    """
    Check if the function f (given as a tuple of length n) is monotone with respect
    to the divisibility order. That is, if k divides l then f(k) divides f(l).
    (Remember: f[i] represents f(i+1).)
    """
    for k, l in div_pairs:
        if f[l-1] % f[k-1] != 0:
            return False
    return True

def generate_monotone_maps(n):
    """
    Generate all functions f: {1,...,n} -> {1,...,n} that are monotone
    with respect to the divisibility order.
    """
    div_pairs = divisibility_pairs(n)
    maps = []
    for f in itertools.product(range(1, n+1), repeat=n):
        if is_monotone(f, n, div_pairs):
            maps.append(f)
    return maps

def compose(f, g):
    """
    Compose two functions f and g (each represented as a tuple of length n)
    to return the composition f ∘ g, i.e. (f ∘ g)(x) = f(g(x)).
    """
    return tuple(f[g[i]-1] for i in range(len(f)))

# A simple union–find (disjoint set) implementation.
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1

def process_for_index(args):
    """
    Worker function to process one outer index.
    For a given i (with corresponding map p), iterate over all q (by index)
    and compute the pair (f_index, g_index) where:
      f = p ∘ q   and   g = q ∘ p.
    Returns a list of such pairs.
    """
    i, monotone_maps, map_to_index = args
    edges = []
    p = monotone_maps[i]
    N = len(monotone_maps)
    for j in range(N):
        q = monotone_maps[j]
        f = compose(p, q)
        g = compose(q, p)
        # Look up the indices for the composed maps.
        f_index = map_to_index[f]
        g_index = map_to_index[g]
        edges.append((f_index, g_index))
    return edges

def main():
    n = 10
    
    # Step 1: Generate the monotone maps.
    print("Generating monotone maps...")
    monotone_maps = generate_monotone_maps(n)
    N = len(monotone_maps)
    print(f"Number of monotone maps in End(D_{n}): {N}")
    
    # Build a dictionary mapping each monotone map (tuple) to its index.
    map_to_index = {f: i for i, f in enumerate(monotone_maps)}
    
    # Initialize the union-find structure.
    uf = UnionFind(N)
    
    print("Processing union-find in parallel using 12 cores with progress bar...")
    # Prepare arguments for each worker: each gets an index i, along with
    # the monotone maps and lookup dictionary.
    args_list = [(i, monotone_maps, map_to_index) for i in range(N)]
    
    # Use ProcessPoolExecutor with 12 workers.
    with ProcessPoolExecutor(max_workers=12) as executor:
        # Wrap the iterator in tqdm to show progress.
        for edge_list in tqdm(executor.map(process_for_index, args_list),
                              total=len(args_list),
                              desc="Processing"):
            # Process each batch of union edges immediately.
            for (u, v) in edge_list:
                uf.union(u, v)
    
    # Count the number of connected components (i.e. conjugacy classes).
    components = set(uf.find(i) for i in range(N))
    num_components = len(components)
    print(f"\nNumber of conjugacy classes of End(D_{n}): {num_components}")

if __name__ == '__main__':
    main()
