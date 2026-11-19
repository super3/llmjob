# How much linear dependence is there inside one 128-wide chunk of Pearl's noise?
#
# In a chunk, k index kk carries an A edge (pA0, pA1) and a B edge (pB0, pB1) on 128 rank vertices.
# The noised product's chunk update is  dC = sum_kk (La e_pA0 - La e_pA1)(Lb e_pB0 - Lb e_pB1)^T,
# i.e. La * M * Lb^T with M = S_A^T S_B. The inner dimension a GEMM needs is at least rank(M), and
# at most 128 (the direct form). This measures rank(S_A), rank(M), and how many k columns an
# int8-exact rewrite can drop: an A cycle lets one edge's column be written as the signed sum of the
# others (its B row is then added into theirs), which keeps int8 only if no row receives twice, so
# the usable cycles must be edge-disjoint across both graphs (greedy packing, shortest cycle first).
import random, sys
import numpy as np

R = 128

def perm_edges(rng):
    out = []
    for _ in range(R):
        u = rng.getrandbits(32)
        p0 = u & (R - 1)
        p1 = p0 ^ (1 + (((R - 1) * u) >> 32))
        out.append((p0, p1))
    return out

def comps(edges):
    par = list(range(R))
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb: par[ra] = rb
    return len({f(x) for x in range(R)})

def shortest_cycle(edges, alive):
    # shortest cycle among alive edges: for each alive edge (a,b), BFS a->b without it.
    adj = {}
    for i in alive:
        a, b = edges[i]
        adj.setdefault(a, []).append((b, i)); adj.setdefault(b, []).append((a, i))
    best = None
    for i in alive:
        a, b = edges[i]
        prev = {a: None}; q = [a]; found = False
        for x in q:
            if x == b: found = True; break
            for y, j in adj.get(x, []):
                if j == i or y in prev: continue
                prev[y] = (x, j); q.append(y)
        if not found: continue
        path = [i]; x = b
        while prev[x] is not None:
            x, j = prev[x]; path.append(j)
        if best is None or len(path) < len(best): best = path
    return best

def packing(eA, eB):
    used = set(); n = 0
    while True:
        alive = [i for i in range(R) if i not in used]
        cA = shortest_cycle(eA, alive); cB = shortest_cycle(eB, alive)
        cands = [c for c in (cA, cB) if c]
        if not cands: return n
        c = min(cands, key=len)
        used.update(c); n += 1

def inc(edges):
    S = np.zeros((R, R))
    for kk, (a, b) in enumerate(edges): S[kk, a] = 1; S[kk, b] = -1
    return S

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 200
rows = []
for t in range(trials):
    eA, eB = perm_edges(rng), perm_edges(rng)
    rA, rB = R - comps(eA), R - comps(eB)
    rM = np.linalg.matrix_rank(inc(eA).T @ inc(eB))
    rows.append((rA, rB, rM, packing(eA, eB)))
a = np.array(rows)
for name, col in (('rank S_A', 0), ('rank S_B', 1), ('rank M', 2), ('droppable (int8, greedy)', 3)):
    v = a[:, col]
    print(f'{name:26s} mean {v.mean():6.1f}  min {v.min():4d}  max {v.max():4d}')
print('chunks with >=16 droppable:', (a[:, 3] >= 16).mean(), ' >=32:', (a[:, 3] >= 32).mean())
print('chunks with rank M <= 112:', (a[:, 2] <= 112).mean(), ' <= 96:', (a[:, 2] <= 96).mean())
