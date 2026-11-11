#!/usr/bin/env python3
"""
Unit-square lid driven cavity mesh of 9-node quadrilaterals for the CFD-DEM example.

Node sets
  1   bottom wall  (y = 0)
  2   right wall   (x = 1)
  3   lid          (y = 1)
  4   left wall    (x = 0)
  100 every node   (used to pin VOID_FRACTION, the coupler overwrites it)

Side sets 1-4 are the same walls (Goma's built-in parallel decomposition needs at least one).

Usage
  PYTHONPATH=$SEACAS/lib python3 make_mesh.py [--n 20] [--out ldc_dem.exoII]
"""
import argparse
import os

import exodus


def build(n, out):
    nn = 2 * n + 1  # Q9: two node spacings per element edge
    h = 1.0 / (2 * n)
    xs, ys = [], []
    for j in range(nn):
        for i in range(nn):
            xs.append(i * h)
            ys.append(j * h)

    def nid(i, j):  # 1-based exodus node id on the (2n+1)^2 grid
        return j * nn + i + 1

    conn = []
    for ey in range(n):
        for ex in range(n):
            i, j = 2 * ex, 2 * ey
            # corners counter-clockwise, then midsides 0-1, 1-2, 2-3, 3-0, then center
            conn += [
                nid(i, j), nid(i + 2, j), nid(i + 2, j + 2), nid(i, j + 2),
                nid(i + 1, j), nid(i + 2, j + 1), nid(i + 1, j + 2), nid(i, j + 1),
                nid(i + 1, j + 1),
            ]

    sets = {
        1: [nid(i, 0) for i in range(nn)],
        2: [nid(nn - 1, j) for j in range(nn)],
        3: [nid(i, nn - 1) for i in range(nn)],
        4: [nid(0, j) for j in range(nn)],
        100: list(range(1, nn * nn + 1)),
    }

    # element sides on each wall; exodus QUAD sides: 1 bottom, 2 right, 3 top, 4 left
    def eid(ex, ey):
        return ey * n + ex + 1

    sides = {
        1: ([eid(ex, 0) for ex in range(n)], [1] * n),
        2: ([eid(n - 1, ey) for ey in range(n)], [2] * n),
        3: ([eid(ex, n - 1) for ex in range(n)], [3] * n),
        4: ([eid(0, ey) for ey in range(n)], [4] * n),
    }

    if os.path.exists(out):
        os.remove(out)
    exo = exodus.exodus(out, mode="w", array_type="ctype", title="CFD-DEM lid driven cavity",
                        numDims=2, numNodes=nn * nn, numElems=n * n, numBlocks=1,
                        numNodeSets=len(sets), numSideSets=len(sides))
    exo.put_coord_names(["x", "y"])
    exo.put_coords(xs, ys, [0.0] * len(xs))
    exo.put_elem_blk_info(1, "QUAD9", n * n, 9, 0)
    exo.put_elem_connectivity(1, conn)
    for sid, nodes in sets.items():
        exo.put_node_set_params(sid, len(nodes))
        exo.put_node_set(sid, nodes)
    for sid, (elems, sds) in sides.items():
        exo.put_side_set_params(sid, len(elems), 0)
        exo.put_side_set(sid, elems, sds)
    exo.close()
    print(f"wrote {out}: {n}x{n} QUAD9 elements, {nn * nn} nodes, h = {1.0 / n:g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="elements per side (default 20)")
    ap.add_argument("--out", default="ldc_dem.exoII")
    args = ap.parse_args()
    build(args.n, args.out)
