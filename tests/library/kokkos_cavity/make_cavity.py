"""Write the unit-square QUAD9 cavity mesh used by kokkos_cavity.py.

Node sets: 1 bottom, 2 right, 3 lid (without corners), 4 left, 100 the
bottom-left corner, 101 the bottom midside node next to it (no pressure DOF).
Side set 10 is the bottom wall. Requires numpy and SEACAS exodus3 on PYTHONPATH:

    python3 make_cavity.py cavity.exoII 8
"""

import sys
import exodus3 as exodus


def main(path, n):
    m = 2 * n + 1
    gid = lambda i, j: j * m + i + 1
    xs = [i / (m - 1) for j in range(m) for i in range(m)]
    ys = [j / (m - 1) for j in range(m) for i in range(m)]
    conn = []
    for ej in range(n):
        for ei in range(n):
            i, j = 2 * ei, 2 * ej
            conn += [
                gid(i, j),
                gid(i + 2, j),
                gid(i + 2, j + 2),
                gid(i, j + 2),
                gid(i + 1, j),
                gid(i + 2, j + 1),
                gid(i + 1, j + 2),
                gid(i, j + 1),
                gid(i + 1, j + 1),
            ]
    sets = {
        1: [gid(i, 0) for i in range(m)],
        2: [gid(m - 1, j) for j in range(m)],
        3: [gid(i, m - 1) for i in range(1, m - 1)],
        4: [gid(0, j) for j in range(m)],
        100: [gid(0, 0)],
        101: [gid(1, 0)],
    }
    e = exodus.exodus(
        path,
        mode="w",
        array_type="numpy",
        title="cavity",
        numDims=2,
        numNodes=m * m,
        numElems=n * n,
        numBlocks=1,
        numNodeSets=len(sets),
        numSideSets=1,
    )
    e.put_coord_names(["x", "y"])
    e.put_coords(xs, ys, [0.0] * len(xs))
    e.put_elem_blk_info(1, "QUAD9", n * n, 9, 0)
    e.put_elem_connectivity(1, conn)
    for k, nodes in sets.items():
        e.put_node_set_params(k, len(nodes), 0)
        e.put_node_set(k, nodes)
    # One bottom-wall side set; parallel decomposition expects side-set metadata.
    e.put_side_set_params(10, n, 0)
    e.put_side_set(10, list(range(1, n + 1)), [1] * n)
    e.close()


main(sys.argv[1], int(sys.argv[2]))
