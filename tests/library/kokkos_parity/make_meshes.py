"""Write the Q2 meshes used by kokkos_parity.py.

Requires numpy and SEACAS exodus3 on PYTHONPATH:

    python3 make_meshes.py <output-directory>

2D meshes (9-node quadrilaterals):
  cavity2d_distorted  unit square, 6x6 elements, interior nodes displaced
  channel2d           [0, 2] x [0, 1], 8x4 elements
  node sets: 1 bottom, 2 right, 3 top without corners, 4 left, 5 top,
             100 node (0, 0)

3D meshes (27-node hexahedra in Goma's node ordering):
  cube3d, cube3d_distorted   unit cube, 3x3x3 elements
  node sets: 1 boundary except the lid interior, 3 lid (z = 1) interior,
             5 duct walls (y = 0, y = 1, z = 0, z = 1), 100 node (0, 0, 0)

Distortion moves only interior nodes, along a smooth map that is the identity
on the boundary, so elements have curved edges and boundaries stay planar.
Every mesh has side set 10 on its first element row (parallel decomposition
expects side-set metadata).
"""

import math
import sys
from pathlib import Path

import exodus3 as exodus

# Reference-node offsets in Goma's order (mm_kokkos_basis.hpp: Q2Q1::node_coordinate).
QUAD9 = [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
HEX27 = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    (0, -1, -1), (1, 0, -1), (0, 1, -1), (-1, 0, -1),
    (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),
    (0, -1, 1), (1, 0, 1), (0, 1, 1), (-1, 0, 1),
    (0, 0, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0),
    (1, 0, 0), (0, -1, 0), (0, 1, 0),
]  # fmt: skip


def bump(*unit):
    return math.prod(math.sin(math.pi * u) for u in unit)


def write(path, dim, elements, lengths, distortion, node_sets):
    counts = [2 * n + 1 for n in elements]
    ids = {}
    coordinates = [[] for _ in range(dim)]
    for index in range(math.prod(counts)):
        grid, rest = [], index
        for count in counts:
            grid.append(rest % count)
            rest //= count
        unit = [g / (c - 1) for g, c in zip(grid, counts)]
        shift = distortion * bump(*unit)
        for d in range(dim):
            # Shear each direction by the next one so the map stays one-to-one.
            u = unit[d] + shift * math.sin(math.pi * unit[(d + 1) % dim])
            coordinates[d].append(lengths[d] * u)
        ids[tuple(grid)] = index + 1
    table = QUAD9 if dim == 2 else HEX27
    connectivity = []
    for index in range(math.prod(elements)):
        base, rest = [], index
        for n in elements:
            base.append(2 * (rest % n))
            rest //= n
        for offsets in table:
            connectivity.append(ids[tuple(b + 1 + o for b, o in zip(base, offsets))])
    sets = {
        k: sorted(ids[g] for g in ids if member(g, counts))
        for k, member in node_sets.items()
    }
    path.unlink(missing_ok=True)
    mesh = exodus.exodus(
        str(path),
        mode="w",
        array_type="numpy",
        title=path.stem,
        numDims=dim,
        numNodes=len(ids),
        numElems=len(connectivity) // len(table),
        numBlocks=1,
        numNodeSets=len(sets),
        numSideSets=1,
    )
    mesh.put_coord_names(["x", "y", "z"][:dim])
    mesh.put_coords(*coordinates, *([[0.0] * len(ids)] if dim == 2 else []))
    blocks = len(connectivity) // len(table)
    mesh.put_elem_blk_info(1, "QUAD9" if dim == 2 else "HEX27", blocks, len(table), 0)
    mesh.put_elem_connectivity(1, connectivity)
    for k, nodes in sets.items():
        mesh.put_node_set_params(k, len(nodes), 0)
        mesh.put_node_set(k, nodes)
    mesh.put_side_set_params(10, elements[0], 0)
    mesh.put_side_set(10, list(range(1, elements[0] + 1)), [1] * elements[0])
    mesh.close()


def main(directory):
    directory = Path(directory)
    sets2d = {
        1: lambda g, c: g[1] == 0,
        2: lambda g, c: g[0] == c[0] - 1,
        3: lambda g, c: g[1] == c[1] - 1 and 0 < g[0] < c[0] - 1,
        4: lambda g, c: g[0] == 0,
        5: lambda g, c: g[1] == c[1] - 1,
        100: lambda g, c: g == (0, 0),
    }
    write(directory / "cavity2d_distorted.exoII", 2, [6, 6], [1, 1], 0.08, sets2d)
    write(directory / "channel2d.exoII", 2, [8, 4], [2, 1], 0.0, sets2d)

    def boundary(g, c):
        return any(x == 0 or x == n - 1 for x, n in zip(g, c))

    def lid(g, c):
        return g[2] == c[2] - 1 and all(0 < x < n - 1 for x, n in zip(g[:2], c[:2]))

    sets3d = {
        1: lambda g, c: boundary(g, c) and not lid(g, c),
        3: lid,
        5: lambda g, c: g[1] in (0, c[1] - 1) or g[2] in (0, c[2] - 1),
        100: lambda g, c: g == (0, 0, 0),
    }
    write(directory / "cube3d.exoII", 3, [3, 3, 3], [1, 1, 1], 0.0, sets3d)
    write(directory / "cube3d_distorted.exoII", 3, [3, 3, 3], [1, 1, 1], 0.06, sets3d)


if __name__ == "__main__":
    main(sys.argv[1])
