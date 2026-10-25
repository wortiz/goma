"""Compare Kokkos and legacy assembly fill by fill and solve by solve.

Each case runs twice in one goma_lib_sequence process: with the Kokkos backend
and GOMA_KOKKOS_PARITY=1, so every fill also runs the legacy element loop from
the same state and compares owned residual and matrix contributions before
scaling, then with the legacy backend. The Kokkos-assembled solutions must
match the legacy ones (exodiff, including velocity divergence), with the same
Newton iteration counts, residual history, and printed solution norms. The
body-force channel must also reproduce its exact Poiseuille solution.

Cases cover distorted 9-node quadrilaterals and regular and distorted 27-node
hexahedra; lid-driven, body-force, and decaying flows; steady and transient
(backward Euler and theta = 0.5) integration; hard-set and residual-mode
velocity conditions; residual-only line-search fills; non-unit, per-component
equation term multipliers; and one and two ranks.
Meshes come from kokkos_parity/make_meshes.py.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PARITY = """COORDINATES absolute 1.e-12
TIME STEPS relative 1.e-12
NODAL VARIABLES relative 1.e-8 floor 1.e-11
"""

MATERIAL = """---Physical Properties
Density                         = CONSTANT {density}
---Mechanical Properties and Constitutive Equations
Solid Constitutive Equation     = NONLINEAR
Convective Lagrangian Velocity  = NONE
Lame MU                         = CONSTANT 1.
Lame LAMBDA                     = CONSTANT 1.
Liquid Constitutive Equation    = NEWTONIAN
Viscosity                       = CONSTANT {viscosity}
Surface Tension                 = CONSTANT 0.
---Thermal Properties
Conductivity                    = CONSTANT 1.
Heat Capacity                   = CONSTANT 1.
Volume Expansion                = CONSTANT 0.
Reference Temperature           = CONSTANT 0.
Liquidus Temperature            = CONSTANT 1.
Solidus Temperature             = CONSTANT 1.
---Electrical Properties
Electrical Conductivity         = CONSTANT 1.
---Microstructure Properties
Media Type                      = CONTINUOUS
---Source Terms
Navier-Stokes Source            = CONSTANT {source}
Solid Body Source               = CONSTANT 0. 0. 0.
Mass Source                     = CONSTANT 0.
Heat Source                     = CONSTANT 0.
Current Source                  = CONSTANT 0.
"""

STRATIMIKOS = """<ParameterList>
  <Parameter name="Linear Solver Type" type="string" value="Amesos2"/>
  <ParameterList name="Linear Solver Types">
    <ParameterList name="Amesos2">
      <Parameter name="Solver Type" type="string" value="KLU2"/>
    </ParameterList>
  </ParameterList>
  <Parameter name="Preconditioner Type" type="string" value="None"/>
</ParameterList>
"""


def walls(dim, sets, components=None):
    names = ["U", "V", "W"][:dim]
    return [f"BC = {n} NS {s} 0." for s in sets for n in (components or names)]


CASES = [
    {
        # Re = 100 lid, residual-mode lid condition, backtracking line search.
        "name": "cavity2d_distorted",
        "mesh": "cavity2d_distorted.exoII",
        "dim": 2,
        "ranks": 2,
        "viscosity": 0.01,
        "bcs": walls(2, [1, 2, 4]) + ["BC = U NS 3 1. 1.", "BC = V NS 3 0."],
        "solver": ["Newton line search type          = BACKTRACK"],
    },
    {
        # Plane Poiseuille flow driven by a body force; V = 0 on the ends removes
        # their shear traction, so u = y (1 - y) / (2 nu) with u_max = 1.25 exactly.
        "name": "channel2d_force",
        "mesh": "channel2d.exoII",
        "dim": 2,
        "ranks": 2,
        "viscosity": 0.1,
        "source": "1. 0. 0.",
        "bcs": walls(2, [1, 5]) + walls(2, [2, 4], ["V"]),
        "exact": {"VX": (0.0, 1.25), "VY": (0.0, 0.0), "P": (0.0, 0.0)},
    },
    {
        # Uniform initial flow decaying between no-slip walls (backward Euler).
        "name": "channel2d_decay",
        "mesh": "channel2d.exoII",
        "dim": 2,
        "ranks": 1,
        "viscosity": 0.1,
        "bcs": walls(2, [1, 5]),
        "initialize": "VELOCITY1 0 1.",
        "transient": (0.0, 5),
    },
    {
        "name": "cube3d_cavity",
        "mesh": "cube3d.exoII",
        "dim": 3,
        "ranks": 1,
        "viscosity": 0.1,
        "bcs": walls(3, [1]) + ["BC = U NS 3 1.", "BC = V NS 3 0.", "BC = W NS 3 0."],
    },
    {
        # Duct spin-up under a body force on curved hexahedra (theta = 0.5).
        "name": "cube3d_distorted_force",
        "mesh": "cube3d_distorted.exoII",
        "dim": 3,
        "ranks": 2,
        "viscosity": 0.1,
        "source": "1. 0. 0.",
        "bcs": walls(3, [5]),
        "transient": (0.5, 3),
    },
    {
        # Non-unit, per-component term multipliers, with y advection switched off,
        # a negated continuity row, and non-unit density (backward Euler, lid and
        # body force).
        "name": "cavity2d_multipliers",
        "mesh": "cavity2d_distorted.exoII",
        "dim": 2,
        "ranks": 2,
        "density": 1.5,
        "viscosity": 0.01,
        "source": "0.3 -0.2 0.",
        "bcs": walls(2, [1, 2, 4]) + ["BC = U NS 3 1.", "BC = V NS 3 0."],
        "momentum": ["2. 0.5 1. 3. 0.25 0.", "0.5 0. 1. 0.75 4. 0."],
        "continuity": "-2. 0.",
        "transient": (0.0, 3),
    },
]


def deck(case, backend):
    dim = case["dim"]
    momentum = case.get("momentum", ["1. 1. 1. 1. 1. 0."] * dim)
    equations = [
        f"EQ = momentum{a + 1}  Q2 U{a + 1} Q2  {momentum[a]}" for a in range(dim)
    ]
    equations.append(f"EQ = continuity Q1 P  Q1  {case.get('continuity', '1. 0.')}")
    if "transient" in case:
        theta, steps = case["transient"]
        time = [
            "Time integration                 = transient",
            "delta_t                          = -0.05",
            f"Maximum number of time steps     = {steps}",
            "Maximum time                     = 10.",
            "Minimum time step                = 1e-9",
            f"Time step parameter              = {theta}",
            "Time step error                  = 0.01 1 1 1 0 0 0",
            "Printing Frequency               = 1",
        ]
    else:
        time = ["Time integration                 = steady"]
    general = ["Number of processors             = 1", "Initial Guess = zero"]
    if "initialize" in case:
        general.append(f"Initialize = {case['initialize']}")
    lines = [
        "FEM File Specifications",
        "FEM file                         = mesh.exoII",
        "Output EXODUS II file            = out.exoII",
        "GUESS file                       = contin.dat",
        "SOLN file                        = soln.dat",
        "Write intermediate results       = no",
        "General Specifications",
        *general,
        "Time Integration Specifications",
        *time,
        "Solver Specifications",
        "Solution Algorithm               = stratimikos",
        "Matrix storage format            = tpetra",
        "Number of Newton Iterations      = 12",
        "Newton correction factor         = 1",
        "Normalized Residual Tolerance    = 1e-11",
        "Residual Ratio Tolerance         = 1e-6",
        "Pressure Stabilization           = no",
        f"Assembly Backend                 = {backend}",
        *case.get("solver", []),
        "Boundary Condition Specifications",
        "Number of BC = -1",
        *case["bcs"],
        "BC = P NS 100 0.",
        "END OF BC",
        "Problem Description",
        "Number of Materials = 1",
        "MAT = fluid 1",
        "  Coordinate System = CARTESIAN",
        "  Element Mapping   = isoparametric",
        "  Mesh Motion       = ARBITRARY",
        "  Number of bulk species = 0",
        f"Number of EQ = {dim + 1}",
        *equations,
        "Post Processing Specifications",
        "Velocity Divergence = yes",
        "Stream Function = no",
    ]
    return "\n".join(lines) + "\n"


def field_range(exodiff, path, name):
    """Minimum and maximum of a nodal variable over all time steps."""
    summary = subprocess.run(
        [exodiff, "-summary", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    ).stdout
    match = re.search(
        rf"^\s*{name}\s+# min:\s+(\S+) .*max:\s+(\S+) ", summary, re.MULTILINE
    )
    return (float(match.group(1)), float(match.group(2))) if match else None


NEWTON = re.compile(r"\[(\d+)\]\s+(\S+)\s+(\S+)\s+(\S+)")


def newton_history(log):
    """Iteration numbers and L_2 residuals of every Newton line, per solve."""
    history = []
    for line in log.read_text().splitlines():
        match = NEWTON.search(line)
        if match and re.fullmatch(r"\d\.\de[+-]\d\d", match.group(4)):
            history.append((int(match.group(1)), float(match.group(4))))
    return history


def solution_norms(log):
    return [
        line for line in log.read_text().splitlines() if "scaled solution norms" in line
    ]


def compare_history(kokkos, legacy):
    if [i for i, _ in kokkos] != [i for i, _ in legacy] or not kokkos:
        return "Newton iteration counts differ"
    for (_, a), (_, b) in zip(kokkos, legacy):
        # Printed to two digits; residuals near round-off are not reproducible.
        if max(a, b) > 1e-9 and abs(a - b) > 0.06 * max(a, b):
            return f"Newton residual history differs ({a:.1e} vs {b:.1e})"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--exodiff", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True)
    args = parser.parse_args()
    meshes = Path(__file__).resolve().parent / "kokkos_parity"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [args.mpiexec, "-n", "2", str(args.driver.resolve())]
    runs = []
    for case in CASES:
        for backend in ("kokkos", "legacy"):
            directory = output / f"{case['name']}_{backend}"
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir()
            shutil.copy(meshes / case["mesh"], directory / "mesh.exoII")
            (directory / "input").write_text(deck(case, backend))
            (directory / "fluid.mat").write_text(
                MATERIAL.format(
                    density=case.get("density", 1.0),
                    viscosity=case["viscosity"],
                    source=case.get("source", "0. 0. 0."),
                )
            )
            (directory / "stratimikos.xml").write_text(STRATIMIKOS)
            if runs:
                command.append("--")
            command += [
                str(directory),
                f"--run-ranks=0:{case['ranks']}",
                f"--log={directory / 'log.txt'}",
                "-i",
                "input",
                "-a",
            ]
            runs.append(directory)
    environment = dict(os.environ, GOMA_KOKKOS_PARITY="1")
    result = subprocess.run(
        command,
        cwd=output,
        env=environment,
        capture_output=True,
        text=True,
        timeout=560,
        check=False,
    )
    (output / "stdout.txt").write_text(result.stdout)
    (output / "stderr.txt").write_text(result.stderr)
    failures = []
    if result.returncode != 0:
        failures.append(f"goma_lib_sequence exited with {result.returncode}")
    summaries = [
        line for line in result.stderr.splitlines() if line.startswith("Kokkos parity")
    ]
    if (
        any(not line.endswith("PASS") for line in summaries)
        or "parity failed" in result.stderr
    ):
        failures.append("a fill-level parity comparison failed")
    if not any(line.startswith("Kokkos parity R:") for line in summaries):
        failures.append("no residual-only fill was compared")
    contexts = result.stderr.count("Kokkos assembly context configured")
    if contexts != len(CASES):
        failures.append(
            f"{contexts} Kokkos contexts configured for {len(CASES)} Kokkos solves"
        )
    parity = output / "parity.exodiff"
    parity.write_text(PARITY)
    for case in CASES:
        kokkos = output / f"{case['name']}_kokkos"
        legacy = output / f"{case['name']}_legacy"
        compare = subprocess.run(
            [
                args.exodiff,
                "-f",
                str(parity),
                str(legacy / "out.exoII"),
                str(kokkos / "out.exoII"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        (output / f"exodiff_{case['name']}.txt").write_text(
            compare.stdout + compare.stderr
        )
        if compare.returncode != 0:
            failures.append(f"{case['name']}: Kokkos and legacy solutions differ")
        for name, (low, high) in case.get("exact", {}).items():
            extent = field_range(args.exodiff, kokkos / "out.exoII", name)
            if extent is None or any(
                abs(a - b) > 1e-9 for a, b in zip(extent, (low, high))
            ):
                failures.append(
                    f"{case['name']}: {name} range {extent}, expected {(low, high)}"
                )
        try:
            problem = compare_history(
                newton_history(kokkos / "log.txt"), newton_history(legacy / "log.txt")
            )
        except OSError as error:
            problem = str(error)
            if not problem and solution_norms(kokkos / "log.txt") != solution_norms(
                legacy / "log.txt"
            ):
                problem = "printed solution norms differ"
        if problem:
            failures.append(f"{case['name']}: {problem}")
    if failures:
        print(result.stderr, file=sys.stderr)
        print("; ".join(failures), file=sys.stderr)
        return 1
    print(
        f"{len(summaries)} fill comparisons and {len(CASES)} solution comparisons passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
