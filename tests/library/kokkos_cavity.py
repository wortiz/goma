"""Route cavity fills through Kokkos in library mode and compare with legacy assembly.

One goma_lib_sequence process on two ranks runs, in order: a transient Kokkos
solve whose context setup must fail on every rank (P on a node without a
pressure DOF; steady fill errors are fatal in Goma), steady and transient
Kokkos solves, then the same two legacy solves. Kokkos and legacy
outputs must agree within the parity tolerance.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

TRANSIENT = """Time integration                 = transient
delta_t                          = 0.05
Maximum number of time steps     = 4
Maximum time                     = 10.
Minimum time step                = 1e-9
Time step parameter              = 0.5
Time step error                  = 0.01 1 1 1 0 0 0
Printing Frequency               = 1
"""

PARITY = """COORDINATES absolute 1.e-12
TIME STEPS relative 1.e-12
NODAL VARIABLES relative 1.e-8 floor 1.e-12
"""


def write_case(source, directory, backend, transient, pressure_set):
    if directory.exists():
        shutil.rmtree(directory)
    shutil.copytree(source, directory)
    deck = directory / "input"
    text = deck.read_text()
    replacements = {
        "Assembly Backend                 = kokkos": f"Assembly Backend                 = {backend}",
        "BC = P NS 100 0.": f"BC = P NS {pressure_set} 0.",
    }
    if transient:
        steps = TRANSIENT
        if pressure_set != "100":  # Give up after a few step reductions.
            steps = steps.replace(
                "Minimum time step                = 1e-9",
                "Minimum time step                = 0.01",
            )
        replacements["Time integration                 = steady\n"] = steps
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"Cavity deck has no '{old.strip()}' card")
        text = text.replace(old, new)
    deck.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--exodiff", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "kokkos_cavity"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("rejected", "kokkos", True, "101", ["--expect-error"]),
        ("kokkos_steady", "kokkos", False, "100", []),
        ("kokkos_transient", "kokkos", True, "100", []),
        ("legacy_steady", "legacy", False, "100", []),
        ("legacy_transient", "legacy", True, "100", []),
    ]
    command = [args.mpiexec, "-n", "2", str(args.driver.resolve())]
    for index, (name, backend, transient, pressure_set, options) in enumerate(cases):
        write_case(source, output / name, backend, transient, pressure_set)
        if index > 0:
            command.append("--")
        command += [str(output / name), *options, "-i", "input", "-a"]
    environment = dict(os.environ, GOMA_KOKKOS_QUALIFICATION="1")
    result = subprocess.run(
        command,
        cwd=output,
        env=environment,
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
    )
    (output / "stdout.txt").write_text(result.stdout)
    (output / "stderr.txt").write_text(result.stderr)
    failures = []
    if result.returncode != 0:
        failures.append(f"goma_lib_sequence exited with {result.returncode}")
    rejected = (
        "Kokkos assembly failed: No rank owns a pressure DOF with a nodal P constraint"
    )
    if result.stderr.count(rejected) < 2:
        failures.append("setup failure was not reported on both ranks")
    if result.stderr.count("Kokkos assembly context configured for 2D Q2/Q1 flow") != 2:
        failures.append(
            "the steady and transient Kokkos solves did not each configure a context"
        )
    if result.stderr.count("Kokkos assembly failed") != result.stderr.count(rejected):
        failures.append("a qualified Kokkos solve reported an assembly failure")
    parity = output / "parity.exodiff"
    parity.write_text(PARITY)
    for kind in ("steady", "transient"):
        compare = subprocess.run(
            [
                args.exodiff,
                "-f",
                str(parity),
                str(output / f"legacy_{kind}" / "out.exoII"),
                str(output / f"kokkos_{kind}" / "out.exoII"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        (output / f"exodiff_{kind}.txt").write_text(compare.stdout + compare.stderr)
        if compare.returncode != 0:
            failures.append(f"{kind} Kokkos and legacy solutions differ")
    if failures:
        print(result.stderr, file=sys.stderr)
        print("; ".join(failures), file=sys.stderr)
        return 1
    print(
        "Kokkos setup failure reported without fallback; steady and transient parity passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
