"""Reject a Kokkos setup collectively, then solve a legacy problem in the same process."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--problem-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rejected = output / "rejected"
    recovered = output / "recovered"
    for directory in (rejected, recovered):
        if directory.exists():
            shutil.rmtree(directory)
        shutil.copytree(args.problem_dir, directory)
    for directory, backend, autodiff in (
        (rejected, "kokkos", "yes"),
        (recovered, "legacy", "no"),
    ):
        deck = directory / "input"
        lines = deck.read_text().splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.strip().startswith("Pressure Stabilization Scaling"):
                lines[index + 1 : index + 1] = [
                    f"Assembly Backend = {backend}\n",
                    f"Use AutoDiff Assembly = {autodiff}\n",
                ]
                break
        else:
            raise RuntimeError("Test deck has no expected solver-card insertion point")
        deck.write_text("".join(lines))
    command = [
        args.mpiexec,
        "-n",
        "2",
        str(args.driver.resolve()),
        str(rejected),
        "--expect-setup-error",
        "-i",
        "input",
        "-a",
        "--",
        str(recovered),
        "-i",
        "input",
        "-a",
    ]
    result = subprocess.run(
        command, cwd=output, capture_output=True, text=True, timeout=110, check=False
    )
    (output / "stdout.txt").write_text(result.stdout)
    (output / "stderr.txt").write_text(result.stderr)
    required = [
        "Kokkos assembly preflight rejected this configuration:",
        "Jacobian: Use AutoDiff Assembly",
        "coordinates must be Cartesian",
        "unsupported equation",
        "stabilization: PSPG",
        "must be tpetra",
        "only one homogeneous fluid",
        "explicit nodal P",
    ]
    missing = [message for message in required if message not in result.stderr]
    # Only real incompatibilities are reported; execution itself is no longer gated.
    unexpected = (
        "execution:" in result.stderr or "accepted this configuration" in result.stderr
    )
    if (
        result.returncode != 0
        or missing
        or unexpected
        or result.stderr.count(required[0]) != 1
    ):
        print(result.stderr, file=sys.stderr)
        print(
            f"exit={result.returncode}; missing diagnostics={missing}; unexpected={unexpected}",
            file=sys.stderr,
        )
        return 1
    print(
        "Kokkos setup rejected on both ranks; solve/retry blocked; legacy recovery passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
