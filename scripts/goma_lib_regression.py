#!/usr/bin/env python3
"""Run GUTS problems several times in one process through the goma.h library interface.

Each job runs goma_lib_sequence (tests/library) once, which runs Goma for a
list of problems one after another with goma_create ... goma_destroy.  Every
run gets its own copy of the problem directory and its output is compared with
the blessed results the same way GUTS does.

Modes:
  repeat  run each problem --runs times in one process (default)
  chain   shuffle the problems into chains of --chain-length different
          problems, each chain in one process (problems with the same number
          of processors are chained together)

--partial adds two runs before the full runs of each problem that stop after
goma_create and after goma_setup, to test teardown of a partly set up problem.

--advance-steps solves with goma_advance_steps() and --advance-to with
goma_advance_to() at the given simulation time interval instead of
goma_solve(), the output must still match the blessed results.  Problems that
cannot be advanced in steps are solved with goma_solve() and reported as
unsupported.  With --check-fields, the fields returned by goma_get_field() and
goma_get_post_field() after each step must equal the output file's.

--import-fields replaces the file of each External Field card that reads a
fixed field from a mesh file with IMPORT; the driver reads the field from the
file and sets it with goma_set_external_field() instead, so the output must
still match the blessed results.  With --declare-fields, it removes the card
and declares the field with goma_declare_external_field() instead.

--feedback NAME:FIELD:SCALE (with --import-fields and --advance-*) adds SCALE
times Goma's FIELD to the imported field NAME before each call, a coupling loop
whose results differ from the blessed ones; --compare-runs then compares each
run with the first run of its job exactly, to check the loop is repeatable.

--extra-ranks starts that many more MPI ranks than a problem needs, Goma runs
on a communicator of the first ranks only, to check that it does not
communicate outside the communicator it is given.  With --shift-ranks, Goma
is initialized on all the ranks instead and run i of a job uses the ranks
starting at i modulo (extra ranks + 1), so later runs use ranks that earlier
runs did not, while sharing some with them.

--declare-value NAME:VALUE (repeatable) declares an imported nodal field NAME
with goma_declare_external_field() and sets it to VALUE once before the solve;
with --advance-*, it must still be VALUE after each call, on adapted meshes
too.  The output then has the field and is not compared.

--set-card "NAME=VALUE" (repeatable) sets the value of card NAME in each
problem's input deck.  With --expect-failure, the solve must return an error,
for cards that make the solver give up, and the output is not compared.

If goma was built with -DGOMA_CHECK_GLOBAL_RESET=ON, global variables that
were not restored by goma_destroy are reported for each run.

Each run's standard output is written to stdout.txt in its directory.  When
the blessed standard output of a problem lists PETSc options, the run must
list the same options, so options left behind by an earlier run are caught.

examples:
  scripts/goma_lib_regression.py --driver build/tests/goma_lib_sequence
  scripts/goma_lib_regression.py --driver build/tests/goma_lib_sequence \\
      --mode chain --chain-length 16 -j 8
  scripts/goma_lib_regression.py --driver build/tests/goma_lib_sequence \\
      -p 2_phase_diff -p ldc
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPTS_DIR.parent
# exit status when no problem can run with this build (ctest SKIP_RETURN_CODE)
SKIPPED = 77


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="\n".join(__doc__.splitlines()[2:]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--driver", required=True, help="path of tests/library goma_lib_sequence")
    parser.add_argument(
        "--goma", help="goma executable used to check problem requirements (default: <build>/goma)"
    )
    parser.add_argument("--guts-dir", default=str(SOURCE_DIR / "guts"), help="GUTS checkout")
    parser.add_argument("--output-dir", default="goma_lib_regression", help="work directory")
    parser.add_argument("--mode", choices=("repeat", "chain"), default="repeat")
    parser.add_argument("--runs", type=int, default=2, help="runs of each problem (repeat mode)")
    parser.add_argument(
        "--partial", action="store_true", help="also run stopping after create and setup"
    )
    parser.add_argument(
        "--extra-ranks", type=int, default=0, help="idle MPI ranks outside Goma's communicator"
    )
    parser.add_argument(
        "--declare-value",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="declare an imported nodal field set to VALUE once",
    )
    parser.add_argument(
        "--set-card",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set the value of a card in each input deck",
    )
    parser.add_argument(
        "--expect-failure",
        action="store_true",
        help="the solve must return an error, the output is not compared",
    )
    parser.add_argument(
        "--shift-ranks",
        action="store_true",
        help="with --extra-ranks, shift each run's ranks by one within all the ranks",
    )
    advance = parser.add_mutually_exclusive_group()
    advance.add_argument(
        "--advance-steps", type=int, help="solve with goma_advance_steps(N) instead of goma_solve"
    )
    advance.add_argument(
        "--advance-to", type=float, help="solve with goma_advance_to() every DT of simulation time"
    )
    parser.add_argument(
        "--check-fields",
        action="store_true",
        help="with --advance-*, compare goma_get_field with the output after each call",
    )
    parser.add_argument(
        "--import-fields",
        action="store_true",
        help="set external fields read from files with goma_set_external_field instead",
    )
    parser.add_argument(
        "--declare-fields",
        action="store_true",
        help="with --import-fields, remove the External Field cards and declare the fields "
        "with goma_declare_external_field instead",
    )
    parser.add_argument(
        "--feedback",
        metavar="NAME:FIELD:SCALE",
        help="with --import-fields and --advance-*, add SCALE times Goma's FIELD to imported "
        "field NAME before each call",
    )
    parser.add_argument(
        "--compare-runs",
        action="store_true",
        help="compare each run's output with the first run's exactly instead of the blessed one",
    )
    parser.add_argument("--chain-length", type=int, default=8, help="problems per chain")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed (chain mode)")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="chain the problems in the order given with -p (chain mode)",
    )
    parser.add_argument("-p", "--problem", action="append", default=[], help="problem name")
    parser.add_argument("-f", "--feature", action="append", default=[], help="GUTS feature")
    parser.add_argument("-t", "--max-time", type=int, help="skip problems estimated over N minutes")
    parser.add_argument("--serial", action="store_true", help="only single processor problems")
    parser.add_argument("--parallel", action="store_true", help="only multiprocessor problems")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--timeout", type=int, default=3600, help="seconds per run, a job gets this times its runs"
    )
    parser.add_argument("--json", help="write results to this file")
    args = parser.parse_args()
    if args.check_fields and not (args.advance_steps or args.advance_to):
        parser.error("--check-fields needs --advance-steps or --advance-to")
    for card in args.set_card:
        if "=" not in card:
            parser.error("--set-card needs NAME=VALUE, got %s" % card)
    if args.shift_ranks and args.extra_ranks < 1:
        parser.error("--shift-ranks needs --extra-ranks")
    if args.declare_fields and not args.import_fields:
        parser.error("--declare-fields needs --import-fields")
    return args


def select_problems(args, guts_cfg):
    from pyguts_pkg.discovery import discover_problems, filter_problems
    from pyguts_pkg.requirements import check_requirements, get_goma_capabilities

    problems = discover_problems(Path(args.guts_dir) / "problems")
    names = {p.name for p in problems}
    for name in args.problem:
        if name not in names:
            sys.exit("unknown problem %s" % name)
    to_run, _ = filter_problems(problems, args.problem, args.feature, args.max_time)
    caps = get_goma_capabilities(guts_cfg.goma_exe, guts_cfg.subprocess_env())
    selected = []
    skipped = []
    for p in to_run:
        ok, failures = check_requirements(caps, p.requirements, p.run.num_processors)
        if not ok:
            skipped.append((p.name, "; ".join(failures)))
        elif p.run.use_chemkin:
            skipped.append((p.name, "needs chemkin"))
        elif args.serial and nprocs(p) > 1:
            skipped.append((p.name, "parallel"))
        elif args.parallel and nprocs(p) == 1:
            skipped.append((p.name, "serial"))
        else:
            selected.append(p)
    return selected, skipped


def nprocs(problem):
    return problem.run.num_processors if problem.run.use_mpi else 1


def make_jobs(args, problems):
    """Return [(job name, [(problem, stop after), ...])], each job runs in one process."""
    partial = ["create", "setup"] if args.partial else []
    if args.mode == "repeat":
        return [(p.name, [(p, s) for s in partial] + [(p, "solve")] * args.runs) for p in problems]
    rng = random.Random(args.seed)
    jobs = []
    by_np = {}
    for p in problems:
        by_np.setdefault(nprocs(p), []).append(p)
    for np, group in sorted(by_np.items()):
        if args.no_shuffle:
            group.sort(key=lambda p: args.problem.index(p.name) if p.name in args.problem else 0)
        else:
            rng.shuffle(group)
        for i in range(0, len(group), args.chain_length):
            chain = group[i : i + args.chain_length]
            runs = [(p, s) for p in chain[:1] for s in partial] + [(p, "solve") for p in chain]
            jobs.append(("chain_np%d_%02d" % (np, i // args.chain_length), runs))
    return jobs


def run_tool(cmd, cwd, env):
    with open(Path(cwd) / "tools.txt", "a") as log:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)


def run_job(args, guts_cfg, name, runs):
    from pyguts_pkg.checker import check_exodiff

    env = dict(guts_cfg.subprocess_env())
    if args.extra_ranks > 0:
        # the idle ranks may need more slots than there are cores (Open MPI)
        env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    work = Path(args.output_dir).resolve() / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    np = nprocs(runs[0][0])

    run_dirs = []
    driver_args = []
    for i, (p, stop) in enumerate(runs):
        d = work / ("%02d_%s" % (i, p.name if stop == "solve" else p.name + "_" + stop))
        shutil.copytree(p.source_dir, d)
        for whacked in p.extra.whacked_files:
            if (d / whacked).is_file():
                (d / whacked).unlink()
        if p.run.use_decomp and p.run.brk_exo_in:
            run_tool([guts_cfg.decomp, "-p", str(np), p.run.brk_exo_in], d, env)
        driver_args.append(str(d))
        if stop != "solve":
            driver_args.append("--stop-after=" + stop)
        elif args.advance_steps:
            driver_args.append("--advance-steps=%d" % args.advance_steps)
        elif args.advance_to:
            driver_args.append("--advance-to=%.17g" % args.advance_to)
        if args.set_card:
            set_cards(d / p.run.goma_input, args.set_card)
        for declared in args.declare_value:
            driver_args.append("--declare=%s:Q1" % declared.split(":")[0])
            driver_args.append("--import-value=" + declared)
        if stop == "solve" and args.expect_failure:
            driver_args.append("--expect-error")
        if args.shift_ranks:
            driver_args.append("--run-ranks=%d:%d" % (i % (args.extra_ranks + 1), np))
        if stop == "solve" and args.feedback:
            driver_args.append("--feedback=" + args.feedback)
        if args.import_fields:
            for field, interpolation, mesh_file in import_external_fields(
                d / p.run.goma_input, args.declare_fields
            ):
                driver_args.append("--import=%s:%s" % (field, mesh_file))
                if args.declare_fields:
                    driver_args.append("--declare=%s:%s" % (field, interpolation))
        if stop == "solve" and args.check_fields and p.blessed.exo_output:
            driver_args.append("--check-fields=" + p.blessed.exo_output[0])
        driver_args.append("--log=" + str(d / "stdout.txt"))
        driver_args += ["-i", p.run.goma_input] + p.run.goma_opts.split()
        if p.run.use_aprepro:
            driver_args += ["-a"] + p.run.aprepro_flags.split()
        driver_args.append("--")
        run_dirs.append(d)
    driver_args.pop()

    cmd = [str(Path(args.driver).resolve())] + driver_args
    if args.extra_ranks > 0 and not args.shift_ranks:
        cmd = [cmd[0], "--ranks=%d" % np] + cmd[1:]
    if np > 1 or args.extra_ranks > 0:
        cmd = [guts_cfg.mpirun, "-np", str(np + args.extra_ranks), "--bind-to", "none"] + cmd
    with open(work / "stdout.txt", "w") as out, open(work / "stderr.txt", "w") as err:
        try:
            exit_code = subprocess.run(
                cmd, cwd=work, env=env, stdout=out, stderr=err, timeout=args.timeout * len(runs)
            ).returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout"

    results = []
    for (p, stop), d in zip(runs, run_dirs):
        if stop != "solve":
            results.append(
                {
                    "problem": p.name,
                    "dir": str(d),
                    "exodiff": "n/a",
                    "symbols": reset_check_symbols(args.driver, d),
                }
            )
            continue
        if p.run.use_decomp and p.run.fix_exo_basename:
            run_tool(
                [guts_cfg.epu, "-auto", "%s.exoII.%d.0" % (p.run.fix_exo_basename, np)], d, env
            )
        if args.expect_failure or args.declare_value:
            # the solver gave up or the output has the declared fields, it is not the blessed one
            exodiff = "n/a"
        elif args.compare_runs:
            exodiff = compare_with_first_run(guts_cfg, p, d, run_dirs[0], env)
        else:
            code, log = check_exodiff(p, d, guts_cfg)
            (d / "exoComparison.txt").write_text(log)
            exodiff = code.value
        results.append(
            {
                "problem": p.name,
                "dir": str(d),
                "exodiff": exodiff,
                "petsc_options": "n/a" if args.expect_failure else check_petsc_options(p, d),
                "symbols": reset_check_symbols(args.driver, d),
                "advance": advance_status(args, d),
            }
        )
    return {"job": name, "exit": exit_code, "runs": results}


EXTERNAL_FIELD = re.compile(r"^(\s*External Field\s*=\s*(\S+)\s+(\S+)\s+)(\S+)\s*$")


def import_external_fields(deck, remove=False):
    """Replace the mesh file of fixed External Field cards in deck with IMPORT, or remove
    the cards.

    Returns [(field name, interpolation, mesh file)]."""
    lines = deck.read_text().splitlines(keepends=True)
    imports = []
    for i, line in enumerate(lines):
        m = EXTERNAL_FIELD.match(line)
        if m and m.group(4) not in ("IMPORT", "IMPORT_EV") and m.group(4).endswith(".exoII"):
            imports.append((m.group(2), m.group(3), m.group(4)))
            lines[i] = "" if remove else m.group(1) + "IMPORT\n"
    deck.write_text("".join(lines))
    return imports


def set_cards(deck, cards):
    """Set the value of each NAME=VALUE card in deck where the deck has the card."""
    lines = deck.read_text().splitlines(keepends=True)
    for card in cards:
        name, value = (part.strip() for part in card.split("=", 1))
        pattern = re.compile(r"^\s*%s\s*=" % r"\s+".join(map(re.escape, name.split())), re.I)
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = "%s = %s\n" % (name, value)
    deck.write_text("".join(lines))


def compare_with_first_run(guts_cfg, problem, run_dir, first_dir, env):
    """Compare the output files of a run with the first run's exactly: OK, FAILED or n/a."""
    if run_dir == first_dir:
        return "n/a"
    for name in problem.blessed.exo_output:
        cmd = [
            guts_cfg.exodiff,
            "-absolute",
            "-tolerance",
            "0",
            "-coordinate_tolerance",
            "0",
            "-Floor",
            "0",
            str(first_dir / name),
            str(run_dir / name),
        ]
        with open(run_dir / "exoComparison.txt", "a") as log:
            if subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
                return "FAILED"
    return "OK"


def advance_status(args, run_dir):
    """Whether a run was solved in steps: OK, unsupported (solved by goma_solve) or n/a."""
    if not (args.advance_steps or args.advance_to):
        return "n/a"
    try:
        stdout = (run_dir / "stdout.txt").read_text(errors="replace")
    except OSError:
        return "n/a"
    return "unsupported" if "can be advanced" in stdout else "OK"


def petsc_options(stdout_file):
    """Options in the first PETSc option table Goma printed, None if there is none."""
    try:
        lines = Path(stdout_file).read_text(errors="replace").splitlines()
    except OSError:
        return None
    if "#PETSc Option Table entries:" not in lines:
        return None
    start = lines.index("#PETSc Option Table entries:") + 1
    options = []
    for line in lines[start:]:
        if line.startswith("#End of PETSc Option Table"):
            break
        # drop the "# (source: file)" annotation, it depends on how the option was set
        options.append(line.split("#")[0].strip())
    return sorted(options)


def check_petsc_options(problem, run_dir):
    """Compare the PETSc options of a run with the blessed standard output."""
    blessed = petsc_options(Path(problem.source_dir) / "blessed.stdout.txt")
    if blessed is None:
        return "n/a"
    actual = petsc_options(run_dir / "stdout.txt")
    (run_dir / "petscOptions.txt").write_text(
        "blessed: %s\nrun:     %s\n" % (" ".join(blessed), " ".join(actual or []))
    )
    return "OK" if actual == blessed else "FAILED"


def reset_check_symbols(driver, run_dir):
    """Globals the reset checker found unrestored after this run, None if it was not enabled."""
    reports = sorted(run_dir.glob("goma_reset_check.*.txt"))
    if not reports:
        return None
    symbols = set()
    for report in reports:
        out = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "resolve_reset_check.py"),
                "--exe",
                str(driver),
                str(report),
            ],
            capture_output=True,
            text=True,
        ).stdout
        symbols.update(out.split())
    return sorted(symbols)


def job_passed(result):
    return result["exit"] == 0 and all(
        r["exodiff"] in ("OK", "n/a")
        and r.get("petsc_options", "n/a") in ("OK", "n/a")
        and not r["symbols"]
        for r in result["runs"]
    )


def main():
    args = parse_args()
    # the GUTS package finds, filters and checks the problems
    sys.path.insert(0, str(Path(args.guts_dir) / "scripts"))
    from pyguts_pkg.config import load_config

    goma = args.goma or str(Path(args.driver).resolve().parent.parent / "goma")
    guts_cfg = load_config(None, goma, None, None, str(Path(args.guts_dir) / "scripts"))
    problems, skipped = select_problems(args, guts_cfg)
    jobs = make_jobs(args, problems)
    print("%d problems, %d skipped, %d jobs" % (len(problems), len(skipped), len(jobs)), flush=True)
    for name, reason in skipped:
        print("skipped %s: %s" % (name, reason))
    if not jobs:
        return SKIPPED

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_job, args, guts_cfg, name, runs) for name, runs in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = "PASSED" if job_passed(result) else "FAILED"
            print("%s %s" % (status, result["job"]), flush=True)
            if status == "FAILED":
                print("  exit %s" % result["exit"])
                for r in result["runs"]:
                    print(
                        "  %s exodiff %s petsc options %s symbols %s"
                        % (
                            Path(r["dir"]).name,
                            r["exodiff"],
                            r.get("petsc_options", "n/a"),
                            r["symbols"],
                        )
                    )

    results.sort(key=lambda r: r["job"])
    failed = [r["job"] for r in results if not job_passed(r)]
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"results": results, "skipped": skipped}, f, indent=2)
    print("%d of %d jobs passed" % (len(results) - len(failed), len(results)))
    unsupported = sorted(
        {
            Path(r["dir"]).name
            for j in results
            for r in j["runs"]
            if r.get("advance") == "unsupported"
        }
    )
    if unsupported:
        print("solved with goma_solve, cannot be advanced: %s" % " ".join(unsupported))
    if failed:
        print("failed: %s" % " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
