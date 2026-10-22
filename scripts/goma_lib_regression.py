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

If goma was built with -DGOMA_CHECK_GLOBAL_RESET=ON, global variables that
were not restored by goma_destroy are reported for each run.

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
    parser.add_argument("--chain-length", type=int, default=8, help="problems per chain")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed (chain mode)")
    parser.add_argument("-p", "--problem", action="append", default=[], help="problem name")
    parser.add_argument("-f", "--feature", action="append", default=[], help="GUTS feature")
    parser.add_argument("-t", "--max-time", type=int, help="skip problems estimated over N minutes")
    parser.add_argument("--serial", action="store_true", help="only single processor problems")
    parser.add_argument("--parallel", action="store_true", help="only multiprocessor problems")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per job")
    parser.add_argument("--json", help="write results to this file")
    return parser.parse_args()


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

    env = dict(guts_cfg.subprocess_env(), GOMA_EXPERIMENTAL_MULTIPLE_RUNS="1")
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
        driver_args += ["-i", p.run.goma_input] + p.run.goma_opts.split()
        if p.run.use_aprepro:
            driver_args += ["-a"] + p.run.aprepro_flags.split()
        driver_args.append("--")
        run_dirs.append(d)
    driver_args.pop()

    cmd = [str(Path(args.driver).resolve())] + driver_args
    if np > 1:
        cmd = [guts_cfg.mpirun, "-np", str(np), "--bind-to", "none"] + cmd
    with open(work / "stdout.txt", "w") as out, open(work / "stderr.txt", "w") as err:
        try:
            exit_code = subprocess.run(
                cmd, cwd=work, env=env, stdout=out, stderr=err, timeout=args.timeout
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
        code, log = check_exodiff(p, d, guts_cfg)
        (d / "exoComparison.txt").write_text(log)
        results.append(
            {
                "problem": p.name,
                "dir": str(d),
                "exodiff": code.value,
                "symbols": reset_check_symbols(args.driver, d),
            }
        )
    return {"job": name, "exit": exit_code, "runs": results}


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
        r["exodiff"] in ("OK", "n/a") and not r["symbols"] for r in result["runs"]
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
                    print("  %s exodiff %s symbols %s" % (Path(r["dir"]).name, r["exodiff"], r["symbols"]))

    results.sort(key=lambda r: r["job"])
    failed = [r["job"] for r in results if not job_passed(r)]
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"results": results, "skipped": skipped}, f, indent=2)
    print("%d of %d jobs passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("failed: %s" % " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
