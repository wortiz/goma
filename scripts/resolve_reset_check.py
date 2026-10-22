#!/usr/bin/env python3
"""Map goma reset-check output to symbol names.

goma built with -DGOMA_CHECK_GLOBAL_RESET writes lines of the form
"<object> <offset> <length>" for writable data that differs from its load time
value after goma_destroy.  This script resolves each range to the global or
static variables it covers using nm.

usage: resolve_reset_check.py --exe build/goma reset_check.txt
"""

import argparse
import bisect
import collections
import glob
import os
import subprocess
import sys

# State that is expected to differ: library-level flags in goma_api.c, the
# error handling mode, and the reset checker's own tables
EXPECTED = {
    "goma_library_initialized",
    "goma_context_used",
    "goma_petsc_preinitialized",
    "Goma_Owns_MPI",
    "segments",
    "num_segments",
    # GOMA_WH warn-once state, reset through goma_wh_generation
    "printed_generation",
    "goma_wh_generation",
    # scratch statics that are reinitialized before every use
    "CA_id",
    "CA_fselem",
    "CA_sselem",
    "CA_proc",
    "mm_fill_start",
    "mm_fill_end",
    "jacCol",
    "sr",
    "strcpy_rtn",
    "mpistringbuffer",
    # cached system constants (md_timer.c)
    "tm",
    "tickPerSecond",
    "secondsPerTick",
    # given their operator and vectors on every solve
    "Problem",
    "ls_Problem",
    # std::string statics, reset with clear() but their inline buffer keeps old bytes
    "Pkg_Name",
    "ls_Pkg_Name",
    "adapt_base_name",
    # filled before every use (mm_input.c)
    "System_Command",
    # print-once flags in the user_*.c templates
    "warning",
    # Fortran common block of accumulated eigensolver timers
    "timing_",
    # contact angle normals computed before use within a fill (bc_curve.c, bc_special.c)
    "fsnormal",
    "dfsnormal_dx",
    "ssnormal",
    "dssnormal_dx",
    "clnormal",
    "dclnormal_dx",
    "fsnrml",
    "dfsnrml_dx",
    "ssnrml",
    "dssnrml_dx",
}

# Files whose function statics are reverse-communication state, initialized at
# the start of every eigensolve
EXPECTED_STATIC_FILES = {
    "sl_eggroll01.c.o",
    "sl_eggroll02.c.o",
    "loca_eigen_cayley.F.o",
}

# (file, static) pairs for statics whose names are too generic for EXPECTED
EXPECTED_FILE_STATICS = {
    ("ac_particles.c.o", "result"),  # string buffer returned by a formatting routine
    ("dp_utils.c.o", "request"),  # MPI_Isend handle in print_sync_start, set on every call
}


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def load_symbols(path):
    """Return sorted (address, size, type, name) for data/bss symbols.

    Symbols copied into the executable from shared libraries (R_*_COPY
    relocations, e.g. MPI globals) are not Goma state and are left out.
    """
    copied = set()
    for line in run(["readelf", "-rW", path]).splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[2].endswith("_COPY"):
            copied.add(parts[4].split("@")[0])
    EXPECTED.update(copied)
    symbols = []
    for line in run(["nm", "-S", "-n", "--defined-only", path]).splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in "DdBbVv":
            symbols.append((int(parts[0], 16), int(parts[1], 16), parts[2], parts[3]))
    names = [s[3] for s in symbols]
    DEMANGLED.update(zip(names, demangle(names)))
    return symbols


def load_static_files(exe):
    """Map local (static) symbol names to object files using the build's libraries."""
    objects = collections.defaultdict(set)
    build = os.path.dirname(os.path.dirname(os.path.abspath(exe)))
    for d in (os.path.dirname(os.path.abspath(exe)), build):
        for lib in glob.glob(os.path.join(d, "lib*.a")):
            for line in run(["nm", "-A", "--defined-only", lib]).splitlines():
                parts = line.split()
                if len(parts) == 3 and parts[1] in "db":
                    objects[parts[2]].add(parts[0].split(":")[1])
    return objects


DEMANGLED = {}


def is_expected(name, static_files):
    base = DEMANGLED.get(name, name).split("@")[0].split(".")[0]
    if base in EXPECTED or name.split("@")[0] in EXPECTED:
        return True
    files = static_files.get(name)
    if not files:
        return False
    return files <= EXPECTED_STATIC_FILES or all((f, base) in EXPECTED_FILE_STATICS for f in files)


def load_data_sections(path):
    """Return [(start, end)] of the .data and .bss sections."""
    sections = []
    for line in run(["readelf", "-SW", path]).splitlines():
        parts = line.replace("[ ", "[").split()
        if len(parts) > 5 and parts[1] in (".data", ".bss"):
            start = int(parts[3], 16)
            sections.append((start, start + int(parts[5], 16)))
    return sections


def demangle(names):
    try:
        out = subprocess.run(
            ["c++filt"], input="\n".join(names), check=True, capture_output=True, text=True
        ).stdout
        return out.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", required=True, help="path of the executable (for /proc/self/exe)")
    parser.add_argument("report", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    args = parser.parse_args()

    tables = {}
    static_files = load_static_files(args.exe)
    found = collections.OrderedDict()
    unresolved = 0
    for line in args.report:
        parts = line.split()
        if len(parts) != 3 or not parts[1].startswith("0x"):
            continue
        obj = args.exe if parts[0] == "/proc/self/exe" else parts[0]
        start = int(parts[1], 16)
        end = start + int(parts[2])
        if obj not in tables:
            symbols = load_symbols(obj)
            tables[obj] = (symbols, [s[0] for s in symbols], load_data_sections(obj))
        symbols, addresses, sections = tables[obj]
        # loader state (.got.plt, .dynamic, ...) is not Goma state
        if not any(s <= start < e for s, e in sections):
            continue
        i = max(bisect.bisect_right(addresses, start) - 1, 0)
        hit = False
        while i < len(symbols) and symbols[i][0] < end:
            addr, size, _, name = symbols[i]
            if addr + max(size, 1) > start and not is_expected(name, static_files):
                found[name] = found.get(name, 0) + 1
                hit = True
            i += 1
        if not hit and not any(
            is_expected(s[3], static_files)
            for s in symbols
            if s[0] <= start < s[0] + max(s[1], 1)
        ):
            unresolved += 1

    names = list(found)
    for name, pretty in sorted(zip(names, demangle(names)), key=lambda p: p[1]):
        print(pretty)
    print(f"# {len(found)} symbols differ, {unresolved} unresolved ranges", file=sys.stderr)


if __name__ == "__main__":
    main()
