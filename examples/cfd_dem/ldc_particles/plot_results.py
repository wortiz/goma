#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "matplotlib", "netCDF4", "imageio-ffmpeg"]
# ///
"""
Plot Goma + LAMMPS results of the CFD-DEM lid driven cavity.

Reads the Goma EXODUS II output (with netCDF4, no SEACAS needed) and the LAMMPS dump
(`dump custom ... id x y vx vy radius`, written with `dump_modify time yes`) and draws a
fluid field, streamlines and the particles at their true size.

Examples
  python3 plot_results.py                          # last time -> snapshot.png
  python3 plot_results.py --time 1 5 10            # one PNG per requested time
  python3 plot_results.py --field VOID_FRACTION    # any nodal variable, or vmag
  python3 plot_results.py --animate cavity.gif     # .gif (Pillow) or .mp4 (ffmpeg)
  python3 plot_results.py --history history.png    # particle speed / void fraction vs time
  python3 plot_results.py --list                   # show available nodal variables

Dependencies: numpy, matplotlib, netCDF4, and for .mp4 an ffmpeg on PATH or imageio-ffmpeg
(`pip install numpy matplotlib netCDF4 imageio-ffmpeg`, or `uv run plot_results.py ...`
which reads the header above).
"""
import argparse
import os
import sys

import numpy as np

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import EllipseCollection, LineCollection  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

FIELD_LABELS = {
    "vmag": "|u|",
    "VOID_FRACTION": "void fraction ε",
    "PRESSURE": "pressure",
    "VX": "u_x",
    "VY": "u_y",
    "STREAM": "stream function",
}

# --------------------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------------------


def _chars(arr):
    """netCDF char array -> list of python strings."""
    out = []
    for row in np.atleast_2d(arr):
        b = b"".join(c for c in np.ma.filled(row, b"") if c not in (b"", b"\x00"))
        out.append(b.decode().strip())
    return out


class ExodusResults:
    """Nodal results of a (serial) EXODUS II file."""

    def __init__(self, path):
        self.ds = Dataset(path, "r")
        v = self.ds.variables
        if "coordx" in v:
            self.x, self.y = np.array(v["coordx"][:]), np.array(v["coordy"][:])
        else:
            self.x, self.y = np.array(v["coord"][0]), np.array(v["coord"][1])
        self.times = np.array(v["time_whole"][:]) if "time_whole" in v else np.zeros(0)
        self.names = _chars(v["name_nod_var"][:]) if "name_nod_var" in v else []
        self.tri = mtri.Triangulation(self.x, self.y, self._triangles())

    def _triangles(self):
        """Split every element of every block into linear triangles using all its nodes."""
        tris = []
        for name, var in self.ds.variables.items():
            if not name.startswith("connect"):
                continue
            conn = np.array(var[:]) - 1
            etype = getattr(var, "elem_type", "").upper()
            nn = conn.shape[1]
            if etype.startswith("QUAD") and nn == 9:
                # ring of corners and midsides around the center node
                ring = [0, 4, 1, 5, 2, 6, 3, 7]
                for k in range(8):
                    tris.append(conn[:, [8, ring[k], ring[(k + 1) % 8]]])
            elif etype.startswith("QUAD"):
                tris += [conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]]
            elif etype.startswith("TRI") and nn == 6:
                tris += [conn[:, [0, 3, 5]], conn[:, [3, 1, 4]], conn[:, [5, 4, 2]],
                         conn[:, [3, 4, 5]]]
            elif etype.startswith("TRI"):
                tris.append(conn[:, :3])
            else:
                sys.exit(f"unsupported element type {etype!r} in {name}")
        return np.vstack(tris)

    def has(self, name):
        return name == "vmag" or name in self.names

    def nodal(self, name, step):
        if name == "vmag":
            return np.hypot(self.nodal("VX", step), self.nodal("VY", step))
        if name not in self.names:
            sys.exit(f"nodal variable {name!r} not in output; available: {', '.join(self.names)}")
        i = self.names.index(name)
        v = self.ds.variables
        if f"vals_nod_var{i + 1}" in v:
            return np.array(v[f"vals_nod_var{i + 1}"][step])
        return np.array(v["vals_nod_var"][step, i])

    def step_at(self, t):
        return int(np.argmin(np.abs(self.times - t)))


def read_dump(path):
    """LAMMPS text dump -> list of (time, step, dict of column arrays)."""
    frames = []
    with open(path) as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        t, step, cols, rows = None, None, None, None
        while i < n and not lines[i].startswith("ITEM: ATOMS"):
            item = lines[i].strip()
            if item == "ITEM: TIME":
                t = float(lines[i + 1])
            elif item == "ITEM: TIMESTEP":
                step = int(lines[i + 1])
            elif item == "ITEM: NUMBER OF ATOMS":
                rows = int(lines[i + 1])
            i += 1
        if i >= n:
            break
        cols = lines[i].split()[2:]
        data = np.array([lines[i + 1 + k].split() for k in range(rows)], dtype=float)
        data = data.reshape(rows, len(cols))
        i += 1 + rows
        frames.append((t, step, {c: data[:, j] for j, c in enumerate(cols)}))
    if frames and frames[0][0] is None:
        print("note: dump has no ITEM: TIME (add `dump_modify ... time yes` to in.dem); "
              "using step * --dem-dt", file=sys.stderr)
    return frames


# --------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------


class CavityPlot:
    def __init__(self, exo, frames, args):
        self.exo, self.frames, self.args = exo, frames, args
        self.ptimes = np.array([fr[0] if fr[0] is not None else fr[1] * args.dem_dt
                                for fr in frames]) if frames else np.zeros(0)
        self.fig, self.ax = plt.subplots(figsize=(7.2, 6.4), dpi=args.dpi,
                                         layout="constrained")
        self.levels = self._levels()
        self.cbar = None

    def _levels(self):
        a = self.args
        if a.vmin is not None and a.vmax is not None:
            lo, hi = a.vmin, a.vmax
        else:
            # one color scale over the whole run so animations do not flicker
            steps = range(len(self.exo.times))
            stride = max(1, len(self.exo.times) // 40)
            vals = [self.exo.nodal(a.field, s) for s in list(steps)[::stride] + [steps[-1]]]
            lo = a.vmin if a.vmin is not None else min(v.min() for v in vals)
            hi = a.vmax if a.vmax is not None else max(v.max() for v in vals)
        if hi <= lo:
            hi = lo + 1e-12
        return np.linspace(lo, hi, a.levels + 1)

    def particle_frame(self, t):
        if len(self.ptimes) == 0:
            return None
        return int(np.argmin(np.abs(self.ptimes - t)))

    def draw(self, step):
        a, ax, exo = self.args, self.ax, self.exo
        ax.clear()
        t = exo.times[step]
        cmap = a.cmap or ("viridis_r" if a.field == "VOID_FRACTION" else "viridis")
        cs = ax.tricontourf(exo.tri, exo.nodal(a.field, step), levels=self.levels, cmap=cmap,
                            extend="both")
        if self.cbar is None:
            self.cbar = self.fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02)
            self.cbar.set_label(FIELD_LABELS.get(a.field, a.field))

        if not a.no_streamlines and exo.has("STREAM"):
            psi = exo.nodal("STREAM", step)
            if np.ptp(psi) > 0:
                ax.tricontour(exo.tri, psi, levels=a.stream_levels, colors="w",
                              linewidths=0.6, alpha=0.7)

        k = self.particle_frame(t)
        if k is not None:
            if a.trails > 0:
                self._draw_trails(k)
            p = self.frames[k][2]
            d = 2.0 * p["radius"]
            ax.add_collection(EllipseCollection(
                d, d, np.zeros_like(d), units="xy", offsets=np.column_stack([p["x"], p["y"]]),
                offset_transform=ax.transData, facecolors="white", edgecolors="black",
                linewidths=0.5, zorder=3))
            lag = abs(self.ptimes[k] - t)
            ptxt = f"   particles t = {self.ptimes[k]:.3f}" if lag > 1e-6 else ""
        else:
            ptxt = ""

        ax.set_xlim(exo.x.min(), exo.x.max())
        ax.set_ylim(exo.y.min(), exo.y.max())
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{FIELD_LABELS.get(a.field, a.field)},  t = {t:.3f}{ptxt}")
        return ax

    def _draw_trails(self, k):
        a = self.args
        k0 = max(0, k - a.trails)
        if k - k0 < 1:
            return
        ids = self.frames[k][2]["id"]
        xs, ys = [], []
        for j in range(k0, k + 1):
            p = self.frames[j][2]
            order = np.argsort(p["id"])
            pos = np.searchsorted(p["id"][order], ids)
            pos = np.clip(pos, 0, len(order) - 1)
            idx = order[pos]
            ok = p["id"][idx] == ids
            xs.append(np.where(ok, p["x"][idx], np.nan))
            ys.append(np.where(ok, p["y"][idx], np.nan))
        xs, ys = np.array(xs), np.array(ys)  # [time, particle]
        segs = [np.column_stack([xs[:, i], ys[:, i]]) for i in range(xs.shape[1])]
        self.ax.add_collection(LineCollection(segs, colors="0.95", linewidths=0.6, alpha=0.6,
                                              zorder=2))


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------


def snapshots(plot, args):
    exo = plot.exo
    times = args.time if args.time else [exo.times[-1]]
    stem, ext = os.path.splitext(args.out)
    for t in times:
        step = exo.step_at(t)
        plot.draw(step)
        out = args.out if len(times) == 1 else f"{stem}_t{exo.times[step]:07.3f}{ext}"
        plot.fig.savefig(out)
        print(f"wrote {out} (t = {exo.times[step]:g})")


def animate(plot, args):
    exo = plot.exo
    steps = list(range(0, len(exo.times), max(1, args.stride)))
    if steps[-1] != len(exo.times) - 1:
        steps.append(len(exo.times) - 1)

    def update(i):
        plot.draw(steps[i])
        return []

    anim = animation.FuncAnimation(plot.fig, update, frames=len(steps), blit=False)
    if args.animate.lower().endswith(".gif"):
        writer = animation.PillowWriter(fps=args.fps)
    else:
        if not animation.writers.is_available("ffmpeg"):
            try:  # pip install imageio-ffmpeg ships an ffmpeg binary
                import imageio_ffmpeg

                plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                sys.exit("ffmpeg not found: install it, `pip install imageio-ffmpeg`, "
                         "or write a .gif instead (uses Pillow)")
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2500, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p"])
    anim.save(args.animate, writer=writer,
              progress_callback=lambda i, n: print(f"\rframe {i + 1}/{n}", end="", flush=True))
    print(f"\nwrote {args.animate} ({len(steps)} frames)")


def history(exo, frames, args):
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), dpi=args.dpi, sharex=True,
                           layout="constrained")
    if frames:
        pt = np.array([fr[0] if fr[0] is not None else fr[1] * args.dem_dt for fr in frames])
        speed = [np.hypot(fr[2]["vx"], fr[2]["vy"]) for fr in frames]
        ax[0].plot(pt, [s.mean() for s in speed], label="mean")
        ax[0].plot(pt, [s.max() for s in speed], label="max", alpha=0.7)
        ax[0].set_ylabel("particle speed")
        ax[0].legend()
    if exo.has("VOID_FRACTION"):
        eps = [exo.nodal("VOID_FRACTION", s) for s in range(len(exo.times))]
        ax[1].plot(exo.times, [e.min() for e in eps], label="min")
        ax[1].plot(exo.times, [e.mean() for e in eps], label="nodal mean")
        ax[1].set_ylabel("void fraction ε")
        ax[1].legend()
    ax[1].set_xlabel("t")
    fig.savefig(args.history)
    print(f"wrote {args.history}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exo", default="out.exoII", help="Goma output (default out.exoII)")
    ap.add_argument("--dump", default="dump.particles", help="LAMMPS dump (default dump.particles)")
    ap.add_argument("--field", default="vmag",
                    help="nodal variable to color by, or vmag for |u| (default vmag)")
    ap.add_argument("--time", type=float, nargs="+", help="snapshot time(s); default: last")
    ap.add_argument("--out", default="snapshot.png", help="snapshot file name (default snapshot.png)")
    ap.add_argument("--animate", metavar="FILE", help="write an animation (.gif or .mp4)")
    ap.add_argument("--history", metavar="FILE", help="write a time-history plot")
    ap.add_argument("--stride", type=int, default=1, help="animate every Nth Goma step")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--trails", type=int, default=0,
                    help="draw particle paths over the last N dump frames")
    ap.add_argument("--levels", type=int, default=24, help="contour levels")
    ap.add_argument("--stream-levels", type=int, default=16, help="streamline count")
    ap.add_argument("--no-streamlines", action="store_true")
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--cmap", help="matplotlib colormap (default viridis)")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--dem-dt", type=float, default=1.0e-4,
                    help="DEM step for dumps without ITEM: TIME (default 1e-4)")
    ap.add_argument("--list", action="store_true", help="list nodal variables and times, then exit")
    args = ap.parse_args()

    exo = ExodusResults(args.exo)
    if args.list:
        print("nodal variables:", ", ".join(exo.names))
        print(f"{len(exo.times)} times from {exo.times.min():g} to {exo.times.max():g}")
        return
    if len(exo.times) == 0:
        sys.exit(f"{args.exo} has no time steps")
    frames = read_dump(args.dump) if os.path.exists(args.dump) else []
    if not frames:
        print(f"note: no particle frames ({args.dump}); drawing the fluid only", file=sys.stderr)

    if args.history:
        history(exo, frames, args)
        if not args.animate and not args.time:
            return
    plot = CavityPlot(exo, frames, args)
    if args.animate:
        animate(plot, args)
    else:
        snapshots(plot, args)


if __name__ == "__main__":
    main()
