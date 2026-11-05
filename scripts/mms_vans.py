#!/usr/bin/env python3
"""
Method of Manufactured Solutions (MMS) forcing terms for the momentum
equations of the volume-averaged Navier-Stokes (VANS) equations, Model A and
Model B (see "DEM Project.md").

Momentum, written as  L(u, p, eps) = f :

  Model A:  f_A = rho*eps*(du/dt + (u.grad)u) - div(eps*T) - rho*eps*g + F_pf
  Model B:  f_B = rho*eps*(du/dt + (u.grad)u) - div(T)     - rho*eps*g + F_pf

  T = -p I + mu (grad u + grad u^T) - (2/3) mu (div u) I

f is the source you add to the right-hand side of the momentum equation in
the solver so that the manufactured (u, p, eps) is an exact solution.
By default g = 0 and F_pf = 0 (pass --g / --Fpf to include them).

Conventions
  (grad u)_ij = d u_j / d x_i      ((u.grad)u)_j = u_i d_i u_j
  (div T)_j   = d_i T_ij           (T is symmetric, so the index choice
                                    does not matter)

Usage (command line)
  python mms_vans.py --u "sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(pi*y)" \
                     --p "cos(pi*x)*cos(pi*y)" \
                     --eps "1/2 + 1/4*sin(pi*x)*sin(pi*y)"

  The number of components in --u sets the dimension (2 -> x,y; 3 -> x,y,z).
  Expressions may use x, y, z, t, rho, mu, and anything sympy understands.

Usage (from Python / a notebook)
  from mms_vans import mms_forcing
  res = mms_forcing("sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(pi*y)",
                    "cos(pi*x)*cos(pi*y)", "1/2 + 1/4*sin(pi*x)*sin(pi*y)")
  res["A"], res["B"]   # sympy Matrix column vectors
  res["mass_residual"] # S_mass = d(eps)/dt + div(eps u)  (0 if u is compatible)
"""

import argparse

import sympy as sp

x, y, z, t = sp.symbols("x y z t", real=True)
rho, mu = sp.symbols("rho mu", positive=True)
COORDS = (x, y, z)
NAMESPACE = {"x": x, "y": y, "z": z, "t": t, "rho": rho, "mu": mu}


def _parse_scalar(s):
    if not isinstance(s, str):
        return sp.sympify(s)
    return sp.parse_expr(s, local_dict=NAMESPACE,
                         transformations=sp.parsing.sympy_parser.standard_transformations
                         + (sp.parsing.sympy_parser.convert_xor,))


def _parse_vector(v):
    if isinstance(v, str):
        parts = [s for s in v.split(",") if s.strip()]
    else:
        parts = list(v)
    comps = [_parse_scalar(s) for s in parts]
    if len(comps) not in (2, 3):
        raise ValueError(f"u must have 2 or 3 components, got {len(comps)}")
    return sp.Matrix(comps)


def grad_scalar(f, X):
    return sp.Matrix([sp.diff(f, xi) for xi in X])


def grad_vector(u, X):
    """(grad u)_ij = d u_j / d x_i"""
    n = len(X)
    return sp.Matrix(n, n, lambda i, j: sp.diff(u[j], X[i]))


def div_vector(u, X):
    return sum(sp.diff(u[i], X[i]) for i in range(len(X)))


def div_tensor(T, X):
    """(div T)_j = d_i T_ij"""
    n = len(X)
    return sp.Matrix([sum(sp.diff(T[i, j], X[i]) for i in range(n)) for j in range(n)])


def stress(u, p, X):
    n = len(X)
    I = sp.eye(n)
    G = grad_vector(u, X)
    return -p * I + mu * (G + G.T) - sp.Rational(2, 3) * mu * div_vector(u, X) * I


def mms_forcing(u, p, eps, g=None, Fpf=None, simplify=True):
    """Return dict with forcing vectors for Model A and Model B.

    u   : str "ux, uy[, uz]" or sequence of str/sympy exprs
    p   : str or sympy expr
    eps : str or sympy expr
    g   : optional gravity vector (same format as u); default 0
    Fpf : optional particle-fluid force vector (same format as u); default 0
    """
    u = _parse_vector(u)
    n = len(u)
    X = COORDS[:n]
    p = _parse_scalar(p)
    eps = _parse_scalar(eps)
    g = sp.zeros(n, 1) if g is None else _parse_vector(g)
    Fpf = sp.zeros(n, 1) if Fpf is None else _parse_vector(Fpf)
    if len(g) != n or len(Fpf) != n:
        raise ValueError("g and Fpf must have the same dimension as u")

    G = grad_vector(u, X)
    dudt = sp.diff(u, t)
    convection = G.T * u                       # (u.grad)u, j-th = u_i d_i u_j
    inertia = rho * eps * (dudt + convection)

    T = stress(u, p, X)
    divT_A = div_tensor(eps * T, X)            # div(eps T)
    divT_B = div_tensor(T, X)                  # div(T)

    f_A = inertia - divT_A - rho * eps * g + Fpf
    f_B = inertia - divT_B - rho * eps * g + Fpf

    # Mass-conservation residual (useful sanity check: is u compatible with eps?)
    # Standard VANS continuity: d(eps)/dt + div(eps u) = 0
    mass = sp.diff(eps, t) + div_vector(eps * u, X)

    if simplify:
        f_A = f_A.applyfunc(sp.simplify)
        f_B = f_B.applyfunc(sp.simplify)
        mass = sp.simplify(mass)

    return {"A": f_A, "B": f_B, "mass_residual": mass,
            "u": u, "p": p, "eps": eps, "coords": X}


def _print_vector(name, f, X, fmt):
    labels = "xyz"
    for i in range(len(f)):
        expr = f[i]
        if fmt == "c":
            s = sp.ccode(expr)
        elif fmt == "latex":
            s = sp.latex(expr)
        elif fmt == "python":
            s = sp.pycode(expr)
        else:
            s = sp.sstr(expr)
        print(f"{name}_{labels[i]} = {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--u", required=True, help='velocity, e.g. "ux, uy" or "ux, uy, uz"')
    ap.add_argument("--p", required=True, help="pressure")
    ap.add_argument("--eps", required=True, help="fluid volume fraction epsilon_f")
    ap.add_argument("--g", default=None, help="gravity vector (default 0)")
    ap.add_argument("--Fpf", default=None, help="particle-fluid force vector (default 0)")
    ap.add_argument("--model", choices=["A", "B", "both"], default="both")
    ap.add_argument("--format", choices=["sympy", "c", "python", "latex"], default="sympy",
                    help="output format of the forcing expressions")
    ap.add_argument("--no-simplify", action="store_true", help="skip sympy.simplify (faster)")
    args = ap.parse_args()

    res = mms_forcing(args.u, args.p, args.eps, g=args.g, Fpf=args.Fpf,
                      simplify=not args.no_simplify)
    X = res["coords"]

    print(f"# dimension: {len(X)}  coords: {X}")
    print(f"# u   = {list(res['u'])}")
    print(f"# p   = {res['p']}")
    print(f"# eps = {res['eps']}")
    print()
    print("# Conservation of mass forcing:  d(eps)/dt + div(eps u) = S_mass")
    S = res["mass_residual"]
    if args.format == "c":
        s = sp.ccode(S)
    elif args.format == "latex":
        s = sp.latex(S)
    elif args.format == "python":
        s = sp.pycode(S)
    else:
        s = sp.sstr(S)
    print(f"S_mass = {s}")
    if S == 0:
        print("# OK: 0 == 0, manufactured u satisfies VANS mass conservation")
    else:
        print("# WARNING: S_mass != 0; the manufactured u does not satisfy VANS mass")
        print("#          conservation, so a mass source term is also required.")
    print()
    for m in (["A", "B"] if args.model == "both" else [args.model]):
        print(f"# Model {m} momentum forcing")
        _print_vector(f"f{m}", res[m], X, args.format)
        print()


if __name__ == "__main__":
    main()
