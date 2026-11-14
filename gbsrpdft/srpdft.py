#!/usr/bin/env python
#
# Author: Minseok Oh <msjeff2001@snu.ac.kr>
#
'''Short-range exchange-correlation functional on the on-top pair density.

The spin polarization inside the short-range functional is taken from the
on-top pair density, zeta^2 = u = 1 - 4 Pi / rho^2 (the t translation of
MC-PDFT).  The functional is evaluated through libxc for u >= 0; for u < 0
it is continued by a polynomial in u through libxc values at u >= 0, so that
derivatives with respect to Pi are finite at u = 0.  GB-srPDFT energies use
the screened form (u < 0 set to 0).
'''

import numpy as np

from pyscf.dft import libxc

# interpolation nodes in u (Chebyshev on [0, U_NODE_MAX], plus u = 0)
U_NODE_MAX = 0.6
N_NODES = 8
U_CLAMP = -1.0          # R = 2
U_MAX = 1.0             # zeta = 1
RHO_CUTOFF = 1e-12
# relative shift applied to isolated grid points where libxc returns NaN
NUDGE = 1e-12


def eval_xc_finite(fnal, rho, spin=0, deriv=1, omega=None, on_fail="raise"):
    """``libxc.eval_xc`` with isolated non-finite points recomputed at
    ``rho * (1 + NUDGE)``.  Points that stay non-finite raise, or are returned
    as NaN with ``on_fail="nan"``."""
    rho = np.asarray(rho, dtype=float)
    out = libxc.eval_xc(fnal, rho, spin=spin, deriv=deriv, omega=omega)
    exc = out[0]

    def nonfinite(arrays):
        bad = np.zeros(exc.shape[0], dtype=bool)
        for a in arrays:
            a = np.asarray(a)
            bad |= ~(np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1))
        return bad
    derivs = [a for a in (out[1] if deriv > 0 else ()) if a is not None]
    bad = nonfinite([exc] + derivs)
    if bad.any():
        sub = libxc.eval_xc(fnal, rho[..., bad] * (1.0 + NUDGE), spin=spin,
                            deriv=deriv, omega=omega)
        exc[bad] = sub[0]
        if deriv > 0:
            for a, b in zip(out[1], sub[1]):
                if a is not None:
                    a[bad] = b
        still = nonfinite([exc] + derivs)
        if still.any() and on_fail == "raise":
            raise FloatingPointError("%s (spin=%d, omega=%s) is non-finite at "
                                     "%d point(s)" % (fnal, spin, omega,
                                                      int(still.sum())))
    return out


def _nodes():
    k = np.arange(N_NODES - 1)
    cheb = np.cos(np.pi * (2 * k + 1) / (2 * (N_NODES - 1)))
    u = 0.5 * U_NODE_MAX * (1.0 - cheb)
    return np.concatenate(([0.0], np.sort(u)))


_U_NODES = _nodes()


def _rho_spin(rho, zeta):
    '''Spin densities from a spin-summed GGA density and zeta (tPBE: the
    gradient of zeta is dropped).'''
    out = np.zeros((2,) + rho.shape)
    half = 0.5 * rho
    out[0] = half * (1.0 + zeta)
    out[1] = half * (1.0 - zeta)
    return out


def _lagrange(nodes, vals, x):
    '''Lagrange interpolation; ``vals`` is (n_nodes, npoints), ``x`` is
    (npoints,) and never coincides with a node.'''
    nn = nodes.size
    w = np.ones(nn)
    for i in range(nn):
        d = nodes[i] - nodes
        d[i] = 1.0
        w[i] = 1.0 / np.prod(d)
    coef = w[:, None] / (x[None, :] - nodes[:, None])
    return (coef * vals).sum(axis=0) / coef.sum(axis=0)


def exc_ontop(fnal, rho, u, omega=None):
    '''Exchange-correlation energy per particle at zeta^2 = u.

    ``rho`` is (4, N) (value and gradient, spin summed), ``u`` is
    1 - 4 Pi / rho^2 per point, clipped to [U_CLAMP, U_MAX].
    '''
    rho = np.asarray(rho, dtype=float)
    if rho.ndim == 1:
        rho = rho[None, :]
    u = np.clip(np.asarray(u, dtype=float), U_CLAMP, U_MAX)
    out = np.zeros(u.size)
    live = rho[0] > RHO_CUTOFF
    pos = (u >= 0.0) & live
    if pos.any():
        out[pos] = eval_xc_finite(fnal, _rho_spin(rho[:, pos], np.sqrt(u[pos])),
                                  spin=1, deriv=0, omega=omega, on_fail="nan")[0]
    neg = (~pos) & live
    if neg.any():
        rn = rho[:, neg]
        vals = np.empty((_U_NODES.size, int(neg.sum())))
        for k, uk in enumerate(_U_NODES):
            vals[k] = eval_xc_finite(fnal, _rho_spin(rn, np.sqrt(uk)), spin=1,
                                     deriv=0, omega=omega, on_fail="nan")[0]
        out[neg] = _lagrange(_U_NODES, vals, u[neg])
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = eval_xc_finite(fnal, rho[:, bad], spin=0, deriv=0,
                                  omega=omega)[0]
    return out


def ratio_from_pair_density(rho, Pi, tol=1e-12):
    '''u = 1 - 4 Pi / rho^2, set to 0 where the density vanishes.'''
    rho = np.asarray(rho, dtype=float)
    Pi = np.asarray(Pi, dtype=float)
    u = np.zeros_like(rho)
    ok = rho > tol
    u[ok] = 1.0 - 4.0 * Pi[ok] / rho[ok] ** 2
    return u


def dexc_du(fnal, rho, u, omega=None, h=1e-4):
    '''d e_xc / d u at fixed rho, by central difference on `exc_ontop`.'''
    ep = exc_ontop(fnal, rho, np.asarray(u) + h, omega=omega)
    em = exc_ontop(fnal, rho, np.asarray(u) - h, omega=omega)
    return (ep - em) / (2.0 * h)
