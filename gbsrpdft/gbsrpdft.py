#!/usr/bin/env python
#
# Author: Minseok Oh <msjeff2001@snu.ac.kr>
#
'''GB-srPDFT: GBCI with a short-range on-top functional.

State-specific energy on a GBCI state Psi_I:

    E_I = (1 - lam) (<Psi_I|H^lr|Psi_I> + E^sr_Hxc[rho_I, Pi_I])
          + lam <Psi_I|H|Psi_I>

H^lr is the GBCI Hamiltonian with erf-attenuated two-electron integrals
(range separation omega) on the same bath, E^sr_Hxc the short-range Hartree
plus exchange-correlation energy with the spin polarization taken from the
on-top pair density (`srpdft`, screened).  lam = 0 is srPBE, lam = 0.25 is
srPBE0.

Multi-state schemes on a GBCI model space, all with the full-Coulomb GBCI
Hamiltonian off the diagonal and the state-specific energy on it:

    kernel_xms  XMS intermediate states (pyscf.gbci.xmsgbpdft)
    kernel_fcd  FCD diabats (pyscf.gbci.fcdgbpdft, operator from group_a)
    kernel_lin  L-PDFT form: E^sr_Hxc linearized in (rho, Pi) about the
                average of the model states, then diagonalized
'''

import numpy as np
from scipy import linalg

from pyscf import ao2mo
from pyscf.dft import gen_grid, numint
from pyscf.fci import cistring
from pyscf.gbci import direct_gbci, fcdgbpdft, msgbpdft, xmsgbpdft
from pyscf.gbci import otpd as gbci_otpd
from pyscf.gbci import rdm as gbci_rdm

from . import srpdft

DEFAULT_OMEGA = 0.4
SR_X = "GGA_X_PBE_ERF_GWS"
SR_C = "GGA_C_PBE_ERF_GWS"


class _range_coulomb(object):
    '''erf-attenuated two-electron integrals inside the context; the SCF
    object's in-core integrals (full Coulomb) are set aside.'''

    def __init__(self, mc, omega):
        self.mc, self.omega = mc, omega

    def __enter__(self):
        self._range = self.mc.mol.with_range_coulomb(self.omega)
        self._range.__enter__()
        self._eri = getattr(self.mc._scf, "_eri", None)
        self.mc._scf._eri = None
        return self

    def __exit__(self, *args):
        self.mc._scf._eri = self._eri
        return self._range.__exit__(*args)


class _extra_hcore(object):
    '''A one-body AO operator added to ``mc.get_hcore`` inside the context.'''

    def __init__(self, mc, v_ao):
        self.mc, self.v_ao = mc, v_ao

    def __enter__(self):
        self._saved = self.mc.get_hcore
        shifted = np.asarray(self._saved()) + np.asarray(self.v_ao)
        self.mc.get_hcore = lambda *args, **kwargs: shifted
        return self

    def __exit__(self, *args):
        self.mc.get_hcore = self._saved
        return False


def _grids(mc):
    grids = getattr(mc, "_gbsrpdft_grids", None)
    if grids is None:
        grids = gen_grid.Grids(mc.mol)
        grids.build()
        mc._gbsrpdft_grids = grids
    return grids


def sr_hxc_energy(mc, dm1, ci, otpd_data, mo_coeff, omega=DEFAULT_OMEGA):
    '''Short-range Hartree + xc energy of one GBCI state.'''
    dm1 = np.asarray(dm1)
    j_full = np.asarray(mc._scf.get_j(dm=dm1, hermi=1))
    with _range_coulomb(mc, omega):
        j_lr = np.asarray(mc._scf.get_j(dm=dm1, hermi=1))
    e = 0.5 * float(np.tensordot(j_full - j_lr, dm1))
    ci = np.asarray(ci).real
    cache = gbci_otpd.make_root_intermediates(ci, otpd_data)
    ni = numint.NumInt()
    for ao, mask, weight, _ in ni.block_loop(mc.mol, _grids(mc), mc.mol.nao, 1,
                                             mc.max_memory):
        rho = ni.eval_rho(mc.mol, ao, dm1, mask, "GGA", hermi=1)
        Pi = np.asarray(gbci_otpd.get_ontop_pair_density(
            mc, ao, ci, otpd_data, mo_coeff, 0, mask,
            ot_root_cache=cache)).ravel()
        u = np.maximum(srpdft.ratio_from_pair_density(rho[0], Pi), 0.0)
        for fnal in (SR_X, SR_C):
            e += float(np.dot(srpdft.exc_ontop(fnal, rho, u, omega=omega)
                              * rho[0], weight))
    if not np.isfinite(e):
        raise RuntimeError("short-range energy is not finite")
    return e


def lr_hamiltonian(mc, mo_coeff, intermediates, omega, v_ao=None):
    '''(h1e, ecore_list, eri) of H^lr (+ v_ao) on the GBCI bath.'''
    act = mc.get_active_dm(mo_coeff)
    with _range_coulomb(mc, omega):
        if v_ao is None:
            h1e, ecore = mc.get_h1cas(act, intermediates["mo_list"],
                                      intermediates["dmet_core_list"])
        else:
            with _extra_hcore(mc, v_ao):
                h1e, ecore = mc.get_h1cas(act, intermediates["mo_list"],
                                          intermediates["dmet_core_list"])
        eri = mc.get_h2eff(mo_coeff)
    return np.asarray(h1e), np.asarray(ecore), np.asarray(eri)


def full_hamiltonian(mc, mo_coeff, intermediates):
    '''(h1e, ecore_list, eri) of the full-Coulomb GBCI Hamiltonian.'''
    act = mc.get_active_dm(mo_coeff)
    h1e, ecore = mc.get_h1cas(act, intermediates["mo_list"],
                              intermediates["dmet_core_list"])
    return (np.asarray(h1e), np.asarray(ecore),
            np.asarray(mc.get_h2eff(mo_coeff)))


def mix_hamiltonians(ham_a, ham_b, wa, wb):
    return tuple(wa * np.asarray(a) + wb * np.asarray(b)
                 for a, b in zip(ham_a, ham_b))


def solve(mc, ham, intermediates, nroots, ci0=None):
    '''Lowest ``nroots`` eigenstates of a GBCI Hamiltonian ``ham``.'''
    h1e, ecore, eri = ham
    if ci0 is not None:
        ci0 = [np.asarray(c).ravel() for c in ci0]
    solver = mc.fcisolver
    saved = solver.nroots
    solver.nroots = nroots
    try:
        e, c = solver.kernel(h1e, eri, mc.ncas, mc.nelecas,
                             intermediates["conf_info_list"],
                             intermediates["ov_list"], ecore, ci0=ci0)
    finally:
        solver.nroots = saved
    ci = list(c) if isinstance(c, (list, tuple)) else [c]
    return np.atleast_1d(np.asarray(e, dtype=float)), ci


def hamiltonian_matrix(mc, ham, intermediates, ci_list):
    '''<Psi_I|ham|Psi_J> over ``ci_list``.'''
    h1e, ecore, eri = ham
    erieff = mc.fcisolver.absorb_h1e(h1e, eri, mc.ncas, mc.nelecas, 0.5)
    link_index = (cistring.gen_linkstr_index(range(mc.ncas), mc.nelecas[0]),
                  cistring.gen_linkstr_index(range(mc.ncas), mc.nelecas[1]))
    ts = direct_gbci.gen_excitations(mc.ncas, mc.nelecas, link_index=link_index)
    hci = []
    for civec in ci_list:
        hc = msgbpdft._contract_hamiltonian(
            mc.fcisolver, erieff, np.asarray(civec), mc.ncas, mc.nelecas,
            intermediates["conf_info_list"], intermediates["ov_list"],
            ecore, link_index, ts)
        hci.append(np.asarray(hc).ravel())
    n = len(ci_list)
    h = np.empty((n, n))
    for i, bra in enumerate(ci_list):
        for j in range(n):
            h[i, j] = np.vdot(np.asarray(bra).ravel(), hci[j]).real
    return 0.5 * (h + h.T)


class _Context(object):
    '''Bath-dependent quantities, built once per model space.'''

    def __init__(self, mc, mo_coeff, intermediates, omega, lam):
        self.mc = mc
        self.mo = mo_coeff
        self.inter = intermediates
        self.omega = float(omega)
        self.lam = float(lam)
        self.rdm_data = gbci_rdm.precompute_rdm1s_data(
            mo_coeff, mc.ncas, mc.nelecas, mc.ncore,
            intermediates["dmet_core_list"], intermediates["conf_info_list"],
            intermediates["ov_list"])
        self.otpd_data = gbci_otpd.make_otpd_intermediates(
            mc.mol.intor("int1e_ovlp"), mo_coeff, mc.ncas, mc.nelecas,
            mc.ncore, intermediates["dmet_core_list"],
            intermediates["conf_info_list"], intermediates["ov_list"])
        self.ham_lr = lr_hamiltonian(mc, mo_coeff, intermediates, omega)
        self.ham_full = full_hamiltonian(mc, mo_coeff, intermediates)

    def dm1(self, civec):
        d = gbci_rdm.make_rdm1s_precomputed(np.asarray(civec).real,
                                            self.rdm_data, mo_coeff=self.mo)
        return np.asarray(d[0]) + np.asarray(d[1])

    def energies(self, ci_list):
        '''State-specific GB-srPDFT energy of each state.'''
        e = np.diag(hamiltonian_matrix(self.mc, self.ham_lr, self.inter,
                                       ci_list)).copy()
        e += [sr_hxc_energy(self.mc, self.dm1(c), c, self.otpd_data, self.mo,
                            self.omega) for c in ci_list]
        if self.lam:
            e_full = np.diag(hamiltonian_matrix(self.mc, self.ham_full,
                                                self.inter, ci_list))
            e = (1 - self.lam) * e + self.lam * e_full
        return e

    def effective(self, theta):
        '''H^eff in the basis ``theta``: GBCI off the diagonal, GB-srPDFT on
        it; returns the eigenvalues and the adiabatic states.'''
        heff = hamiltonian_matrix(self.mc, self.ham_full, self.inter, theta)
        heff[np.diag_indices_from(heff)] = self.energies(theta)
        e, si = linalg.eigh(heff)
        return {"e_states": e, "si": si, "heff": heff, "ci_theta": theta,
                "ci": list(np.tensordot(si.T, np.asarray(theta), axes=1))}


def _context(mc, mo_coeff, intermediates, omega, lam):
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if intermediates is None:
        intermediates = mc.get_gbci_intermediates(mo_coeff)
    return _Context(mc, mo_coeff, intermediates, omega, lam)


def kernel_ss(mc, ci, mo_coeff=None, omega=DEFAULT_OMEGA, lam=0.0,
              intermediates=None):
    '''State-specific GB-srPDFT energies of the GBCI states ``ci``.'''
    ctx = _context(mc, mo_coeff, intermediates, omega, lam)
    return {"e_states": ctx.energies([np.asarray(c) for c in ci])}


def kernel_xms(mc, ci, mo_coeff=None, omega=DEFAULT_OMEGA, lam=0.0,
               intermediates=None):
    '''XMS-GB-srPDFT on the model space ``ci``.'''
    ctx = _context(mc, mo_coeff, intermediates, omega, lam)
    _, theta = xmsgbpdft.solve_safock(mc, mo_coeff=ctx.mo,
                                      ci=[np.asarray(c) for c in ci],
                                      data=ctx.rdm_data)
    return ctx.effective([np.asarray(t) for t in theta])


def kernel_fcd(mc, ci, mo_coeff=None, omega=DEFAULT_OMEGA, lam=0.0,
               intermediates=None):
    '''FCD-GB-srPDFT on the model space ``ci``; the FCD operator follows
    ``mc.group_a`` (`pyscf.gbci.fcdgbpdft`).'''
    ctx = _context(mc, mo_coeff, intermediates, omega, lam)
    _, theta = fcdgbpdft.solve_fcd(mc, mo_coeff=ctx.mo,
                                   ci=[np.asarray(c) for c in ci])
    return ctx.effective([np.asarray(t) for t in theta])


def _ontop_partials(fnal, rho, u, omega, h=1e-4):
    '''Per grid point, partials of f = rho e(rho, sigma, u) at fixed Pi:
    (df/drho, df/dsigma, df/dPi), by central differences.'''
    rho = np.asarray(rho, dtype=float)
    r0 = rho[0]
    live = r0 > srpdft.RHO_CUTOFF
    vrho = np.zeros_like(r0)
    vsig = np.zeros_like(r0)
    vpi = np.zeros_like(r0)
    if not live.any():
        return vrho, vsig, vpi
    rl, ul = rho[:, live], np.asarray(u)[live]
    f = lambda rr, uu: rr[0] * srpdft.exc_ontop(fnal, rr, uu, omega=omega)
    dr = h * rl[0]
    rp, rm = rl.copy(), rl.copy()
    rp[0] += dr
    rm[0] -= dr
    df_dr_u = (f(rp, ul) - f(rm, ul)) / (2 * dr)
    sig = (rl[1:4] ** 2).sum(axis=0)
    gp, gm = rl.copy(), rl.copy()
    gp[1:4] *= (1 + h)
    gm[1:4] *= (1 - h)
    ds = sig * ((1 + h) ** 2 - (1 - h) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_ds = np.where(ds > 0, (f(gp, ul) - f(gm, ul)) / ds, 0.0)
    e_u = srpdft.dexc_du(fnal, rl, ul, omega=omega, h=h)
    # d/drho at fixed Pi = d/drho at fixed u + 2 (1 - u) e_u
    vrho[live] = df_dr_u + 2.0 * (1.0 - ul) * e_u
    vsig[live] = df_ds
    vpi[live] = -4.0 * e_u / rl[0]
    return vrho, vsig, vpi


def _linearize(mc, ci_ref, ctx):
    '''E^sr_Hxc linearized in (rho, Pi) about the average of ``ci_ref``:
    E_lin[Psi] = <Psi|H^lr + V + W + h1e_pi + ecore_pi|Psi> + const.
    Returns (V_ao, W, h1e_pi, ecore_pi, const).'''
    omega, mo = ctx.omega, ctx.mo
    refs = [np.asarray(x).real for x in ci_ref]
    wts = np.full(len(refs), 1.0 / len(refs))
    dm0 = sum(w * ctx.dm1(x) for w, x in zip(wts, refs))
    caches = [gbci_otpd.make_root_intermediates(x, ctx.otpd_data) for x in refs]
    ncas, ncore = mc.ncas, mc.ncore
    nao = mc.mol.nao
    dcl = ctx.inter["dmet_core_list"]
    p = dcl.shape[0]
    ni = numint.NumInt()
    n2 = ncas * ncas
    Vxc = np.zeros((nao, nao))
    W = np.zeros((n2, n2))
    N = np.zeros((n2, nao, nao))
    ecore_pi = np.zeros(p)
    e_xc = 0.0
    int_vpi_pi0 = 0.0
    act = mo[:, ncore:ncore + ncas]
    diag = [np.asarray(dcl[g, g]) for g in range(p)]
    for ao, mask, weight, _ in ni.block_loop(mc.mol, _grids(mc), nao, 1,
                                             mc.max_memory):
        rho = ni.eval_rho(mc.mol, ao, dm0, mask, "GGA", hermi=1)
        Pi = sum(w * np.asarray(gbci_otpd.get_ontop_pair_density(
            mc, ao, x, ctx.otpd_data, mo, 0, mask,
            ot_root_cache=cc)).ravel() for w, x, cc in zip(wts, refs, caches))
        u = srpdft.ratio_from_pair_density(rho[0], Pi)
        clip = u < 0.0
        u = np.maximum(u, 0.0)
        vrho = np.zeros(rho.shape[1])
        vsig = np.zeros(rho.shape[1])
        vpi = np.zeros(rho.shape[1])
        for fnal in (SR_X, SR_C):
            e_xc += float(np.dot(rho[0] * srpdft.exc_ontop(
                fnal, rho, u, omega=omega), weight))
            a, b, d = _ontop_partials(fnal, rho, u, omega)
            # screened points: no Pi dependence
            a[clip] += 0.5 * rho[0][clip] * d[clip]
            d[clip] = 0.0
            vrho += a
            vsig += b
            vpi += d
        wv = numint._rks_gga_wv0(rho, (vrho, vsig), weight)
        aow = numint._scale_ao(ao[:4], wv)
        a0 = ao[0]
        blk = numint._dot_ao_ao(mc.mol, a0, aow, mask, None, None)
        Vxc += blk + blk.T
        vw = vpi * weight
        int_vpi_pi0 += float(np.dot(vw, Pi))
        Ma = a0 @ act
        P = (Ma[:, :, None] * Ma[:, None, :]).reshape(-1, n2)
        W += P.T @ (P * vw[:, None])
        for tu in range(n2):
            N[tu] += numint._dot_ao_ao(mc.mol, a0,
                                       numint._scale_ao(a0, vw * P[:, tu]),
                                       mask, None, None)
        for g in range(p):
            rs = ni.eval_rho(mc.mol, a0, diag[g], mask, "LDA", hermi=0)
            ecore_pi[g] += float(np.dot(vw, rs * rs))
    N = 0.5 * (N + N.transpose(0, 2, 1))
    Nf = N.reshape(n2, nao * nao)
    h1e_pi = np.zeros((p, p, ncas, ncas))
    pi_, pj_ = np.tril_indices(p)
    dm_block = np.asarray(dcl[pi_, pj_], order="C").reshape(len(pi_), nao * nao)
    blk = (dm_block @ Nf.T).reshape(len(pi_), ncas, ncas)
    h1e_pi[pi_, pj_] = blk
    h1e_pi[pj_, pi_] = blk.transpose(0, 2, 1)
    j_full = np.asarray(mc._scf.get_j(dm=dm0, hermi=1))
    with _range_coulomb(mc, omega):
        j_lr = np.asarray(mc._scf.get_j(dm=dm0, hermi=1))
    vh = j_full - j_lr
    e_h = 0.5 * float(np.tensordot(vh, dm0))
    V = vh + 0.5 * (Vxc + Vxc.T)
    const = e_h + e_xc - float(np.tensordot(V, dm0)) - int_vpi_pi0
    return V, W, h1e_pi, ecore_pi, const


def kernel_lin(mc, ci, mo_coeff=None, omega=DEFAULT_OMEGA, lam=0.0,
               intermediates=None):
    '''L-GB-srPDFT on the model space ``ci``.'''
    ctx = _context(mc, mo_coeff, intermediates, omega, lam)
    ci = [np.asarray(c).real for c in ci]
    ncas = mc.ncas
    V, W, h1e_pi, ecore_pi, const = _linearize(mc, ci, ctx)
    h1e, ecore, eri = lr_hamiltonian(mc, ctx.mo, ctx.inter, omega, v_ao=V)
    eri = ao2mo.restore(1, eri, ncas) + W.reshape(ncas, ncas, ncas, ncas)
    ham = (np.asarray(h1e) + h1e_pi, np.asarray(ecore) + ecore_pi + const, eri)
    if lam:
        f1, fc, fe = ctx.ham_full
        ham = mix_hamiltonians(ham, (f1, fc, ao2mo.restore(1, np.asarray(fe), ncas)),
                               1 - lam, lam)
    heff = hamiltonian_matrix(mc, ham, ctx.inter, ci)
    e, si = linalg.eigh(heff)
    return {"e_states": e, "si": si, "heff": heff,
            "ci": list(np.tensordot(si.T, np.asarray(ci), axes=1))}
