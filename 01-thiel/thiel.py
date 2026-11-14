import os
import sys

import numpy as np
from pyscf import gto, scf, dft, fci
from pyscf.mcscf import addons
from pyscf.gbci import gbpdft

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gbsrpdft  # noqa: E402

HA2EV = 27.211386245988
OMEGA = 0.4
LAM = 0.25      # srPBE0
XYZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00-xyz", "thiel")

# ncas, nelecas, active MOs (1-based, B3LYP/def2-TZVP), number of states
MOLECULES = {
    "ethene":           (2,  (1, 1), [8, 9], 2),
    "E-butadiene":      (4,  (2, 2), [14, 15, 16, 17], 3),
    "all-E-hexatriene": (6,  (3, 3), [20, 21, 22, 23, 24, 26], 3),
    "benzene":          (6,  (3, 3), [17, 20, 21, 22, 23, 30], 11),
    "formaldehyde":     (4,  (3, 3), [6, 7, 8, 9], 4),
    "formamide":        (4,  (3, 3), [10, 11, 12, 13], 4),
    "furan":            (5,  (3, 3), [12, 17, 18, 19, 20], 4),
    "pyrrole":          (5,  (3, 3), [14, 17, 18, 19, 21], 4),
    "imidazole":        (7,  (4, 4), [14, 16, 17, 18, 19, 21, 29], 6),
    "pyridine":         (7,  (4, 4), [17, 19, 20, 21, 22, 23, 29], 11),
    "pyrazine":         (8,  (5, 5), [16, 18, 19, 20, 21, 22, 23, 29], 16),
    "pyrimidine":       (9,  (6, 6), [8, 15, 18, 19, 20, 21, 22, 23, 28], 13),
    "pyridazine":       (9,  (6, 6), [7, 16, 18, 19, 20, 21, 22, 23, 28], 13),
    "s-triazine":       (9,  (6, 6), [14, 17, 18, 19, 20, 21, 22, 23, 27], 19),
    "s-tetrazine":      (10, (7, 7), [15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 21),
    "uracil":           (10, (7, 7), [19, 24, 25, 26, 27, 28, 29, 30, 31, 37], 11),
}


def read_xyz(name):
    lines = open(os.path.join(XYZ, name + ".xyz")).read().splitlines()
    return "\n".join(lines[2:])


def build(name):
    mol = gto.M(atom=read_xyz(name), basis="def2-tzvp", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.kernel()
    # GBCI keeps the HF Hamiltonian; only the orbitals come from B3LYP
    ks = dft.RKS(mol, xc="b3lyp")
    ks.conv_tol = 1e-9
    ks.max_cycle = 300
    ks.kernel()
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = ks.mo_coeff, ks.mo_occ, ks.mo_energy
    return mol, mf


def make_gbpdft(mf, otxc, ncas, nelecas, nroots):
    nocc = sum(nelecas) // 2
    mc = gbpdft.GBCI(mf, otxc, ncas, nelecas,
                     group_a={"mo": [list(range(nocc)), list(range(nocc, ncas))]})
    mc.fcisolver.nroots = nroots
    mc.fcisolver.spin = 0
    mc.fix_spin_(ss=0.0, shift=1.0)
    # the default 100 Davidson iterations do not converge the highest of 18-21 roots
    mc.fcisolver.max_cycle = 300
    return mc


def singlet_excitations(e, ci, ncas, nelecas):
    e = np.asarray(e)
    order = np.argsort(e)
    s2 = [fci.spin_op.spin_square0(np.asarray(ci[i]), ncas, nelecas)[0] for i in order]
    e = np.array([e[i] for i, s in zip(order, s2) if s < 0.3])
    return (e[1:] - e[0]) * HA2EV


name = sys.argv[1]
ncas, nelecas, cas_list, nroots = MOLECULES[name]
mol, mf = build(name)
weights = np.ones(nroots) / nroots
res = {}

for otxc in ("tPBE", "tPBE0"):
    # state-specific GB-PDFT on the GBCI roots
    mc = make_gbpdft(mf, otxc, ncas, nelecas, nroots)
    mo = addons.sort_mo(mc, mf.mo_coeff, cas_list, 1)
    mc.kernel(mo)
    if otxc == "tPBE":
        res["GBCI"] = (mc.e_gbci, mc.ci)
        mc_ref = mc
    res["SS " + otxc] = (mc.e_states, mc.ci)
    # multi-state GB-PDFT: XMS, FCD (excitation rank, from the MO groups), L
    for method in ("xms", "fcd", "lin"):
        ms = make_gbpdft(mf, otxc, ncas, nelecas, nroots).multi_state(weights, method)
        ms.kernel(mo)
        ci = ms.ci if method == "lin" else ms.get_ci_adiabats(uci="MSGBPDFT")
        res["%s %s" % (method.upper().replace("LIN", "L"), otxc)] = (ms.e_states, ci)

# GB-srPDFT (srPBE, srPBE0) on the same GBCI states
ci_gb = list(mc_ref.ci)
for fx, lam in (("srPBE", 0.0), ("srPBE0", LAM)):
    for label, kernel in (("SS", gbsrpdft.kernel_ss), ("XMS", gbsrpdft.kernel_xms),
                          ("FCD", gbsrpdft.kernel_fcd), ("L", gbsrpdft.kernel_lin)):
        r = kernel(mc_ref, ci_gb, mo_coeff=mo, omega=OMEGA, lam=lam)
        res["%s %s" % (label, fx)] = (r["e_states"], r.get("ci", ci_gb))

print("%s: singlet excitation energies (eV)" % name)
for key, (e, ci) in res.items():
    ex = singlet_excitations(e, ci, ncas, nelecas)
    print("%-12s %s" % (key, " ".join("%6.2f" % x for x in ex)))
