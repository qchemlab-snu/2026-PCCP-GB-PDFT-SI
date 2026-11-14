import os
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.mcscf import addons
from pyscf.gbci import gbpdft, rdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gbsrpdft  # noqa: E402

HA2EV = 27.211386245988
OMEGA = 0.4
LAM = 0.25      # srPBE0
NROOTS = 4
XYZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00-xyz", "CT")

# ncas, nelecas, active MOs (1-based, ROHF/cc-pVDZ), atoms of [donor, acceptor]
SYSTEMS = {
    "NH3_F2":             (3, (2, 2), [9, 14, 15],
                           [[0, 1, 2, 3], [4, 5]]),
    "NH3_OF2":            (3, (2, 2), [17, 18, 19],
                           [[0, 1, 2, 3], [4, 5, 6]]),
    "acetone_F2":         (4, (3, 3), [23, 24, 25, 26],
                           [list(range(10)), [10, 11]]),
    "acetone_CH3NO2":     (4, (2, 2), [31, 32, 33, 34],
                           [list(range(10)), list(range(10, 17))]),
    "pyrazine_F2":        (4, (3, 3), [23, 29, 30, 31],
                           [list(range(10)), [10, 11]]),
    "NH3_pyrazine":       (3, (1, 1), [23, 27, 29],
                           [[10, 11, 12, 13], list(range(10))]),
    "pyrrole_pyrazine_h": (3, (1, 1), [39, 40, 41],
                           [list(range(10, 20)), list(range(10))]),
    "pyrrole_pyrazine_s": (4, (2, 2), [38, 39, 40, 41],
                           [list(range(10, 20)), list(range(10))]),
}


def read_xyz(name):
    lines = open(os.path.join(XYZ, name + ".xyz")).read().splitlines()
    return "\n".join(lines[2:])


def make_gbpdft(mf, otxc, ncas, nelecas, fragments):
    # the donor is the first atom group; FCD uses its charge
    mc = gbpdft.GBCI(mf, otxc, ncas, nelecas, group_a={"atom": fragments},
                     grids_level=3)
    mc.fcisolver.nroots = NROOTS
    mc.fcisolver.spin = 0
    mc.fix_spin_(ss=0.0, shift=1.0)
    return mc


def donor_charge(mol, mo, ci, data, donor):
    s = mol.intor_symmetric("int1e_ovlp")
    aoslice = mol.aoslice_by_atom()
    aos = np.concatenate([np.arange(*aoslice[a, 2:4]) for a in donor])
    z = sum(mol.atom_charge(a) for a in donor)
    q = []
    for c in ci:
        dm1s = np.asarray(rdm.make_rdm1s_precomputed(np.asarray(c), data, mo_coeff=mo))
        q.append(z - np.einsum("ij,ji->i", dm1s[0] + dm1s[1], s)[aos].sum())
    return np.array(q)


name = sys.argv[1]
ncas, nelecas, cas_list, fragments = SYSTEMS[name]
mol = gto.M(atom=read_xyz(name), basis="cc-pvdz", verbose=0)
mf = scf.ROHF(mol)
mf.conv_tol = 1e-10
mf.max_cycle = 300
mf.kernel()
weights = np.ones(NROOTS) / NROOTS
res = {}

for otxc in ("tPBE", "tPBE0"):
    # state-specific GB-PDFT on the GBCI roots
    mc = make_gbpdft(mf, otxc, ncas, nelecas, fragments)
    mo = addons.sort_mo(mc, mf.mo_coeff, cas_list, 1)
    mc.kernel(mo)
    if otxc == "tPBE":
        res["GBCI"] = (mc.e_gbci, mc.ci)
        mc_ref = mc
    res["SS " + otxc] = (mc.e_states, mc.ci)
    # multi-state GB-PDFT: XMS, FCD (donor charge, from the atom groups), L
    for method in ("xms", "fcd", "lin"):
        ms = make_gbpdft(mf, otxc, ncas, nelecas, fragments).multi_state(weights, method)
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

# CT states are identified by the donor charge q
inter = mc_ref.get_gbci_intermediates(mo)
data = rdm.precompute_rdm1s_data(mo, ncas, nelecas, mc_ref.ncore, inter["dmet_core_list"],
                                 inter["conf_info_list"], inter["ov_list"])
print("%s: excitation energies (eV) [donor charge]" % name)
for key, (e, ci) in res.items():
    order = np.argsort(e)
    e = np.asarray(e)[order]
    q = donor_charge(mol, mo, [ci[i] for i in order], data, fragments[0])
    print("%-12s %s" % (key, " ".join("%6.2f [%4.2f]" % (x, y) for x, y in
                                        zip((e[1:] - e[0]) * HA2EV, q[1:]))))
