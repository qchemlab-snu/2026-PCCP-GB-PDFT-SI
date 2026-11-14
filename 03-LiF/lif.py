import os
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.gbci import gbpdft

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gbsrpdft

basis = {'Li': 'cc-pvtz', 'F': 'aug-cc-pvtz'}
nroots = 3
weights = np.ones(nroots) / nroots
r_list = np.round(np.arange(1.0, 9.01, 0.1), 2)


def make_mol(r):
    return gto.M(atom='Li 0 0 0; F 0 0 %.10f' % r, basis=basis, verbose=0)


def run_rhf(mol, prev=None):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.level_shift = 0.5
    mf.damp = 0.2
    mf.diis_space = 16
    dm0 = None
    if prev is not None:
        dm0 = mf.make_rdm1(scf.addons.project_mo_nr2nr(prev[0], prev[1], mol), prev[2])
    mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton().run(mf.make_rdm1())
    mf.level_shift = 0.0
    mf.damp = 0.0
    for _ in range(6):
        mo_i, _, stable, _ = mf.stability(return_status=True)
        if stable:
            break
        dm0 = mf.make_rdm1(mo_i, mf.mo_occ)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf.max_cycle = 300
        mf.kernel(dm0=dm0)
    return mf


def sigma_cas(mol, mf):
    # F 2pz (occupied) and Li 2s/3s/2pz/3pz (virtual) by projection on 6-31G AOs
    ref = mol.copy()
    ref.basis = '6-31g'
    ref.build()
    c = mf.mo_coeff
    occ = np.where(mf.mo_occ > 0)[0]
    vir = np.where(mf.mo_occ == 0)[0]

    def rotate(idx, labels, n):
        ao = ref.search_ao_label(labels)
        s12 = gto.intor_cross('int1e_ovlp', mol, ref)[:, ao]
        s22 = ref.intor('int1e_ovlp')[np.ix_(ao, ao)]
        x = c[:, idx].T @ s12
        v = np.linalg.eigh(x @ np.linalg.solve(s22, x.T))[1][:, ::-1]
        rot = c[:, idx] @ v
        return rot[:, :n], rot[:, n:]

    a_occ, c_occ = rotate(occ, ['F 2pz'], 1)
    a_vir, r_vir = rotate(vir, ['Li 2s', 'Li 3s', 'Li 2pz', 'Li 3pz'], 4)
    return np.hstack([c_occ, a_occ, a_vir, r_vir])


# configuration groups from the atom grouping at 1.6 A, kept for all R
mol = make_mol(1.6)
mf = run_rhf(mol)
mo = sigma_cas(mol, mf)
mc = gbpdft.GBCI(mf, 'tPBE0', 5, (1, 1), group_a={'atom': [[0], [1]]}, grids_level=3)
group_a = {'occ': [list(map(int, g)) for g in mc.get_gbci_intermediates(mo)['group']]}


def make_mc(mf):
    mc = gbpdft.GBCI(mf, 'tPBE0', 5, (1, 1), group_a=group_a, grids_level=3)
    mc.fcisolver.nroots = nroots
    mc.fcisolver.spin = 0
    mc.fix_spin_(ss=0, shift=1.0)
    mc.fcd_atoms = [0]
    return mc


print('R(A)  two lowest 1Sigma+ energies (Hartree): GBCI | SS, XMS, FCD, L tPBE0 | SS, XMS, FCD, L srPBE0')
prev = None
for r in r_list:
    mol = make_mol(r)
    mf = run_rhf(mol, prev)
    prev = (mol.copy(), mf.mo_coeff, mf.mo_occ)
    mo = sigma_cas(mol, mf)

    mc = make_mc(mf)
    mc.kernel(mo)
    e = [mc.e_gbci, mc.e_states]
    for method in ['xms', 'fcd', 'lin']:
        ms = make_mc(mf).multi_state(weights, method)
        ms.kernel(mo)
        e.append(ms.e_states)
    for kernel in [gbsrpdft.kernel_ss, gbsrpdft.kernel_xms, gbsrpdft.kernel_fcd, gbsrpdft.kernel_lin]:
        e.append(kernel(mc, mc.ci, mo, omega=0.4, lam=0.25)['e_states'])
    print('%4.1f ' % r + ' '.join('%10.6f %10.6f' % tuple(np.sort(x)[:2]) for x in e), flush=True)
