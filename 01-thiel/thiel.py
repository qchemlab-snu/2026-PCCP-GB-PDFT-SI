import os
import sys

import numpy as np
from pyscf import gto, scf, dft, fci
from pyscf.mcscf import addons
from pyscf.gbci import gbpdft

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gbsrpdft

# name: ncas, nelecas, active MOs (B3LYP/def2-TZVP), nroots
molecules = {
    'ethene':           (2,  (1, 1), [8, 9], 2),
    'E-butadiene':      (4,  (2, 2), [14, 15, 16, 17], 3),
    'all-E-hexatriene': (6,  (3, 3), [20, 21, 22, 23, 24, 26], 3),
    'benzene':          (6,  (3, 3), [17, 20, 21, 22, 23, 30], 11),
    'formaldehyde':     (4,  (3, 3), [6, 7, 8, 9], 4),
    'formamide':        (4,  (3, 3), [10, 11, 12, 13], 4),
    'furan':            (5,  (3, 3), [12, 17, 18, 19, 20], 4),
    'pyrrole':          (5,  (3, 3), [14, 17, 18, 19, 21], 4),
    'imidazole':        (7,  (4, 4), [14, 16, 17, 18, 19, 21, 29], 6),
    'pyridine':         (7,  (4, 4), [17, 19, 20, 21, 22, 23, 29], 11),
    'pyrazine':         (8,  (5, 5), [16, 18, 19, 20, 21, 22, 23, 29], 16),
    'pyrimidine':       (9,  (6, 6), [8, 15, 18, 19, 20, 21, 22, 23, 28], 13),
    'pyridazine':       (9,  (6, 6), [7, 16, 18, 19, 20, 21, 22, 23, 28], 13),
    's-triazine':       (9,  (6, 6), [14, 17, 18, 19, 20, 21, 22, 23, 27], 19),
    's-tetrazine':      (10, (7, 7), [15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 21),
    'uracil':           (10, (7, 7), [19, 24, 25, 26, 27, 28, 29, 30, 31, 37], 11),
}

name = sys.argv[1]
ncas, nelecas, cas_list, nroots = molecules[name]
nocc = sum(nelecas) // 2
weights = np.ones(nroots) / nroots

xyz = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '00-xyz', 'thiel', name + '.xyz')
mol = gto.M(atom='\n'.join(open(xyz).read().splitlines()[2:]), basis='def2-tzvp', verbose=0)

mf = scf.RHF(mol)
mf.conv_tol = 1e-10
mf.max_cycle = 300
mf.kernel()

ks = dft.RKS(mol, xc='b3lyp')
ks.conv_tol = 1e-9
ks.max_cycle = 300
ks.kernel()
mf.mo_coeff = ks.mo_coeff
mf.mo_occ = ks.mo_occ
mf.mo_energy = ks.mo_energy


def make_mc(otxc):
    mc = gbpdft.GBCI(mf, otxc, ncas, nelecas,
                     group_a={'mo': [list(range(nocc)), list(range(nocc, ncas))]})
    mc.fcisolver.nroots = nroots
    mc.fcisolver.spin = 0
    mc.fcisolver.max_cycle = 300
    mc.fix_spin_(ss=0, shift=1.0)
    return mc


def show(label, e, ci):
    s2 = [fci.spin_op.spin_square0(np.asarray(c), ncas, nelecas)[0] for c in ci]
    e = np.sort([ei for ei, s in zip(e, s2) if s < 0.3])
    print('%-12s' % label, ' '.join('%6.2f' % x for x in (e[1:] - e[0]) * 27.211386245988))


mc = make_mc('tPBE')
mo = addons.sort_mo(mc, mf.mo_coeff, cas_list, 1)
mc.kernel(mo)
show('GBCI', mc.e_gbci, mc.ci)

for otxc in ['tPBE', 'tPBE0']:
    ss = make_mc(otxc)
    ss.kernel(mo)
    show('SS ' + otxc, ss.e_states, ss.ci)

    xms = make_mc(otxc).multi_state(weights, 'xms')
    xms.kernel(mo)
    show('XMS ' + otxc, xms.e_states, xms.get_ci_adiabats(uci='MSGBPDFT'))

    fcd = make_mc(otxc).multi_state(weights, 'fcd')
    fcd.kernel(mo)
    show('FCD ' + otxc, fcd.e_states, fcd.get_ci_adiabats(uci='MSGBPDFT'))

    lin = make_mc(otxc).multi_state(weights, 'lin')
    lin.kernel(mo)
    show('L ' + otxc, lin.e_states, lin.ci)

for fx, lam in [('srPBE', 0.0), ('srPBE0', 0.25)]:
    r = gbsrpdft.kernel_ss(mc, mc.ci, mo, omega=0.4, lam=lam)
    show('SS ' + fx, r['e_states'], mc.ci)

    r = gbsrpdft.kernel_xms(mc, mc.ci, mo, omega=0.4, lam=lam)
    show('XMS ' + fx, r['e_states'], r['ci'])

    r = gbsrpdft.kernel_fcd(mc, mc.ci, mo, omega=0.4, lam=lam)
    show('FCD ' + fx, r['e_states'], r['ci'])

    r = gbsrpdft.kernel_lin(mc, mc.ci, mo, omega=0.4, lam=lam)
    show('L ' + fx, r['e_states'], r['ci'])
