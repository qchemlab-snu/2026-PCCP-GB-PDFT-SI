import os
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.mcscf import addons
from pyscf.gbci import gbpdft, rdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gbsrpdft

# name: ncas, nelecas, active MOs (ROHF/cc-pVDZ), [donor atoms, acceptor atoms]
systems = {
    'NH3_F2':             (3, (2, 2), [9, 14, 15], [[0, 1, 2, 3], [4, 5]]),
    'NH3_OF2':            (3, (2, 2), [17, 18, 19], [[0, 1, 2, 3], [4, 5, 6]]),
    'acetone_F2':         (4, (3, 3), [23, 24, 25, 26], [list(range(10)), [10, 11]]),
    'acetone_CH3NO2':     (4, (2, 2), [31, 32, 33, 34], [list(range(10)), list(range(10, 17))]),
    'pyrazine_F2':        (4, (3, 3), [23, 29, 30, 31], [list(range(10)), [10, 11]]),
    'NH3_pyrazine':       (3, (1, 1), [23, 27, 29], [[10, 11, 12, 13], list(range(10))]),
    'pyrrole_pyrazine_h': (3, (1, 1), [39, 40, 41], [list(range(10, 20)), list(range(10))]),
    'pyrrole_pyrazine_s': (4, (2, 2), [38, 39, 40, 41], [list(range(10, 20)), list(range(10))]),
}

name = sys.argv[1]
ncas, nelecas, cas_list, fragments = systems[name]
nroots = 4
weights = np.ones(nroots) / nroots

xyz = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '00-xyz', 'CT', name + '.xyz')
mol = gto.M(atom='\n'.join(open(xyz).read().splitlines()[2:]), basis='cc-pvdz', verbose=0)

mf = scf.ROHF(mol)
mf.conv_tol = 1e-10
mf.max_cycle = 300
mf.kernel()


def make_mc(otxc):
    mc = gbpdft.GBCI(mf, otxc, ncas, nelecas, group_a={'atom': fragments}, grids_level=3)
    mc.fcisolver.nroots = nroots
    mc.fcisolver.spin = 0
    mc.fix_spin_(ss=0, shift=1.0)
    return mc


mc = make_mc('tPBE')
mo = addons.sort_mo(mc, mf.mo_coeff, cas_list, 1)
mc.kernel(mo)

# Mulliken charge of the donor, to identify the CT states
inter = mc.get_gbci_intermediates(mo)
data = rdm.precompute_rdm1s_data(mo, ncas, nelecas, mc.ncore, inter['dmet_core_list'],
                                 inter['conf_info_list'], inter['ov_list'])
s = mol.intor_symmetric('int1e_ovlp')
aoslice = mol.aoslice_by_atom()
donor_ao = np.concatenate([np.arange(*aoslice[a, 2:4]) for a in fragments[0]])
z_donor = sum(mol.atom_charge(a) for a in fragments[0])


def show(label, e, ci):
    order = np.argsort(e)
    out = []
    for i in order[1:]:
        dm = rdm.make_rdm1s_precomputed(np.asarray(ci[i]), data, mo_coeff=mo)
        q = z_donor - np.einsum('ij,ji->i', dm[0] + dm[1], s)[donor_ao].sum()
        out.append('%6.2f [%4.2f]' % ((e[i] - e[order[0]]) * 27.211386245988, q))
    print('%-12s' % label, ' '.join(out))


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
