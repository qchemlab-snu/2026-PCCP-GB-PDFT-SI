# 2026-PCCP-GB-PDFT-SI

Input scripts for the calculations in the GB-PDFT paper (PCCP, 2026).

## Setup
This repository requires **pyscf** : https://github.com/pyscf/pyscf (version : 2.11.0)
This repository requires **pyscf-forge** : https://github.com/pyscf/pyscf-forge (for gbci)

## Contents

- `00-xyz` : geometries of the Thiel set (16 molecules) and the CT set (8 complexes)
- `01-thiel` : Thiel set, def2-TZVP, B3LYP orbitals
- `02-CT` : intermolecular charge-transfer set, cc-pVDZ, ROHF orbitals
- `gbsrpdft` : GB-srPDFT (short-range on-top functional), used by the scripts above

Each script prints GBCI and, for tPBE, tPBE0, srPBE and srPBE0, the
state-specific (SS), XMS, FCD and L-PDFT results.

GB-PDFT (tPBE, tPBE0) is run through `pyscf.gbci.gbpdft`; the multi-state
schemes follow `pyscf.mcpdft`:

```python
mc = gbpdft.GBCI(mf, "tPBE0", ncas, nelecas, group_a=group_a)
mc.kernel(mo)                                   # state-specific
mc.multi_state(weights, "xms").kernel(mo)       # XMS
mc.multi_state(weights, "fcd").kernel(mo)       # FCD
mc.multi_state(weights, "lin").kernel(mo)       # L-PDFT
```

The FCD operator follows `group_a`: the charge of the first fragment for
`{"atom": ...}` groups (CT set), the excitation rank for `{"mo": [occ, vir]}`
groups (Thiel set).  GB-srPDFT is provided in `gbsrpdft` and takes the GBCI
states of `mc` (`kernel_ss`, `kernel_xms`, `kernel_fcd`, `kernel_lin`;
`lam=0.25` gives srPBE0).

## Usage

```
python 01-thiel/thiel.py benzene
python 02-CT/ct.py NH3_F2
```

The argument is the molecule name (the file name in `00-xyz`).
