from ase.db import connect
import numpy as np
from matplotlib import pyplot as plt



db = connect('../db/dft-analysis.db')
nrgs_bare = np.array([entry.energy for entry in db.select(type='bare')])
nrgs_acr = np.array([entry.energy for entry in db.select(type='acr')])
[Ag_bulk_nrg, Pd_in_Ag_bulk_nrg] = [entry.energy for entry in db.select(type='bulk')]


acr_potential = -47.3357
# segregation energy under vacuum
n_pd = np.array([entry.symbols.count('Pd') for entry in db.select(type='bare')])
seg_nrg_vac = (nrgs_bare + n_pd * Ag_bulk_nrg) - (nrgs_bare[0] + n_pd * Pd_in_Ag_bulk_nrg)

# adsorption energy
ad_nrg = nrgs_acr - nrgs_bare - acr_potential

# segregation energy under oxygen
seg_nrg_acr = seg_nrg_vac + (ad_nrg - ad_nrg[0])



fig = plt.figure(figsize=(7, 4))
xs = range(1, 17)
ax = fig.add_subplot(111)
ax.plot(xs, seg_nrg_vac, 'C3-o')
ax.plot(xs, ad_nrg, 'C4-o')
ax.plot(xs, seg_nrg_acr, 'C1-o')
ax.legend([r'Pd $\Delta E_{seg}$', r'Acrolein $\Delta E_{ads}$', r'$\Delta E_{seg} + \Delta\Delta E_{ads}$'])
ax.plot(xs, np.zeros(16), 'k--')
ax.set_ylabel('energy (eV)')
ax.set_xlabel('configuration id')
ax.set_xticks(range(1, 17))
ax.plot([1.5, 1.5], [-0.8, 0.6], 'k--', alpha=0.5)
ax.plot([5.5, 5.5], [-0.8, 1.2], 'k--', alpha=0.5)
ax.plot([11.5, 11.5], [-0.8, 1.2], 'k--', alpha=0.5)
ax.plot([15.5, 15.5], [-0.8, 1.2], 'k--', alpha=0.5)
ax.text(0.7, -0.8, 'Ag')
ax.text(2.5, -0.8, '1 Pd')
ax.text(7.5, -0.8, '2 Pd')
ax.text(12.5, -0.8, '3 Pd')
ax.text(15.6, -0.8, '4 Pd')
fig.savefig('./images/pd-acrolein-seg-vasp.png', dpi=300)