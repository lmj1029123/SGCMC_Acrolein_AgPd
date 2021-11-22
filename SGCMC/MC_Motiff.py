import sys
sys.path.append("../ML_Models")
from MC_Motiff_Helper import site_locator
from preprocess import cal_energy

import numpy as np
from ase.build import fcc111
from ase import Atom, Atoms
from ase.visualize import view
from ase.io import Trajectory
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from ase.units import kB

import os
import json
import pickle
import time
from copy import deepcopy

seed = int(sys.argv[1])
n_Pd = int(sys.argv[2])
d_cp = float(sys.argv[3])
T = float(sys.argv[4])
m_seed = int(sys.argv[5])
np.random.seed(seed)

slab_model_dir = f'../ML_Models/AgPd_slab_334_{m_seed}'
site_model_dir = f'../ML_Models/acrolein_adsorption_{m_seed}'
MC_model_path = f'AgPd_acrolein_MC_{m_seed}'

if not os.path.exists(MC_model_path):
	os.makedirs(MC_model_path)

traj_name = f'AgPd_acrolein_{d_cp}_{T}K_{n_Pd}_{seed}'
logfile = open(MC_model_path+'/'+traj_name+'log.txt','w+')
result = {}
result['sites'] = {}
result['1_Pd'] = {}
result['2_Pd'] = {}
result['Pd'] = {}

epochs = 500000

acrolein_energy = -47.3357 # 6 A vacuum, encut = 450
Ag_ads_energy = -0.04

# This is an arbitrary number as long as it is larger than 1/9 coverage
max_N_acrolein = 400

size = (30,30,4)
h = 1.5
cutoff = 6.5

# This is the closest distance between pseudo atoms
site_distance = 6.5

# We desorb whenever the atom in first layer is changed
desorb_chance = 1

# Initialize the slab
n_total = np.prod(size)
slab = fcc111('Ag', size = size, a = 4.163) # DFT calculated lattice
slab.center(vacuum = 6, axis = 2)
Pd_ids_offset = size[0]*size[1]*(size[2] - 2)
Pd_ids = np.random.choice(size[0]*size[1]*2, n_Pd, replace = False) + Pd_ids_offset
for i in Pd_ids:
	slab[i].symbol = 'Pd'


t0 = time.time()
result['size'] = size
result['h'] = h
result['cutoff'] = cutoff
result['site_distance'] = site_distance
result['max_N_acrolein'] = max_N_acrolein

# Initialization
# Note: It is important to use deepcopy here. Otherwise, slab will be changed for update steps.
sl = site_locator(slab = deepcopy(slab),
				h = h,
				slab_size = size,
				cutoff = cutoff,
				site_distance = site_distance,
				site_model_dir = site_model_dir,
				max_N_acrolein = max_N_acrolein,
				m_element = 'Pd')
t1 = time.time()

# Initialize active sites
all_active_sites = sl.get_all_active_sites()
t2 = time.time()

# Get adsorption energy
site_energies = sl.get_site_energies(all_active_sites)
t3 = time.time()

# Fill the sites based on adsortpion energy
occupied_sites = np.array((),dtype=np.int16)
occupied_dict= {}
N_acrolein, occupied_sites, occupied_dict = sl.allocate_sites(site_energies,
															all_active_sites,
															occupied_sites,
															occupied_dict)
t4 = time.time()
sl.update_sites(occupied_sites, occupied_dict, N_acrolein)

site_E0 = np.sum([occupied_dict[str(int(s))] for s in occupied_sites])
slab_E0 = cal_energy(slab_model_dir, slab)
E0 = site_E0 + slab_E0
E = [E0]

nl_bare1 = NeighborList(cutoffs = [cutoff/2]*len(slab),
						skin = 0.0,
						bothways = True,
						self_interaction = True,
						primitive = NewPrimitiveNeighborList)
nl_bare1.update(slab)

nl_bare2 = NeighborList(cutoffs = [cutoff]*len(slab),
						skin = 0.0,
						bothways = True,
						self_interaction = True,
						primitive = NewPrimitiveNeighborList)
nl_bare2.update(slab)

t5 = time.time()

success = 0
attempt = 0

for epo in range(epochs):
	print(f'{success}/{attempt}')
	t5 = time.time()
	ind = np.random.choice(size[0]*size[1]*2, 1)[0]+size[0]*size[1]*(size[2]-2)

	indices0, _ = nl_bare1.get_neighbors(ind)
	indices1, _ = nl_bare2.get_neighbors(ind)
	cal_list = np.array([list(indices1).index(i) for i in indices0])

	# This calculate the change of energy of the bare slab
	# This is not affected by the adsorbate
	E1 = cal_energy(slab_model_dir, slab[indices1], cal_list)
	temp_slab = deepcopy(slab)
	if temp_slab[ind].symbol == 'Ag':
		temp_slab[ind].symbol = 'Pd'
		sl.update_slab(ind, 'Pd')
		label = 'Ag2Pd'
	elif temp_slab[ind].symbol == 'Pd':
		temp_slab[ind].symbol = 'Ag'
		sl.update_slab(ind, 'Ag')
		label = 'Pd2Ag'
	E2 = cal_energy(slab_model_dir, temp_slab[indices1], cal_list)



	# Find sites affected by the change of atom
	sites = sl.get_sites(ind)
	occupied_sites = sl.occupied_sites
	occupied_dict = sl.occupied_dict
	N_acrolein = sl.N_acrolein

	affected_sites = np.intersect1d(occupied_sites, sites)

	old_sites_energy = np.sum([occupied_dict[str(int(s))] for s in affected_sites])

	# Decide whether we desorb the affected acrolein:
	if ((ind >= (np.prod(size) - np.prod(size[:2])))
		and (np.random.uniform(0,1,1) <= desorb_chance)):

		# Desorb the pseudo atom and re-allocate them
		temp_occupied_sites = np.array(list(set(occupied_sites) - set(affected_sites)))
		temp_occupied_dict = deepcopy(occupied_dict)
		for a_s in affected_sites:
			del temp_occupied_dict[str(int(a_s))]
		potential_sites = sl.get_potential_sites(ind)
		potential_open_sites = np.array((),dtype=np.int16)
		for p_s in potential_sites:
			if sl.is_site_open(p_s, temp_occupied_sites):
				potential_open_sites = np.append(potential_open_sites, p_s)

		# This will make use of the updated slab
		active_sites = sl.get_active_sites(potential_open_sites)
		site_energies = sl.get_site_energies(active_sites)
		N_acrolein, occupied_sites, occupied_dict = sl.allocate_sites(site_energies,
																	active_sites,
																	temp_occupied_sites,
																	temp_occupied_dict)

		new_affected_sites = np.array(list(set(occupied_sites) - set(temp_occupied_sites)))
		temp_sites_energy = np.sum([occupied_dict[str(int(s))] for s in new_affected_sites])
	else:
		# Calculate new adsorption energy
		temp_site_energies = sl.get_site_energies(affected_sites)
		for a_s, s_e in zip(affected_sites, temp_site_energies):
			occupied_dict[str(int(a_s))] = s_e
		temp_sites_energy = np.sum([occupied_dict[str(int(s))] for s in affected_sites])

		# No need to update occupied_sites, they are the same as before.

	sl.update_sites(occupied_sites, occupied_dict, N_acrolein)

	dE = ((temp_sites_energy + E2 + (Ag_ads_energy)*(max_N_acrolein - sl.N_acrolein))
		- (old_sites_energy + E1 + (Ag_ads_energy)*(max_N_acrolein - sl.last_N_acrolein)))

	# Sign of chemical potential will be based on the elements changed
	if label == 'Ag2Pd':
		cp = d_cp
		n_Pd_temp = n_Pd + 1
	elif label == 'Pd2Ag':
		cp = -d_cp
		n_Pd_temp = n_Pd - 1

	if dE - cp < 0:
		slab = temp_slab
		E += [E[-1]+dE]
		success += 1
		n_Pd = n_Pd_temp
		logfile.write(f'{E[-1]}\n')
		result['sites'][f'{success}'] = [int(s) for s in occupied_sites]
		first_Pd = 0
		for atom in slab[size[0]*size[1]*(size[2]-1):]:
			if atom.symbol == 'Pd':
				first_Pd += 1
		second_Pd = 0
		for atom in slab[size[0]*size[1]*(size[2]-2):size[0]*size[1]*(size[2]-1)]:
			if atom.symbol == 'Pd':
				second_Pd += 1
		result['1_Pd'][f'{success}'] = int(first_Pd)
		result['2_Pd'][f'{success}'] = int(n_Pd - first_Pd)
		symbols = slab[size[0]*size[1]*2:].get_chemical_symbols()
		result['Pd'][f'{success}'] = [i for i, x in enumerate(symbols) if x == 'Pd']

	elif np.exp(-(dE - cp) / (kB * T)) > np.random.rand():
		slab = temp_slab
		E += [E[-1]+dE]
		success += 1
		n_Pd = n_Pd_temp
		logfile.write(f'{E[-1]}\n')
		result['sites'][f'{success}'] = [int(s) for s in occupied_sites]
		first_Pd = 0
		for atom in slab[size[0]*size[1]*(size[2]-1):]:
			if atom.symbol == 'Pd':
				first_Pd += 1
		second_Pd = 0
		for atom in slab[size[0]*size[1]*(size[2]-2):size[0]*size[1]*(size[2]-1)]:
			if atom.symbol == 'Pd':
				second_Pd += 1
		result['1_Pd'][f'{success}'] = int(first_Pd)
		result['2_Pd'][f'{success}'] = int(n_Pd - first_Pd)
		symbols = slab[size[0]*size[1]*2:].get_chemical_symbols()
		result['Pd'][f'{success}'] = [i for i, x in enumerate(symbols) if x == 'Pd']

	else:
		sl.revert_update_sites()
		sl.revert_update_slab()

	attempt += 1
	t6 = time.time()
	logfile.write(f'{t6-t5}, {success}/{attempt}, {n_Pd/n_total}\r\n')
	logfile.flush()
	if epo % 5000 == 0:
		with open(MC_model_path+'/'+traj_name+'_result.json', 'w') as outfile:
			json.dump(result, outfile)


logfile.close()
with open(MC_model_path+'/'+traj_name+'_result.json', 'w') as outfile:
	json.dump(result, outfile)
