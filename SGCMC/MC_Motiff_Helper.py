import sys
sys.path.append("../SimpleNN")
from fp_calculator import set_sym, cal_fp_only

import numpy as np
import torch
from ase import Atom
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from ase.data import atomic_numbers

from copy import deepcopy
import pickle
import hashlib
import os


class site_locator:
	""" This is a helper class used to keep track of the acrolein positions on
	AgPd slab"""

	def __init__(self, slab, h, slab_size, cutoff, site_distance, site_model_dir,
					max_N_acrolein, m_element = 'Pd', p_atom = 'C', eps = 0.1):
		"""Parameters:
		   ----------
			slab: the slab atoms. For each atom in slab,
			there should be a tag and tag=1 corresponds to a surface atom.
			h: distance of pseudo atom above the surface atom
			slab_size: the size of slab
			cutoff: cutoff radius for interactions
			site_distance: the minimum distances between sites
			site_model_dir: the directory for adsorption energy model
			max_N_acrolein: the maximum number of acrolein on the surface
			m_element: element type assigned to the active metal atom
			p_atom: element type assigned to the pseudo atom
			eps: error tolerance for distances between sites

		"""

		self.slab = slab
		self.m_element = m_element
		self.p_atom = p_atom
		self.slab_size = slab_size
		self.max_N_acrolein = max_N_acrolein
		self.site_model_dir = site_model_dir

		# Initialization
		self.N_acrolein = 0
		self.occupied_sites = np.array((),dtype = np.int16)
		self.occupied_dict = {}

		# this is used to store all active sites
		grand_slab = deepcopy(slab)

		site_ids = []
		site_positions = []
		si = 0

		# construct slab with pseudo atoms on top.
		ontop_ids = []
		p1 = slab[0].position
		p2 = slab[1].position
		p3 = slab[slab_size[0]].position
		p4 = slab[slab_size[0] + 1].position
		for atom in slab:
			if atom.tag == 1:
				position = atom.position + [0, 0, h]
				p_a = Atom(p_atom, position = position)
				site_positions += [position]
				grand_slab += p_a
				ontop_ids += [si]
				site_ids += [si]
				si += 1

		ontop_cutoff = np.sqrt(np.linalg.norm(p1 - p2)**2 + h**2) + eps

		# construct slab with pseudo atoms at bridge.
		bridge_ids = []
		for atom in slab:
			if atom.tag == 1:
				position = atom.position + 0.5*(p2 - p1) + [0, 0, h]
				p_a = Atom(p_atom, position = position)
				site_positions += [position]
				grand_slab += p_a
				bridge_ids += [si]
				site_ids += [si]
				si += 1

				position = atom.position + 0.5*(p3 - p1) + [0, 0, h]
				p_a = Atom(p_atom, position = position)
				site_positions += [position]
				grand_slab += p_a
				bridge_ids += [si]
				site_ids += [si]
				si += 1

				position = atom.position + 0.5*(p4 - p1) + [0, 0, h]
				p_a = Atom(p_atom, position = position)
				site_positions += [position]
				grand_slab += p_a
				bridge_ids += [si]
				site_ids += [si]
				si += 1

		# This only applies to fcc
		bridge_cutoff = np.sqrt(3/4* np.linalg.norm(p1 - p2)**2 + h**2) + eps


		# construct slab with pseudo atoms at hollow.
		hollow_ids = []
		for atom in slab:
			if atom.tag == 1:
				position = atom.position + 1/3*(p1 + p2 + p3)-p1 + [0, 0, h]

				p_a = Atom(p_atom, position = position)
				site_positions += [position]
				grand_slab += p_a
				hollow_ids += [si]
				site_ids += [si]
				si += 1

				position = atom.position + 1/3*(p2 + p3 + p4)-p1 + [0, 0, h]
				p_a = Atom(p_atom, position = position)
				site_positions += [position]
				grand_slab += p_a
				hollow_ids += [si]
				site_ids += [si]
				si += 1

		hollow_cutoff = np.sqrt(12/9*np.linalg.norm(p1 - p2)**2 + h**2) + eps

		layer1_slab = grand_slab[slab_size[0]*slab_size[1]*(slab_size[2] - 1):]
		layer2_slab = grand_slab[slab_size[0]*slab_size[1]*(slab_size[2] - 2):]
		site_cutoff = np.max([ontop_cutoff, bridge_cutoff, hollow_cutoff])

		# This is the neighborlist used to determine whether the site is active.
		site_cutoffs = [site_cutoff/2]*len(layer1_slab)
		site_nl = NeighborList(cutoffs = site_cutoffs,
								self_interaction = False,
								bothways = True,
								primitive = NewPrimitiveNeighborList)
		site_nl.update(layer1_slab)

		# This is the neighborlist used to find affected sites
		aff_cutoffs = [cutoff/2]*len(layer2_slab)
		nl = NeighborList(cutoffs = aff_cutoffs,
							self_interaction = False,
							bothways = True,
							primitive = NewPrimitiveNeighborList)
		nl.update(layer2_slab)

		# This is the neighborlist used for pseudo atom re-allocation
		potential_cutoffs = [(cutoff + site_distance)/2]*len(layer2_slab)
		potential_site_nl = NeighborList(cutoffs = potential_cutoffs,
											self_interaction = False,
											bothways = True,
											primitive = NewPrimitiveNeighborList)
		potential_site_nl.update(layer2_slab)

		# This is the neighborlist used to see if the active site is open
		occupied_cutoffs = [site_distance/2]*len(layer1_slab)
		occupied_site_nl = NeighborList(cutoffs = occupied_cutoffs,
										self_interaction = False,
										bothways = True,
										primitive = NewPrimitiveNeighborList)
		occupied_site_nl.update(layer1_slab)


		# This is the neighborlist used to calculate site fingerprint
		fp_cutoffs = [cutoff/2]*len(grand_slab)
		fp_nl = NeighborList(cutoffs = fp_cutoffs,
							self_interaction = False,
							bothways = True,
							primitive = NewPrimitiveNeighborList)
		fp_nl.update(grand_slab)

		self.grand_slab = grand_slab
		self.layer1_slab = layer1_slab
		self.layer2_slab = layer2_slab

		self.site_nl = site_nl
		self.nl = nl
		self.potential_site_nl = potential_site_nl
		self.occupied_site_nl = occupied_site_nl
		self.fp_nl = fp_nl

		self.ontop_ids = ontop_ids
		self.bridge_ids = bridge_ids
		self.hollow_ids = hollow_ids
		self.site_ids = site_ids
		self.site_positions = site_positions





	def update_slab(self, atom_id, new_element):
		"""When we update slab, we do not need to update neighborlist.
		We only update layer1_slab as it will determine the site"""

		# Used for revertion
		self.last_grand_slab = deepcopy(self.grand_slab)
		self.last_slab = deepcopy(self.slab)

		slab_size = self.slab_size
		self.grand_slab[atom_id].symbol = new_element
		self.layer1_slab = self.grand_slab[slab_size[0]*slab_size[1]*(slab_size[2] - 1):]
		self.layer2_slab = self.grand_slab[slab_size[0]*slab_size[1]*(slab_size[2] - 2):]
		self.slab[atom_id].symbol = new_element

	def revert_update_slab(self):
		slab_size = self.slab_size
		self.grand_slab = deepcopy(self.last_grand_slab)
		self.slab = deepcopy(self.last_slab)
		self.layer1_slab = self.grand_slab[slab_size[0]*slab_size[1]*(slab_size[2] - 1):]
		self.layer2_slab = self.grand_slab[slab_size[0]*slab_size[1]*(slab_size[2] - 2):]


	def update_sites(self, occupied_sites, occupied_dict, N_acrolein):
		self.last_N_acrolein = deepcopy(self.N_acrolein)
		self.last_occupied_sites = deepcopy(self.occupied_sites)
		self.last_occupied_dict = deepcopy(self.occupied_dict)
		self.N_acrolein = N_acrolein
		self.occupied_sites = occupied_sites
		self.occupied_dict = occupied_dict

	def revert_update_sites(self):
		self.N_acrolein = self.last_N_acrolein
		self.occupied_sites = self.last_occupied_sites
		self.occupied_dict = self.last_occupied_dict


	def get_sites(self, atom_id):
		"""Return all the sites within cutoff radius of the selected atom"""
		slab_id = atom_id - self.slab_size[0]*self.slab_size[1]*(self.slab_size[2] - 2)

		indices, offsets = self.nl.get_neighbors(slab_id)
		sites = []
		for i in indices:
			if self.layer2_slab[i].symbol == self.p_atom:
				sites += [i]

		# This will corresponds to site_ids
		sites = np.array(sites) - self.slab_size[0]*self.slab_size[1]*2
		return sites

	def get_potential_sites(self, atom_id):
		"""Return all the sites within cutoff + site_distance radius of the selected atom"""
		slab_id = atom_id - self.slab_size[0]*self.slab_size[1]*(self.slab_size[2] - 2)

		indices, offsets = self.potential_site_nl.get_neighbors(slab_id)
		sites = []
		for i in indices:
			if self.layer2_slab[i].symbol == self.p_atom:
				sites += [i]

		# This will corresponds to site_ids
		sites = np.array(sites) - self.slab_size[0]*self.slab_size[1]*2
		return sites


	def get_active_sites(self, sites):
		"""Return possible active sites among the sites in the input."""
		active_sites = []
		for i in sites:
			if i in self.ontop_ids:
				if self.is_active_site(site_id = i, site_type = 'ontop'):
					active_sites += [i]
			elif i in self.bridge_ids:
				if self.is_active_site(site_id = i, site_type = 'bridge'):
					active_sites += [i]
			elif i in self.hollow_ids:
				if self.is_active_site(site_id = i, site_type = 'hollow'):
					active_sites += [i]
			else:
				print('Invalid sites. This should not occur. (debug purpose)')
				raise
		return active_sites


	def get_all_active_sites(self):
		return self.get_active_sites(self.site_ids)



	def is_active_site(self, site_id, site_type):
		slab = self.layer1_slab
		nl = self.site_nl

		# This corresponds to atom_id of the site in layer1_slab
		site_id += self.slab_size[0]*self.slab_size[1]
		indices, offsets = nl.get_neighbors(site_id)

		symbols = []
		distances = []

		# Sort non pseudo atom neighbors based on distance
		for i, offset in zip(indices, offsets):
			if slab[i].symbol != self.p_atom:
				symbols += [slab[i].symbol]
				offset_dist = np.dot(offset, slab.get_cell())
				dist = np.linalg.norm(slab.positions[i] + offset_dist - slab.positions[site_id])
				distances += [dist]
		sorted_symbols = np.array([s for _, s in sorted(zip(distances, symbols))])

		# This is a hardcoded  detection of active sites for fcc111 lattice
		if site_type == 'ontop':
			if np.all((sorted_symbols == self.m_element)[:7] == [True, False, False, False, False, False, False]):
				return True
			else:
				return False

		if site_type == 'bridge':
			if np.all((sorted_symbols == self.m_element)[:4] == [True, True, False, False]):
				return True
			elif np.all((sorted_symbols == self.m_element)[:4] == [True, True, True, True]):
				return True
			else:
				return False

		if site_type == 'hollow':
			if np.all((sorted_symbols == self.m_element)[:6] == [True, True, True, False, False, False]):
				return True
			else:
				return False


	def allocate_sites(self, site_energies, all_active_sites, occupied_sites, occupied_dict):
		N_acrolein = len(occupied_sites)

		for s_e, a_s in sorted(zip(site_energies, all_active_sites)):
			if N_acrolein >= self.max_N_acrolein:
				break
			if self.is_site_open(a_s, occupied_sites):
				N_acrolein += 1
				occupied_sites = np.append(occupied_sites,a_s)
				occupied_dict[str(a_s)] = s_e

		assert N_acrolein == len(occupied_sites)
		occupied_sites = np.array(occupied_sites)
		return N_acrolein, occupied_sites, occupied_dict


	def is_site_open(self, site_id, occupied_sites):
		# This corresponds to atom_id of the site in layer1_slab
		if site_id in occupied_sites:
			return False
		site_id += self.slab_size[0]*self.slab_size[1]
		indices, offsets = self.occupied_site_nl.get_neighbors(site_id)
		for i in indices:
			if int(i - self.slab_size[0]*self.slab_size[1]) in occupied_sites:
				return False
		return True




	def get_site_energies(self, sites):
		N_metal = np.prod(self.slab_size)
		all_energy = np.array(())
		for s in sites:
			s_p = self.site_positions[s]
			indices, offsets = self.fp_nl.get_neighbors(s+N_metal)
			indices = indices[indices<N_metal]
			indices = np.append(indices,N_metal)
			temp_slab = self.slab + Atom('Pd',position=s_p)
			energy= cal_energy_ads(model_dir = self.site_model_dir,
									atoms = temp_slab[indices],
									cal_list = [len(indices)-1])
			all_energy = np.append(all_energy, energy)
		return all_energy





def cal_energy_ads(model_dir, atoms, cal_list = None):
	sym_params = pickle.load(open(model_dir+"/sym_params.sav", "rb" ))
	[kmeans,energy_labels] = pickle.load(open(model_dir+"/kmeans_models.sav", "rb" ))
	[Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, elements, weights, element_energy]=sym_params
	params_set = set_sym(elements = elements,
						Gs = Gs,
						cutoff = cutoff,
						g2_etas = g2_etas,
						g2_Rses = g2_Rses,
						g4_etas = g4_etas,
						g4_zetas = g4_zetas,
						g4_lambdas = g4_lambdas,
						weights = weights)

	if cal_list is None:
		N_atoms = len(atoms)
	else:
		N_atoms = len(cal_list)
	nelem = len(elements)
	N_sym = params_set[elements[0]]['num']

	data = cal_fp_only(atoms, elements, params_set, cal_list = cal_list)
	fps = data['x']
	fp = torch.zeros((N_atoms,N_sym))
	elements_num = torch.tensor([atomic_numbers[ele] for ele in elements])
	atom_idx = data['atom_idx'] - 1

	a_num = elements_num[atom_idx]
	atom_numbers = a_num.repeat_interleave(nelem).view(len(a_num),nelem)

	# change to float for pytorch to be able to run without error
	if cal_list is not None:
		e_mask = (atom_numbers == elements_num).float()[cal_list]
		atom_idx = atom_idx[cal_list]
	else:
		e_mask = (atom_numbers == elements_num).float()
	fp_track = [0]*nelem
	if element_energy is not None:
		element_energy = torch.sum(element_energy * e_mask)

	for i,idx in enumerate(atom_idx):
		ele = elements[idx]
		fp[i,:] = torch.tensor(fps[ele][fp_track[idx],:]).float()
		fp_track[idx] += 1
	fp = np.round(fp.data.numpy()[0],6).reshape(1,-1)
	E_out = energy_labels[kmeans.predict(fp)][0]
	return E_out
