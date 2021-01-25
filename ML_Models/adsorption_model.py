import sys
sys.path.append("../SimpleNN")
sys.path.append("../Utils")

from fp_calculator import set_sym, calculate_fp
import os
import shutil
from ase.db import connect
from sklearn.cluster import KMeans
import numpy as np
import torch
from ase.data import atomic_numbers
from ContextManager import cd
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

def get_fp(atoms,cal_list= [36]):
	
	data = calculate_fp(atoms, elements, params_set, cal_list)
	fps = data['x']
	dfpdXs = data['dx']
	if cal_list is not None:
		atoms = atoms[cal_list]
	N_atoms= len(atoms)
	fp = torch.zeros((N_atoms,N_sym))
	dfpdX = torch.zeros((N_atoms, N_sym, N_atoms, 3))

	elements_num = torch.tensor([atomic_numbers[ele] for ele in elements])
	atom_idx = data['atom_idx'] - 1
	if cal_list is not None:
		atom_idx = atom_idx[cal_list]
	a_num = elements_num[atom_idx]
	atom_numbers = a_num.repeat_interleave(nelem).view(len(a_num),nelem)

	# change to float for pytorch to be able to run without error
	e_mask = (atom_numbers == elements_num).float()

	fp_track = [0]*nelem
	for i,idx in enumerate(atom_idx):
		ele = elements[idx] 
		fp[i,:] = torch.tensor(fps[ele][fp_track[idx],:]).float()
		if cal_list is None:
			dfpdX[i,:,:,:] = torch.tensor(dfpdXs[ele][fp_track[idx],:,:,:]).float()
		fp_track[idx] += 1
	return fp


if __name__ == '__main__':
	Gaussian_Noise = False
	seed = 1
	if Gaussian_Noise:
		np.random.seed(seed)
		noise_std = 0.05
		seed = 11
	
	db = connect('../db/AgPd_acrolein_334_pseudo.db')
	N_clusters = db.count()
	Name = f'acrolein_adsorption_{seed}'
	if not os.path.exists(Name):
		os.makedirs(Name)
	elements = ['Pd','Ag']
	nelem = len(elements)

	# This is the energy of the metal in its ground state structure
	#if you don't know the energy of the ground state structure,
	# you can set it to None
	element_energy = None
	weights = None
	
	Gs = [2,4]
	cutoff = 6.5
	g2_etas = [0.05, 4.0, 20.0]
	g2_Rses = [0.0]
	g4_etas=[0.05]
	g4_zetas=[1.0, 4.0]
	g4_lambdas=[-1.0, 1.0]
	sym_params = [Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, elements, weights, element_energy]
	params_set = set_sym(elements, Gs, cutoff,
		g2_etas=g2_etas, g2_Rses=g2_Rses,
		g4_etas=g4_etas, g4_zetas = g4_zetas,
		g4_lambdas= g4_lambdas, weights=weights)
	N_sym = params_set[elements[0]]['num']
	acrolein_energy = 47.3357
	
	all_E = []
	all_fp = []
	ID = []
	for row in db.select():
		ID += [row.id]
		slab = row.toatoms()
		slab[-1].symbol = 'Pd'
		if Gaussian_Noise:
			all_E += [row.energy+acrolein_energy+np.random.normal(0,noise_std,1)]
		else:
			all_E += [row.energy+acrolein_energy]
		fp = np.round(get_fp(slab).data.numpy()[0],6)
		all_fp += [fp]

	X = np.array(all_fp)
	y = np.array(all_E)
	kmeans = KMeans(n_clusters=N_clusters, random_state=0).fit(X)
	energy_labels = np.zeros((N_clusters))
	ID_labels = np.zeros((N_clusters))
	E_predict = np.zeros((len(all_E)))
	fp_labels = np.zeros((N_clusters,len(fp)))
	for i, E in enumerate(all_E):
		label = kmeans.labels_[i]
		if E < energy_labels[label]:
			energy_labels[label] = E
			ID_labels[label] = i+1
			fp_labels[label] = all_fp[i]
	with cd(Name):
		pickle.dump(sym_params, open("sym_params.sav", "wb"))
		pickle.dump([kmeans,energy_labels],open("kmeans_models.sav","wb")) 

		