import sys
sys.path.append("../SimpleNN")

from fp_calculator import set_sym, calculate_fp


from ase.db import connect
from sklearn.cluster import KMeans
import numpy as np
import torch
from ase.data import atomic_numbers

import matplotlib.pyplot as plt


def get_fp(atoms):
	cal_list = [36]
	data = calculate_fp(atoms, elements, params_set, [36])
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
	N_clusters = 286
	rmse = []
	knn_rmse = []
	linear_rmse = []
	seed = 1
	db = connect('../db/AgPd_acrolein_334_pseudo.db')

	elements = ['Pd','Ag']
	acrolein_energy = 47.3357
	nelem = len(elements)

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
	all_E = []
	all_fp = []
	ID = []
	ontop_ID = []
	hollow_ID = []
	bridge_ID = []
	for row in db.select():
		ID += [row.id]
		slab = row.toatoms()
		if slab[27:].get_chemical_symbols().count('Pd') == 1:
			ontop_ID += [row.id-1]
		elif slab[27:].get_chemical_symbols().count('Pd') == 2:
			bridge_ID += [row.id-1]
		else:
			hollow_ID += [row.id-1]
		slab[-1].symbol = 'Pd'
		symbols = slab[27:].get_chemical_symbols()
		all_E += [row.energy+acrolein_energy]
		fp = np.round(get_fp(slab).data.numpy()[0],6)
		all_fp += [fp]

	all_fp = np.array(all_fp)
	all_E = np.array(all_E)
	E_p = []
	all_label = []
	for i_cluster in range(287):
		if i_cluster == 0:
			X = all_fp[1:]
			y = all_E[1:]
		elif i_cluster == 286:
			X = all_fp[:286]
			y = all_E[:286]
		else:
			# print(all_fp[:i_cluster].shape)
			X = np.vstack((all_fp[:i_cluster],all_fp[(i_cluster+1):]))
			# print(X.shape)
			y = np.append(all_E[:i_cluster],all_E[(i_cluster+1):])
		kmeans = KMeans(n_clusters=N_clusters, random_state=0).fit(X)
		energy_labels = np.zeros((N_clusters))
		ID_labels = np.zeros((N_clusters))
		E_predict = np.zeros((len(y)))
		fp_labels = np.zeros((N_clusters,len(fp)))
		for i, E in enumerate(y):
			label = kmeans.labels_[i]
			if E < energy_labels[label]:
				energy_labels[label] = E
				ID_labels[label] = i+1
				fp_labels[label] = all_fp[i]
		E_p += [energy_labels[kmeans.predict(all_fp[i_cluster].reshape(1,-1))]]
		all_label += [list(kmeans.labels_).index(kmeans.predict(all_fp[i_cluster].reshape(1,-1)))+1]
		# print(all_label[-1])
		# print(E_p[-1][0]-all_E[i_cluster])
	E_p = np.array(E_p).reshape(-1)
	all_E = np.array(all_E).reshape(-1)
	rmse = np.mean((E_p-all_E)**2)**0.5
	ontop_rmse = np.mean((E_p[ontop_ID]-all_E[ontop_ID])**2)**0.5
	bridge_rmse = np.mean((E_p[bridge_ID]-all_E[bridge_ID])**2)**0.5
	hollow_rmse = np.mean((E_p[hollow_ID]-all_E[hollow_ID])**2)**0.5
	

	plt.plot(all_E,all_E)
	plt.plot(all_E,all_E+0.05,'k--')
	plt.plot(all_E,all_E-0.05,'k--')
	plt.scatter(all_E[ontop_ID],E_p[ontop_ID],label = f'Ontop, RMSE = {ontop_rmse:.3f}', alpha=0.8)
	plt.scatter(all_E[bridge_ID],E_p[bridge_ID],label = f'Bridge, RMSE = {bridge_rmse:.3f}', alpha=0.8)
	plt.scatter(all_E[hollow_ID],E_p[hollow_ID],label = f'Hollow, RMSE = {hollow_rmse:.3f}', alpha=0.8)
	plt.xlabel('DFT energy (eV)')
	plt.ylabel('Predicted energy (eV)')
	#plt.title(f'RMSE = {rmse:.3f} eV')
	plt.legend()
	plt.savefig('images/leave_one_out.png')
	plt.close()
	plt.hist(abs(all_E-E_p))
	plt.xlabel('Absolute prediction error (eV)')
	plt.ylabel('count')
	plt.title(f'RMSE = {rmse:.3f} eV')
	plt.savefig('images/leav_one_out_hist.png')


	from ase.visualize import view
	plt.clf()
	def on_pick(event):
		atoms1 = db.get(id = event.artist.id).toatoms()
		k = all_label[event.artist.id-1]
		if k< event.artist.id:
			atoms2 = db.get(id = k).toatoms()
		else:
			atoms2 = db.get(id = k+1).toatoms()
		atoms1[-1].symbol = 'O'
		view(atoms1)
		view(atoms2)
	fig, ax = plt.subplots()
	for j,i in enumerate(E_p):
		artist = ax.plot(all_E[j], E_p[j],'ko' ,picker=1,markersize=1,color='b')[0]
		artist.id = j+1
	plt.plot(all_E,all_E)
	fig.canvas.callbacks.connect('pick_event', on_pick)
	plt.show()

