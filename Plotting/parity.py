import sys

sys.path.append("../SimpleNN")
sys.path.append("../ML_Models")
from preprocess import cal_energy
import matplotlib.pyplot as plt
from ase.db import connect
import numpy as np
import torch
import pickle
from fp_calculator import set_sym, calculate_fp
from ase.visualize import view

model_seed = 1

model_dir1 = f'../ML_Models/AgPd_slab_225_1'
model_dir2 = f'../ML_Models/AgPd_slab_225_2'
model_dir3 = f'../ML_Models/AgPd_slab_225_3'
model_dir4 = f'../ML_Models/AgPd_slab_225_4'
model_dir5 = f'../ML_Models/AgPd_slab_225_5'

dbfile = f'../db/AgPd_slab_225.db'


db = connect(dbfile)


sym_params = pickle.load(open(model_dir1+"/sym_params.sav", "rb" ))
[Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, elements, weights, element_energy]=sym_params
element_energy = element_energy.data.numpy()

dft_E = np.array(())
nn_E1 = np.array(())
nn_E2 = np.array(())
nn_E3 = np.array(())
nn_E4 = np.array(())
nn_E5 = np.array(())

n_Pd_max = 0
for row in db.select():
	atoms = row.toatoms()
	symbols = atoms.get_chemical_symbols()
	n_Pd = symbols.count('Pd')
	n_Ag = symbols.count('Ag')
	N = len(atoms)
	if n_Pd > n_Pd_max:
		n_Pd_max = n_Pd
	dft_E = np.append(dft_E, (row.energy- n_Pd * element_energy[0] - n_Ag * element_energy[1])/N)
	nn_E1 = np.append(nn_E1, (cal_energy(model_dir1, atoms, cal_list = None)- n_Pd * element_energy[0] - n_Ag * element_energy[1])/N)
	# if n_Pd == 0:
	# 	view(atoms)
	nn_E2 = np.append(nn_E2, (cal_energy(model_dir2, atoms, cal_list = None)- n_Pd * element_energy[0] - n_Ag * element_energy[1])/N) 
	nn_E3 = np.append(nn_E3, (cal_energy(model_dir3, atoms, cal_list = None)- n_Pd * element_energy[0] - n_Ag * element_energy[1])/N) 
	nn_E4 = np.append(nn_E4, (cal_energy(model_dir4, atoms, cal_list = None)- n_Pd * element_energy[0] - n_Ag * element_energy[1])/N) 
	nn_E5 = np.append(nn_E5, (cal_energy(model_dir5, atoms, cal_list = None)- n_Pd * element_energy[0] - n_Ag * element_energy[1])/N) 





train_ids = []
val_ids = []
test_ids = []
np.random.seed(0)
for row in db.select():
    if np.random.uniform(0,1,1) < 0.1:
    	test_ids += [row.id-1]

np.random.seed(model_seed)
for row in db.select():
	i = row.id -1
	if i in test_ids:
		pass
	else:
		if np.random.uniform(0,1,1) > 8/9:
			train_ids += [i]
		else:
			val_ids += [i] 





rmse1 = np.mean((dft_E[test_ids]-nn_E1[test_ids])**2)**0.5
rmse2 = np.mean((nn_E1[test_ids]-nn_E2[test_ids])**2)**0.5
rmse3 = np.mean((nn_E1[test_ids]-nn_E3[test_ids])**2)**0.5
rmse4 = np.mean((nn_E1[test_ids]-nn_E4[test_ids])**2)**0.5
rmse5 = np.mean((nn_E1[test_ids]-nn_E5[test_ids])**2)**0.5

print(rmse1)
plt.figure(figsize=(14,7))

plt.subplot(1,2,1)
plt.plot(dft_E,dft_E)
plt.scatter(dft_E[train_ids],nn_E1[train_ids], label = 'train')
plt.scatter(dft_E[val_ids],nn_E1[val_ids],label = 'val')
plt.scatter(dft_E[test_ids],nn_E1[test_ids], label = 'test')
plt.title(f'rmse = {rmse1:.4f} eV/atom')
plt.xlabel('DFT atomization energy (eV/atom)')
plt.ylabel('NN atomization energy (eV/atom)')
plt.legend()

#plt.savefig('NN_DFT_difference.png')

plt.subplot(1,2,2)
plt.plot(nn_E1[test_ids],nn_E1[test_ids])
plt.scatter(nn_E1[test_ids],nn_E2[test_ids], label = f'rmse2 = {rmse2:.4f}')
plt.scatter(nn_E1[test_ids],nn_E3[test_ids], label = f'rmse3 = {rmse3:.4f}')
plt.scatter(nn_E1[test_ids],nn_E4[test_ids], label = f'rmse4 = {rmse4:.4f}')
plt.scatter(nn_E1[test_ids],nn_E5[test_ids], label = f'rmse5 = {rmse5:.4f}')
plt.xlabel('NN atomization energy (eV/atom), seed = 1')
plt.ylabel('NN atomization energy (eV/atom), other seeds')
plt.legend()
plt.savefig('images/slab_225_parity.png')




