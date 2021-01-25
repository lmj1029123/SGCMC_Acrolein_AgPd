import sys
sys.path.append("../SimpleNN")
sys.path.append("../ML_Models")

import torch
from ase.build import fcc111, bulk
from ase.io.trajectory import Trajectory
from ase.units import kB
import numpy as np
import random
from fp_calculator import set_sym, cal_fp_only
from time import time
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from ase import Atoms
from ase.data import atomic_numbers
from preprocess import cal_energy
import pickle
import json
import os



seed = int(sys.argv[1])
n_Pd = int(sys.argv[2])
d_cp = float(sys.argv[3])
T = int(sys.argv[4])
model_seed = int(sys.argv[5])
np.random.seed(seed)



bulk_model_dir = f'../ML_Models/AgPd_bulk_master_{model_seed}'
MC_model_path = f'AgPd_bulk_MC_{model_seed}_lc'

if os.path.exists(MC_model_path):
    print('Path exists')
else:
    os.mkdir(MC_model_path)



bulk_model = torch.load(bulk_model_dir + '/best_model')
sym_params = pickle.load(open(bulk_model_dir+"/sym_params.sav", "rb" ))
[Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, elements, weights, element_energy]=sym_params
params_set = set_sym(elements, Gs, cutoff,
		     g2_etas=g2_etas, g2_Rses=g2_Rses,
		     g4_etas=g4_etas, g4_zetas = g4_zetas,
		     g4_lambdas= g4_lambdas, weights=weights)



# set parameters for the bulk
size = (10, 10, 10)
n_total = np.prod(size)
x_Pd = n_Pd/n_total

epochs = 100*n_total

lc = 4.163

lc_Pd = 3.956
lc_Ag = 4.163

traj_name = f'AgPd_bulk_{d_cp:.2f}_{T}K_{n_Pd}_{seed}'
#traj = Trajectory(MC_model_path+'/'+traj_name+'.traj', 'w')
logfile = open(MC_model_path+'/'+traj_name+'log.txt','w+')
result = {}
result['size'] = size
result['Pd'] = {}
result['energy'] = {}


# create bulk
atoms = bulk('Ag', 'fcc', a=lc).repeat(size)


ids = np.arange(len(atoms))
np.random.shuffle(ids)
chem_symbols = ['Ag']*n_total  # used to create slab only
for i in ids[:n_Pd]:
    chem_symbols[i] = 'Pd' 
atoms.set_chemical_symbols(chem_symbols)



nl1 = NeighborList([cutoff/2]*len(atoms),
                  skin=0.0, bothways=True,
                  self_interaction=True,
                  primitive=NewPrimitiveNeighborList)
nl1.update(atoms)

nl2 = NeighborList([cutoff]*len(atoms),
                  skin=0.0, bothways=True,
                  self_interaction=True,
                  primitive=NewPrimitiveNeighborList)
nl2.update(atoms)



E = []

E += [cal_energy(bulk_model, sym_params, params_set, atoms)]


success = 0
attempt = 0

for epo in range(epochs):
    #ta = time()
    ind =random.sample(range(n_total),1)[0]
    t1 = time()
    indices0, _ = nl1.get_neighbors(ind)
    indices1, _ = nl2.get_neighbors(ind)
    cal_list = np.array([list(indices1).index(i) for i in indices0])
    E1 = cal_energy(bulk_model, sym_params, params_set, atoms[indices1], cal_list)

    temp_atoms = atoms.copy()
    if temp_atoms[ind].symbol == 'Ag':
        temp_atoms[ind].symbol = 'Pd'
        label = 'Ag2Pd'
    elif temp_atoms[ind].symbol == 'Pd':
        temp_atoms[ind].symbol = 'Ag'
        label = 'Pd2Ag'
        
  
    E2 = cal_energy(bulk_model, sym_params, params_set, temp_atoms[indices1], cal_list)
    dE = E2- E1
    if label == 'Ag2Pd':
        cp = d_cp
        n_Pd_temp = n_Pd + 1
    elif label == 'Pd2Ag':
        cp = -d_cp
        n_Pd_temp = n_Pd - 1
        
    if dE - cp < 0:
        atoms = temp_atoms
        E += [E[-1]+dE]
        success += 1
        n_Pd = n_Pd_temp
        logfile.write(f'{E[-1]}\n')
        symbols = atoms.get_chemical_symbols()
        result['Pd'][f'{success}'] = [i for i, x in enumerate(symbols) if x == 'Pd']
        result['energy'][f'{success}'] = float(E[-1])
    elif np.exp(-(dE - cp) / (kB * T)) > np.random.rand():
        atoms = temp_atoms
        E += [E[-1]+dE]
        success += 1
        n_Pd = n_Pd_temp
        logfile.write(f'{E[-1]}\n')
        symbols = atoms.get_chemical_symbols()
        result['Pd'][f'{success}'] = [i for i, x in enumerate(symbols) if x == 'Pd']
        result['energy'][f'{success}'] = float(E[-1])
        
    attempt += 1
    t2 = time()
    logfile.write(f'{t2-t1}, {success}/{attempt}, {n_Pd/n_total}\r\n')
    logfile.flush()

    x_Pd_temp = n_Pd / n_total
    if (x_Pd_temp - x_Pd) > 0.05:
        x_Pd = x_Pd_temp
        lc = x_Pd * lc_Pd + (1 - x_Pd) * lc_Ag
        chem_symbols = atoms.get_chemical_symbols()
        atoms = bulk('Ag', 'fcc', a=lc).repeat(size)
        atoms.set_chemical_symbols(chem_symbols)
        nl1 = NeighborList([cutoff/2]*len(atoms),
                           skin=0.0, bothways=True,
                           self_interaction=True,
                           primitive=NewPrimitiveNeighborList)
        nl1.update(atoms)
        
        nl2 = NeighborList([cutoff]*len(atoms),
                           skin=0.0, bothways=True,
                           self_interaction=True,
                           primitive=NewPrimitiveNeighborList)
        nl2.update(atoms)
        E += [cal_energy(bulk_model, sym_params, params_set, atoms)]
        logfile.write('change lc\n')
        


    if success % 100 == 0:
        with open(MC_model_path+'/'+traj_name+'_result.json', 'w') as outfile:
            json.dump(result, outfile)
with open(MC_model_path+'/'+traj_name+'_result.json', 'w') as outfile:
    json.dump(result, outfile)
logfile.close()
    
  
