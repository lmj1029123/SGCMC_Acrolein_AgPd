import sys

sys.path.append("../SimpleNN")
import torch
from ase.db import connect
import numpy as np
from ase import Atom
from ase.data import atomic_numbers
import pickle
from fp_calculator import set_sym, cal_fp_only, calculate_fp
import hashlib


def snn2sav(db, elements, params_set, element_energy=None, cal_list=None):
    """The energy is eV/atom
    The dfpdX is /A"""
    
    nelem = len(elements)
    N_sym = params_set[elements[0]]['num']
    data_dict = {}
    for row in db.select():
        i1 = row.id
        atoms = row.toatoms()
        atoms.set_constraint()
        
        energy = torch.tensor(row.energy).float()
        try:
            forces = torch.tensor(row.forces).float()
        except:
            forces = 0
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
            
                
           
        if element_energy is not None:
            energy -= torch.sum(e_mask * element_energy.float())

        energy = energy/N_atoms
            
        data_dict[i1]={'fp':fp,'dfpdX':dfpdX,'e_mask':e_mask,'e':energy,'f':forces}


    print('preprocess done')
    return data_dict





    

def get_scaling(train_dict, fp_scale_method='min_max', e_scale_method='min_max'):
    train_ids = train_dict.keys()
    if fp_scale_method == 'min_max':
        all_fp = torch.tensor(())
        for ID in train_ids:
            fp = train_dict[ID]['fp']
            all_fp = torch.cat((all_fp, fp), dim=0)
        gmax = torch.max(all_fp, 0)[0]
        gmin = torch.min(all_fp, 0)[0]

    # This calculate the min and max energy/atom
    if e_scale_method == 'min_max':
        all_e = torch.tensor(())
        for ID in train_ids:
            e = train_dict[ID]['e'].view(1, 1)
            all_e = torch.cat((all_e, e), dim=0)
        emax = torch.max(all_e)
        emin = torch.min(all_e)
    return {'gmax':gmax, 'gmin':gmin, 'emax':emax, 'emin':emin}




def cal_energy(model_dir, atoms, cal_list = None):
    sym_params = pickle.load(open(model_dir+"/sym_params.sav", "rb" ))
    [Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, elements, weights, element_energy]=sym_params
    params_set = set_sym(elements, Gs, cutoff,
        g2_etas=g2_etas, g2_Rses=g2_Rses,
        g4_etas=g4_etas, g4_zetas = g4_zetas,
        g4_lambdas= g4_lambdas, weights=weights)

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

    model = torch.load(model_dir + '/best_model')
    scaling = model.scaling
    gmin = scaling['gmin']
    gmax = scaling['gmax']
    emin = scaling['emin']
    emax = scaling['emax']
    sfp = (fp - gmin) / (gmax - gmin+1e-5)
    Atomic_Es = model(sfp)
    E_predict = torch.sum(torch.sum(Atomic_Es * e_mask,
                                    dim = 1)*(emax-emin)+emin,dim=0)
    if element_energy is not None:
        return (E_predict + element_energy).data.numpy()
    else:
        return (E_predict).data.numpy()
