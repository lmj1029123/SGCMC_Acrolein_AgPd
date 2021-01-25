import sys
sys.path.append("../SimpleNN")
sys.path.append("../Utils")

import os
import shutil
from ase.db import connect
import torch
from ContextManager import cd
from preprocess import  get_scaling, snn2sav
from fp_calculator import set_sym
import pickle
import numpy as np
from SNN import SingleNNTrainer









if __name__ == '__main__':
    device=torch.device('cpu')
    is_force = False
    ensemble_training = False

    if ensemble_training == True:
        seed = 1
        n_ensemble = 10
    else:
        seed = 1
        n_ensemble = 1


    E_coeff = 1
    if is_force:
        F_coeff = 1
    else:
        F_coeff = 0

    val_interval = 1
    n_val_stop = 500
    epoch = 20000
    convergence = {'E_cov':0.0005,'F_cov':0.005}

    # NN architectures 
    n_nodes = [30,30]
    activations = [torch.nn.Tanh(), torch.nn.Tanh()]


    # Optimizer parameters
    opt_method = 'lbfgs'
    if opt_method == 'lbfgs':
        history_size = 100
        lr = 1
        max_iter = 10
        line_search_fn = 'strong_wolfe'

    optim_params = {}
    optim_params['opt_method'] = opt_method
    optim_params['history_size'] = history_size
    optim_params['lr'] = lr
    optim_params['max_iter'] = max_iter
    optim_params['line_search_fn'] = line_search_fn



    Name = f'AgPd_bulk_master_{seed}'
    if not os.path.exists(Name):
        os.makedirs(Name)


    # This database contains data with less than 50% Pd
    dbfile = f'../db/AgPd_bulk_master.db'
    db = connect(dbfile)

    elements = ['Pd','Ag']
    #elements = ['Pd','Ag']
    nelem = len(elements)

    # This is the energy of the metal in its ground state structure
    #if you don't know the energy of the ground state structure,
    # you can set it to None
    element_energy = torch.tensor([-5.1674,-2.8316])
    weights = None
                   


    Gs = [2,4]
    cutoff = 6.0
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

    data_dict = snn2sav(db, elements, params_set, element_energy=element_energy,cal_list = None) 


    train_ids = []
    val_ids = []
    test_ids = []
    np.random.seed(0)
    for row in db.select():
        if np.random.uniform(0,1,1) < 0.1:
            test_ids += [row.id]

    np.random.seed(seed)
    for row in db.select():
        i = row.id
        if i in test_ids:
            pass
        else:
            if np.random.uniform(0,1,1) < 8/9:
                train_ids += [i]
            else:
                val_ids += [i] 

    with cd(Name):
        pickle.dump(sym_params, open("sym_params.sav", "wb")) 
        train_dict = dict((i, data_dict[i]) for i in train_ids)
        val_dict = dict((i, data_dict[i]) for i in val_ids)

        model_path = f'best_model'
        logfile = open(f'log.txt', 'w+')   
        scaling = get_scaling(train_dict)   
        model_trainer = SingleNNTrainer(model_path, scaling , N_sym, n_nodes, activations, nelem, optim_params, device) 
        model_trainer.train(train_dict, val_dict, E_coeff, F_coeff, epoch, val_interval, n_val_stop, convergence, is_force ,logfile)
