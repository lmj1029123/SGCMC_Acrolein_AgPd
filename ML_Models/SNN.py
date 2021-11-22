import sys
sys.path.append("../SimpleNN")
"""SingleNN potential."""
from fp_calculator import set_sym, calculate_fp
from NN import MultiLayerNet

import torch
from torch.autograd import grad
from Batch import batch_pad
import time
import numpy as np

import pickle
from ase.data import chemical_symbols, atomic_numbers
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)
import os


class SingleNN(Calculator):

    implemented_properties = ['energy', 'energies', 'forces']

    def __init__(self, model_path,cal_list = None,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.model_path = model_path
        self.cal_list = cal_list

    def initialize(self, atoms):
        self.numbers = atoms.get_atomic_numbers()
        self.energies = np.empty(len(atoms))
        self.forces = np.empty((len(atoms), 3))


    def calculate(self, atoms=None, properties=['energy'],system_changes=all_changes):
        cal_list = self.cal_list
        if os.path.exists(self.model_path+'/best_model'):
            model = torch.load(self.model_path+'/best_model')
            ensemble_training = False
        else:
            ensemble_training = True
            models = []
            ensemble = 0
            end = False
            while end is False:
                if os.path.exists(self.model_path+f'/best_model-{ensemble}'):
                    models += [torch.load(self.model_path+f'/best_model-{ensemble}')]
                    ensemble += 1
                else:
                    end = True



        sym_params = pickle.load(open(self.model_path+"/sym_params.sav", "rb" ))
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
        data = calculate_fp(atoms, elements, params_set, cal_list = cal_list)
        fps = data['x']
        dfpdXs = data['dx']

        fp = torch.zeros((N_atoms,N_sym))
        dfpdX = torch.zeros((N_atoms, N_sym, N_atoms, 3))
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
            if cal_list is None:
                dfpdX[i,:,:,:] = torch.tensor(dfpdXs[ele][fp_track[idx],:,:,:]).float()
            fp_track[idx] += 1
        fp.requires_grad = True

        if ensemble_training:
            scaling = models[0].scaling
        else:
            scaling = model.scaling

        gmin = scaling['gmin']
        gmax = scaling['gmax']
        emin = scaling['emin']
        emax = scaling['emax']
        eps = 1e-5
        sfp = (fp - gmin) / (gmax - gmin+eps)

        if ensemble_training:
            all_energy = []
            all_forces = []
            for model in models:
                Atomic_Es = model(sfp)
                E_predict = torch.sum(torch.sum(Atomic_Es * e_mask,
                    dim = 1)*(emax-emin)+emin,dim=0)
                dEdfp = grad(E_predict,
                    fp,
                    grad_outputs=torch.ones_like(E_predict),
                    create_graph = True,
                    retain_graph = True)[0].view(1,fp.shape[0]*fp.shape[1])
                dfpdX = dfpdX.view(fp.shape[0]*fp.shape[1],fp.shape[0]*3)
                F_predict = -torch.mm(dEdfp,dfpdX).view(fp.shape[0],3)
                forces = F_predict.data.numpy()
                if element_energy is not None:
                    energy = (E_predict + element_energy).data.numpy()
                else:
                    energy = E_predict.data.numpy()
                all_energy += [energy]
                all_forces += [forces]
            all_energy = np.array(all_energy)
            all_forces = np.array(all_forces)
            ensemble_energy = np.mean(all_energy)
            energy_std = np.std(all_energy)
            ensemble_forces = np.mean(all_forces, axis=0)
            forces_std = np.std(all_forces, axis=0)
            self.energy = ensemble_energy
            self.forces = ensemble_forces
            self.results['energy'] = self.energy
            self.results['free_energy'] = self.energy
            self.results['forces'] = self.forces
            self.results['energy_std'] = energy_std
            self.results['forces_std'] = forces_std
        else:
            Atomic_Es = model(sfp)
            E_predict = torch.sum(torch.sum(Atomic_Es * e_mask,
                dim = 1)*(emax-emin)+emin,dim=0)
            dEdfp = grad(E_predict,
                fp,
                grad_outputs=torch.ones_like(E_predict),
                create_graph = True,
                retain_graph = True)[0].view(1,fp.shape[0]*fp.shape[1])
            dfpdX = dfpdX.view(fp.shape[0]*fp.shape[1],fp.shape[0]*3)
            F_predict = -torch.mm(dEdfp,dfpdX).view(fp.shape[0],3)
            self.forces = F_predict.data.numpy()
            if element_energy is not None:
                self.energy = (E_predict + element_energy).data.numpy()
            else:
                self.energy = E_predict.data.numpy()
            self.results['energy'] = self.energy
            self.results['free_energy'] = self.energy
            self.results['forces'] = self.forces




class SingleNNTrainer(object):
    def __init__(self, model_path, scaling, N_sym, n_nodes, activations, nelem, optim_params, device=torch.device('cpu')):

        self.model = MultiLayerNet(N_sym, n_nodes, activations, nelem, scaling=scaling)
        self.model_path = model_path
        self.opt_method = optim_params['opt_method']
        if self.opt_method == 'lbfgs':
            self.history_size = optim_params['history_size']
            self.lr = optim_params['lr']
            self.max_iter = optim_params['max_iter']
            self.line_search_fn = optim_params['line_search_fn']
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr,
                max_iter=self.max_iter, history_size=self.history_size,
                line_search_fn=self.line_search_fn)
        else:
            print('Optimization method not implemented!')
            raise
        self.device = device
        # self.convergence = hyperparams['convergence']
        # self.n_nodes = n_nodes
        # self.activations = hyperparams['activations']

    def train(self, train_dict, val_dict, E_coeff, F_coeff, epoch, val_interval, n_val_stop, convergence, is_force, logfile):
        device = self.device
        opt_method = self.opt_method
        optimizer = self.optimizer
        model = self.model.to(device)
        model_path = self.model_path




        t0 = time.time()
        SSE = torch.nn.MSELoss(reduction='sum')
        SAE = torch.nn.L1Loss(reduction='sum')
        scaling = self.model.scaling
        gmin = scaling['gmin'].to(device)
        gmax = scaling['gmax'].to(device)
        emin = scaling['emin'].to(device)
        emax = scaling['emax'].to(device)
        n_val = 0
        E_cov = convergence['E_cov']
        F_cov = convergence['F_cov']
        t_ids = np.array(list(train_dict.keys()))
        batch_info = batch_pad(train_dict,t_ids)
        b_fp = batch_info['b_fp'].to(device)


        if is_force:
            b_dfpdX = batch_info['b_dfpdX'].view(b_fp.shape[0],
                                                 b_fp.shape[1]*b_fp.shape[2],
                                                 b_fp.shape[1]*3,
                                                 ).to(device)
        b_e_mask = batch_info['b_e_mask'].to(device)
        b_fp.requires_grad = True
        eps = 1e-5
        sb_fp = ((b_fp - gmin) / (gmax - gmin + eps))
        N_atoms = batch_info['N_atoms'].view(-1).to(device)
        b_e = batch_info['b_e'].view(-1).to(device)
        b_f = batch_info['b_f'].to(device)



        sb_e = ((b_e - emin) / (emax - emin))
        sb_f = (b_f / (emax - emin))
        t1 = time.time()
        logfile.write(f'Batching takes {t1-t0}.\n')

        v_ids = np.array(list(val_dict.keys()))
        v_batch_info = batch_pad(val_dict,v_ids)
        v_b_fp = v_batch_info['b_fp'].to(device)
        if is_force:
            v_b_dfpdX = v_batch_info['b_dfpdX'].view(v_b_fp.shape[0],
                                                     v_b_fp.shape[1]*v_b_fp.shape[2],
                                                     v_b_fp.shape[1]*3,
                                                     ).to(device)
        v_b_e_mask = v_batch_info['b_e_mask'].to(device)
        v_b_fp.requires_grad = True
        v_sb_fp = ((v_b_fp - gmin) / (gmax - gmin + eps))
        v_N_atoms = v_batch_info['N_atoms'].view(-1).to(device)
        v_b_e = v_batch_info['b_e'].view(-1).to(device)

        v_b_f = v_batch_info['b_f'].to(device)


        v_sb_e = ((v_b_e - emin) / (emax - emin))
        v_sb_f = (v_b_f / (emax - emin))


        if opt_method == 'lbfgs':
            for i in range(epoch):
                def closure():
                    global E_MAE, F_MAE
                    optimizer.zero_grad()
                    Atomic_Es = model(sb_fp)
                    E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
                    if is_force:
                        F_predict = self.get_forces(E_predict, b_fp, b_dfpdX)
                        metrics = self.get_metrics(sb_e, sb_f, N_atoms, t_ids,
                                               E_predict, F_predict, SSE, SAE, scaling, b_e_mask)
                        [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
                        loss = E_coeff * E_loss + F_coeff * F_loss
                    else:
                        metrics =  self.get_metrics(sb_e, None, N_atoms, t_ids,
                                               E_predict, None, SSE, SAE, scaling, b_e_mask)
                        [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
                        loss = E_coeff * E_loss

                    loss.backward(retain_graph=True)
                    return loss

                optimizer.step(closure)


                if i % val_interval == 0:
                    n_val += 1
                    Atomic_Es = model(sb_fp)
                    E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
                    if is_force:
                        F_predict = self.get_forces(E_predict, b_fp, b_dfpdX)
                        metrics =  self.get_metrics(sb_e, sb_f, N_atoms, t_ids,
                                               E_predict, F_predict, SSE, SAE, scaling, b_e_mask)
                        [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
                        loss = E_coeff * E_loss + F_coeff * F_loss
                    else:
                        metrics = self.get_metrics(sb_e, None, N_atoms, t_ids,
                                               E_predict, None, SSE, SAE, scaling, b_e_mask)
                        [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
                        loss = E_coeff * E_loss
                    logfile.write(f'{i}, E_RMSE/atom = {E_RMSE}, F_RMSE = {F_RMSE}, loss={loss}\n')
                    logfile.write(f'{i}, E_MAE/atom = {E_MAE}, F_MAE = {F_MAE}\n')




                    v_Atomic_Es = model(v_sb_fp)
                    v_E_predict = torch.sum(v_Atomic_Es * v_b_e_mask, dim = [1,2])
                    if is_force:
                        v_F_predict = self.get_forces(v_E_predict, v_b_fp, v_b_dfpdX)
                        v_metrics =  self.get_metrics(v_sb_e, v_sb_f, v_N_atoms, v_ids,
                                               v_E_predict, v_F_predict, SSE, SAE, scaling, v_b_e_mask)
                        [v_E_loss, v_F_loss, v_E_MAE, v_F_MAE, v_E_RMSE, v_F_RMSE] = v_metrics
                        v_loss = E_coeff * v_E_loss + F_coeff * v_F_loss
                    else:
                        v_metrics =  self.get_metrics(v_sb_e, None, v_N_atoms, v_ids,
                                                 v_E_predict, None, SSE, SAE, scaling, v_b_e_mask)
                        [v_E_loss, v_F_loss, v_E_MAE, v_F_MAE, v_E_RMSE, v_F_RMSE] = v_metrics
                        v_loss = E_coeff * v_E_loss

                    try:
                        if v_loss < best_v_loss:
                            best_loss = loss
                            best_E_MAE = E_MAE
                            best_F_MAE = F_MAE
                            best_v_loss = v_loss
                            best_v_E_MAE = v_E_MAE
                            best_v_F_MAE = v_F_MAE
                            torch.save(model,model_path)
                            n_val = 1
                    except NameError:
                        best_loss = loss
                        best_E_MAE = E_MAE
                        best_F_MAE = F_MAE
                        best_v_loss = v_loss
                        best_v_E_MAE = v_E_MAE
                        best_v_F_MAE = v_F_MAE
                        torch.save(model,model_path)
                        n_val = 1

                    logfile.write(f'val, E_RMSE/atom = {v_E_RMSE}, F_RMSE = {v_F_RMSE}\n')
                    logfile.write(f'val, E_MAE/atom = {v_E_MAE}, F_MAE = {v_F_MAE}\n')
                    logfile.flush()
                    if n_val > n_val_stop:
                        break

        t2 = time.time()
        logfile.write(f'Training takes {t2-t0}\n')
        logfile.close()
        return [best_loss, best_E_MAE, best_F_MAE, best_v_loss, best_v_E_MAE, best_v_F_MAE]

    def get_forces(self, E_predict, b_fp, b_dfpdX):
        b_dEdfp = grad(E_predict,
                        b_fp,
                        grad_outputs=torch.ones_like(E_predict),
                        create_graph = True,
                        retain_graph = True)[0].view(b_fp.shape[0],1,b_fp.shape[1]*b_fp.shape[2])
        F_predict = - torch.bmm(b_dEdfp,b_dfpdX).view(b_fp.shape[0],b_fp.shape[1],3)
        return F_predict




    def get_metrics(self, sb_e, sb_f, N_atoms, ids, E_predict, F_predict, SSE, SAE, scaling, b_e_mask):

        gmin = scaling['gmin']
        gmax = scaling['gmax']
        emin = scaling['emin']
        emax = scaling['emax']



        E_loss = SSE(sb_e, E_predict / N_atoms) / len(ids)
        E_MAE = SAE(sb_e, E_predict / N_atoms) / len(ids) * (emax - emin)
        E_RMSE = torch.sqrt(E_loss) * (emax - emin)
        if sb_f is None:
            F_loss = 0
            F_MAE = 0
            F_RMSE = 0
        else:
            F_loss = SSE(sb_f, F_predict) / (3 * torch.sum(N_atoms))
            F_MAE = SAE(sb_f, F_predict) / (3 * torch.sum(N_atoms)) * (emax - emin)
            F_RMSE = torch.sqrt(F_loss) * (emax - emin)
            F_max = torch.max(torch.abs(sb_f-F_predict))*(emax-emin)
            print('F_max = ',F_max.data.numpy(), 'eV/A')
        return [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE]


    def evaluate(self, data_dict, E_coeff, F_coeff, is_force):
        device = self.device
        for key in data_dict.keys():
            data_dict[key] = data_dict[key].to(device)

        model_path = self.model_path
        SSE = torch.nn.MSELoss(reduction='sum')
        SAE = torch.nn.L1Loss(reduction='sum')
        model = torch.load(model_path)
        scaling = model.scaling
        gmin = scaling['gmin']
        gmax = scaling['gmax']
        emin = scaling['emin']
        emax = scaling['emax']

        ids = np.array(list(data_dict.keys()))
        batch_info = batch_pad(data_dict,ids)
        b_fp = batch_info['b_fp']

        if is_force:
            b_dfpdX = batch_info['b_dfpdX'].view(b_fp.shape[0],
                                                 b_fp.shape[1]*b_fp.shape[2],
                                                 b_fp.shape[1]*3)
        b_e_mask = batch_info['b_e_mask']
        b_fp.requires_grad = True
        sb_fp = (b_fp - gmin) / (gmax - gmin)
        N_atoms = batch_info['N_atoms'].view(-1)
        b_e = batch_info['b_e'].view(-1)
        b_f = batch_info['b_f']

        sb_e = (b_e - emin) / (emax - emin)
        sb_f = b_f / (emax - emin)


        Atomic_Es = model(sb_fp)
        E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
        if is_force:
            F_predict = get_forces(E_predict, b_fp, b_dfpdX)
            metrics =  get_metrics(sb_e, sb_f, N_atoms, ids,
                                   E_predict, F_predict, SSE, SAE, scaling, b_e_mask)
            [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
            loss = E_coeff * E_loss + F_coeff * F_loss
        else:
            metrics =  get_metrics(sb_e, None, N_atoms, ids,
                                   E_predict, None, SSE, SAE, scaling, b_e_mask)
            [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
            loss = E_coeff * E_loss
        return [loss, E_MAE, F_MAE]
