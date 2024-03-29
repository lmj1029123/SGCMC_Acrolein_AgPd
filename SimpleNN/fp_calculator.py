from _libsymf import lib, ffi
from utils import _gen_2Darray_for_ffi
import numpy as np
from utils.mpiclass import DummyMPI, MPI4PY
import torch

def _read_params(filename):
	params_i = list()
	params_d = list()
	with open(filename, 'r') as fil:
		for line in fil:
			tmp = line.split()
			params_i += [list(map(int,   tmp[:3]))]
			params_d += [list(map(float, tmp[3:]))]

	params_i = np.asarray(params_i, dtype=np.intc, order='C')
	params_d = np.asarray(params_d, dtype=np.float64, order='C')

	return params_i, params_d

def calculate_fp(atoms, elements, params_set, cal_list=None):
	"""
		atoms: ase Atoms class
		symbols: list of unique elements in atoms
		calc_mask: decide whether the fp and dfpdX of the atom will be calculated
	"""
	is_mpi = False

	try:
		import mpi4py
	except ImportError:
		comm = DummyMPI()
	else:
		if is_mpi:
			comm = MPI4PY()
		else:
			comm = DummyMPI()

	cart = np.copy(atoms.get_positions(wrap=True), order='C')
	scale = np.copy(atoms.get_scaled_positions(), order='C')
	cell = np.copy(atoms.cell, order='C')

	cart_p  = _gen_2Darray_for_ffi(cart, ffi)
	scale_p = _gen_2Darray_for_ffi(scale, ffi)
	cell_p  = _gen_2Darray_for_ffi(cell, ffi)

	atom_num = len(atoms.positions)
	symbols = np.array(atoms.get_chemical_symbols())
	atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
	type_num = dict()
	type_idx = dict()

	# This is the list of atoms that will calculate the fingerprints
	if cal_list is not None:
		cal_mask = np.zeros(len(atoms), dtype = np.intc) == 1
		cal_mask[cal_list] = True
	else:
		cal_mask = np.ones(len(atoms), dtype = np.intc)==1
	#cal_mask_p = ffi.cast("int *", cal_mask.ctypes.data)

	for j,jtem in enumerate(elements):
		# tmp0 is for counting the neighbors, we shou;d not modify it
		tmp0 = (symbols==jtem)
		# tmp is for counting atoms to be calculated.
		tmp = (symbols==jtem)*cal_mask
		atom_i[tmp0] = j+1
		type_num[jtem] = np.sum(tmp).astype(np.int64)
		# if atom indexs are sorted by atom type,
		# indexs are sorted in this part.
		# if not, it could generate bug in training process for force training
		type_idx[jtem] = np.arange(atom_num)[tmp]
	atom_i_p = ffi.cast("int *", atom_i.ctypes.data)



	res = dict()
	res['x'] = dict()
	res['dx'] = dict()
	res['params'] = dict()
	res['N'] = type_num
	res['tot_num'] = np.sum(list(type_num.values()))
	res['partition'] = np.ones([res['tot_num']]).astype(np.int32)

	res['atom_idx'] = atom_i
	res['cal_list'] = cal_list

	for j,jtem in enumerate(elements):
		q = type_num[jtem] // comm.size
		r = type_num[jtem] %  comm.size

		begin = comm.rank * q + min(comm.rank, r)
		end = begin + q
		if r > comm.rank:
			end += 1

		cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')

		#print(cal_atoms)
		#cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')
		cal_num = len(cal_atoms)
		cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

		x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
		dx = np.zeros([cal_num, atom_num * params_set[jtem]['num'] * 3], dtype=np.float64, order='C')

		#weights_p = ffi.cast("double **", params_set[jtem]['weights'].ctypes.data)
		weight_id_p = ffi.cast("int *", params_set[jtem]['weight_id'].ctypes.data)

		x_p = _gen_2Darray_for_ffi(x, ffi)
		dx_p = _gen_2Darray_for_ffi(dx, ffi)
		errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
						 atom_i_p, atom_num, cal_atoms_p, cal_num, \
						 params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
						 x_p, dx_p, params_set[jtem]['weights_p'], weight_id_p)
		comm.barrier()
		errnos = comm.gather(errno)
		errnos = comm.bcast(errnos)

		if isinstance(errnos, int):
			errnos = [errno]

		for errno in errnos:
			if errno == 1:
				err = "Not implemented symmetry function type."
				if comm.rank == 0:
					self.parent.logfile.write("\nError: {:}\n".format(err))
				raise NotImplementedError(err)
			elif errno == 2:
				err = "Zeta in G4/G5 must be greater or equal to 1.0."
				if comm.rank == 0:
					self.parent.logfile.write("\nError: {:}\n".format(err))
				raise ValueError(err)
			else:
				assert errno == 0


		if type_num[jtem] != 0:
			res['x'][jtem] = np.array(comm.gather(x, root=0))
			res['dx'][jtem] = np.array(comm.gather(dx, root=0))

			if comm.rank == 0:
				res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], params_set[jtem]['num']])
				res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).\
									reshape([type_num[jtem], params_set[jtem]['num'], atom_num, 3])
				res['partition_'+jtem] = np.ones([type_num[jtem]]).astype(np.int32)
		else:
			res['x'][jtem] = np.zeros([0, params_set[jtem]['num']])
			res['dx'][jtem] = np.zeros([0, params_set[jtem]['num'], atom_num, 3])
			res['partition_'+jtem] = np.ones([0]).astype(np.int32)
		res['params'][jtem] = params_set[jtem]['total']
	return res



def set_sym(elements, Gs, cutoff, g2_etas=None, g2_Rses=None, g4_etas=None, g4_zetas=None, g4_lambdas=None, weights = None):
	"""
	specify symmetry function parameters for each element
	parameters for each element contain:
	integer parameters: [which sym func, surrounding element 1, surrounding element 1]
						surrouding element starts from 1. For G2 sym func, the third
						element is 0. For G4 and G5, the order of the second and the
						third element does not matter.
	double parameters:  [cutoff radius, 3 sym func parameters]
						for G2: eta, Rs, dummy
						for G4 and G5: eta, zeta, lambda
	"""

	# specify all elements in the system
	params_set = dict()
	ratio = 36  # can be changed to fit the values from different packages
	for item in elements:
		params_set[item] = dict()
		int_params = []
		double_params = []
		weight_id = []
		for G in Gs:
			if G == 2:
				weight_id += [-1 for el1 in range(1, len(elements)+1)
										   for g2_eta in g2_etas
										   for g2_Rs in g2_Rses]
				int_params += [[G, el1, 0] for el1 in range(1, len(elements)+1)
										   for g2_eta in g2_etas
										   for g2_Rs in g2_Rses]
				double_params += [[cutoff, g2_eta/ratio, g2_Rs, 0] for el1 in range(1, len(elements)+1)
																   for g2_eta in g2_etas
																   for g2_Rs in g2_Rses]
			elif G == 22:
				for w_id, _ in enumerate(weights):
					weight_id += [w_id for g2_eta in g2_etas
										 for g2_Rs in g2_Rses]
					int_params += [[G, 0, 0] for g2_eta in g2_etas
											 for g2_Rs in g2_Rses]
					double_params += [[cutoff, g2_eta/ratio, g2_Rs, 0] for g2_eta in g2_etas
																	   for g2_Rs in g2_Rses]
			elif G ==24 or G==25:
				for w_id, _ in enumerate(weights):
					weight_id += [w_id for g4_eta in g4_etas
										 for g4_zeta in g4_zetas
										 for g4_lambda in g4_lambdas]
					int_params += [[G, 0, 0] for g4_eta in g4_etas
											 for g4_zeta in g4_zetas
											 for g4_lambda in g4_lambdas]
					double_params += [[cutoff, g4_eta/ratio, g4_zeta, g4_lambda]
												 for g4_eta in g4_etas
												 for g4_zeta in g4_zetas
												 for g4_lambda in g4_lambdas]
			else:
				weight_id += [-1 for el1 in range(1, len(elements)+1)
								   for el2 in range(el1, len(elements)+1)
							   	   for g4_eta in g4_etas
								   for g4_zeta in g4_zetas
								   for g4_lambda in g4_lambdas]
				int_params += [[G, el1, el2] for el1 in range(1, len(elements)+1)
											 for el2 in range(el1, len(elements)+1)
											 for g4_eta in g4_etas
											 for g4_zeta in g4_zetas
											 for g4_lambda in g4_lambdas]
				double_params += [[cutoff, g4_eta/ratio, g4_zeta, g4_lambda]
											 for el1 in range(1, len(elements)+1)
											 for el2 in range(el1, len(elements)+1)
											 for g4_eta in g4_etas
											 for g4_zeta in g4_zetas
											 for g4_lambda in g4_lambdas]


		params_set[item]['i'] = np.array(int_params, dtype=np.intc)
		params_set[item]['d'] = np.array(double_params, dtype=np.float64)
		params_set[item]['ip'] = _gen_2Darray_for_ffi(params_set[item]['i'], ffi, "int")
		params_set[item]['dp'] = _gen_2Darray_for_ffi(params_set[item]['d'], ffi)
		params_set[item]['total'] = np.concatenate((params_set[item]['i'], params_set[item]['d']), axis=1)
		params_set[item]['num'] = len(params_set[item]['total'])
		if weights is not None:
			params_set[item]['weights'] = np.array(weights, dtype=np.float64)
			params_set[item]['weights_p'] = _gen_2Darray_for_ffi(params_set[item]['weights'], ffi)
			params_set[item]['weight_id'] = np.array(weight_id, dtype=np.intc)
		else:
			#print('Warning! Did not assign weights. If you are using weighted symmetry functions, the weights are automatically assinged to be 1 for each element!')
			params_set[item]['weights'] = np.ones(len(elements), dtype=np.float64).reshape((1,-1))
			params_set[item]['weights_p'] = _gen_2Darray_for_ffi(params_set[item]['weights'], ffi)
			params_set[item]['weight_id'] = -np.ones(params_set[item]['num'], dtype=np.intc)

	return params_set




def cal_fp_only(atoms, elements, params_set, cal_list= None):
	"""
		atoms: ase Atoms class
		symbols: list of unique elements in atoms
	"""
	is_mpi = False

	try:
		import mpi4py
	except ImportError:
		comm = DummyMPI()
	else:
		if is_mpi:
			comm = MPI4PY()
		else:
			comm = DummyMPI()

	cart = np.copy(atoms.get_positions(wrap=True), order='C')
	scale = np.copy(atoms.get_scaled_positions(), order='C')
	cell = np.copy(atoms.cell, order='C')

	cart_p  = _gen_2Darray_for_ffi(cart, ffi)
	scale_p = _gen_2Darray_for_ffi(scale, ffi)
	cell_p  = _gen_2Darray_for_ffi(cell, ffi)

	atom_num = len(atoms.positions)
	symbols = np.array(atoms.get_chemical_symbols())
	atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
	type_num = dict()
	type_idx = dict()


	#test
	#cal_list = [0,1,3]

	# This is the list of atoms that will calculate the fingerprints
	if cal_list is not None:
		cal_mask = np.zeros(len(atoms), dtype = np.intc) == 1
		cal_mask[cal_list] = True
	else:
		cal_mask = np.ones(len(atoms), dtype = np.intc)==1
	#cal_mask_p = ffi.cast("int *", cal_mask.ctypes.data)

	for j,jtem in enumerate(elements):
		# tmp0 is for counting the neighbors, we shou;d not modify it
		tmp0 = (symbols==jtem)
		# tmp is for counting atoms to be calculated.
		tmp = (symbols==jtem)*cal_mask
		atom_i[tmp0] = j+1
		type_num[jtem] = np.sum(tmp).astype(np.int64)
		# if atom indexs are sorted by atom type,
		# indexs are sorted in this part.
		# if not, it could generate bug in training process for force training
		type_idx[jtem] = np.arange(atom_num)[tmp]
	atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

	res = dict()
	res['x'] = dict()
	res['dx'] = dict()
	res['cal_list'] = cal_list
	res['atom_idx'] = atom_i

	for j,jtem in enumerate(elements):
		q = type_num[jtem] // comm.size
		r = type_num[jtem] %  comm.size

		begin = comm.rank * q + min(comm.rank, r)
		end = begin + q
		if r > comm.rank:
			end += 1

		cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')
		cal_num = len(cal_atoms)
		cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

		x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
		dx = np.zeros([cal_num, atom_num * params_set[jtem]['num'] * 3], dtype=np.float64, order='C')

		weight_id_p = ffi.cast("int *", params_set[jtem]['weight_id'].ctypes.data)

		x_p = _gen_2Darray_for_ffi(x, ffi)
		dx_p = _gen_2Darray_for_ffi(dx, ffi)
		errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
						 atom_i_p, atom_num, cal_atoms_p, cal_num, \
						 params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
						 x_p, dx_p, params_set[jtem]['weights_p'], weight_id_p)
		comm.barrier()
		errnos = comm.gather(errno)
		errnos = comm.bcast(errnos)

		if isinstance(errnos, int):
			errnos = [errno]

		for errno in errnos:
			if errno == 1:
				err = "Not implemented symmetry function type."
				if comm.rank == 0:
					self.parent.logfile.write("\nError: {:}\n".format(err))
				raise NotImplementedError(err)
			elif errno == 2:
				err = "Zeta in G4/G5 must be greater or equal to 1.0."
				if comm.rank == 0:
					self.parent.logfile.write("\nError: {:}\n".format(err))
				raise ValueError(err)
			else:
				assert errno == 0


		if type_num[jtem] != 0:
			res['x'][jtem] = np.array(comm.gather(x, root=0))

			if comm.rank == 0:
				res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], params_set[jtem]['num']])
		else:
			res['x'][jtem] = np.zeros([0, params_set[jtem]['num']])
	return res
