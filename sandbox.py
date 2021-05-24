# from maker import *
# from os.path import join as jn


# INPUT_FOLDER = '/home/aria/landslide/MDRUNS/IVAN_IGPS'
# OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/GUIDELINES/SAND'

# name_list = ['apo', 'prfar']
# traj_list = [[jn(INPUT_FOLDER, 'prot_{0}_sim{1}_s10.dcd'.format(name, i)) for i in range(1, 2)] for name in name_list]
# topo_list = [jn(INPUT_FOLDER, 'prot.prmtop')]*2
# output_top_list = [jn(OUTPUT_FOLDER, name) for name in name_list]
# create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=50)
# save_top(traj_list, topo_list, name_list, output_top_list)



import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import time

def avg_sparse_dense(list_of_matrices, size, n_frames=None, size2=None):
    size2 = size if size2==None else size2
    avg = np.zeros((size, size))
    if type(list_of_matrices[0]) == tuple:
        n_frames = np.sum(elt[1] for elt in list_of_matrices) if n_frames==None else n_frames
        for mat, fact in list_of_matrices:
            avg[mat.nonzero()] += mat.data*(fact/n_frames)
    else:
        n_frames = len(list_of_matrices) if n_frames==None else n_frames
        for mat in list_of_matrices:
            avg[mat.nonzero()] += mat.data
    avg = csr_matrix(avg)
    avg /= n_frames
    return avg

def avg_sparse(list_of_matrices, size, n_frames, size2=None):
    size2 = size if size2==None else size2
    avg = lil_matrix((size, size))
    if type(list_of_matrices[0]) == tuple:
        for mat, fact in list_of_matrices:
            avg[mat.nonzero()] += mat.data*(fact/n_frames)
    else:
        for mat in list_of_matrices:
            avg[mat.nonzero()] += mat.data
    avg = csr_matrix(avg)
    avg /= n_frames
    return avg






