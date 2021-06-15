from os import makedirs as mkdir
from os.path import join as jn, isfile, basename
import shutil
import tarfile
import pickle as pkl
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree
import mdtraj as md
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.style.use('default')
import matplotlib as mpl
mpl.rc('figure', fc = 'white')
import seaborn as sns
from Bio.PDB.Polypeptide import aa1, aa3
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
warnings.simplefilter('ignore', PDBConstructionWarning)
three2one = dict(zip(aa3, aa1))
t2o = lambda X: three2one[X] if X in three2one else X[0]
label =  lambda X: t2o(X.name)+str(X.index)
from scipy.sparse import csr_matrix, save_npz, coo_matrix, dok_matrix


def avg_sparse(list_of_matrices, size, matrix_generator=np.zeros, n_frames=None, size2=None):
    size2 = size if size2==None else size2
    if max(size, size2) < 10000:
        avg = matrix_generator((size, size), dtype=np.int)
        if type(list_of_matrices[0]) == tuple:
            n_frames = np.sum(elt[1] for elt in list_of_matrices) if n_frames==None else n_frames
            for mat, fact in list_of_matrices:
                avg[mat.nonzero()] += mat.data*fact
        else:
            n_frames = len(list_of_matrices) if n_frames==None else n_frames
            for mat in list_of_matrices:
                avg[mat.nonzero()] += mat.data
        avg = csr_matrix(avg, dtype=np.float64)
        avg /= n_frames
        return avg
    else:
        return avg_coo(list_of_matrices, size, n_frames=n_frames, size2=size2)

def avg_coo(list_of_matrices, size, n_frames=None, size2=None):
    size2 = size if size2==None else size2
    row, col, data = [], [], []
    if type(list_of_matrices[0]) == tuple:
        n_frames = np.sum(elt[1] for elt in list_of_matrices) if n_frames==None else n_frames
        for mat, fact in list_of_matrices:
            _row, _col = mat.nonzero()
            row += _row.tolist()
            col += _col.tolist() 
            _data = mat.data*fact
            data += _data.tolist()
    else:
        n_frames = len(list_of_matrices) if n_frames==None else n_frames
        for mat in list_of_matrices:
            _row, _col = mat.nonzero()
            row += _row.tolist()
            col += _col.tolist() 
            data += mat.data.tolist()
    avg = coo_matrix((data, (row, col)), shape=(size, size2), dtype=np.float64)
    avg = csr_matrix(avg)
    avg /= n_frames
    return avg

def create_topmat(selection, top, subselection='protein'):
    n_atoms = top.n_atoms
    n_residues = top.subset(top.select(subselection)).n_residues
    selection = selection.replace("not hydrogen", "!(name =~'H.*')")
    indexes = top.select(selection)
    top_mat = dok_matrix((n_atoms, n_residues), dtype=np.bool)
    for atom in top.atoms:
        if atom.index in indexes:
            top_mat[atom.index, atom.residue.index] = 1
    top_mat = csr_matrix(top_mat)
    return top_mat

def save_top(traj_list, topo_list, name_list, output_list, selection='protein'):
    for i in range(len(name_list)):
        t = md.load_frame(traj_list[i][0], 0, top=topo_list[i])
        t = t.atom_slice(t.topology.select(selection))
        topo = t.topology
        pkl.dump(topo, open('{0}.topy'.format(output_list[i]), 'wb'))
        if str(next(topo.residues))[-1] == '0':
            for res in t.topology.residues:
                res.resSeq += 1
        t.save('{0}.pdb'.format(output_list[i]))

def divide_expected(mat1, mat2):
    res = mat1 / mat2
    res[np.isnan(res)] = 0
    return res

def inverse_sparse(mat):
    res = mat.copy()
    mat = csr_matrix(mat)
    mat.eliminate_zeros()
    inv_data = 1. / mat.data
    res[mat.nonzero()] = inv_data
    return csr_matrix(res)



class AtomicNetMaker():
    def __init__(self, trajs, topo=None, selection='protein', cutoff=5, chunk=10000, output=None, all_frames=None, interface=None, save_top=True, ponderate=False):
        """Function creating the atomic contact network with a desired base selection in chunks
        Parameters: traj: str or list of str: path trajectories to load
        topo: str: path of topology to use
        selection: str: base selection on which to compute the atomic network. To save computation time, this should be the 
        smallest selection that includes all the selections in the list.
        """
        #Passing some values to the class
        self.cutoff = cutoff
        self.all_frames = all_frames
        self.ponderate = ponderate
        
        #Forcing basename to eventual output
        if output != None:
            output = output.split('.')[0]

        #Loading first frame to pass topological information
        tr = md.load_frame(trajs[0], 0, top=topo)
        tr = tr.atom_slice(tr.topology.select(selection))
        topology = tr.topology
        self.n_atoms, self.n_residues = tr.topology.n_atoms, tr.topology.n_residues

        #Save topology if applicable
        if save_top:
            pkl.dump(topology, open('{0}.topy'.format(output), 'wb'))

        #Initialize interface computation if applicable
        if type(interface) == list:
            self.topg, self.topd = [create_topmat(sel, topology) for sel in interface[0:2]]
            interface_sum = []
            self.interfaced = True
        else:
            self.interfaced = False

        #Initialise total average computation
        total = []
        self.atomic_avg = csr_matrix((self.n_atoms, self.n_atoms))

        for j, traj in enumerate(trajs):
            print('Treating traj {}'.format(traj))
            traj_avg_list = []

            if self.all_frames != None:
                self.traj_output_folder = '{0}_{1}_frames'.format(output, j+1)
                mkdir(self.traj_output_folder, exist_ok=True)

            for n, tr in tqdm(enumerate(md.iterload(traj, top=topo, chunk=chunk))):
                self.prev = n*chunk
                #Slicing atoms of interest
                if selection != 'all':
                    tr = tr.atom_slice(tr.topology.select(selection))
                coords = tr.xyz
                atomicContacts = []
                self.queue = mp.Queue()
                if self.interfaced:
                    self.interface_queue = mp.Queue()
                processes = [mp.Process(target=self.get_contacts, args=([coords[frame]], frame)) for frame in range(tr.n_frames)]
                [p.start() for p in processes]
                atomicContacts = [self.queue.get() for p in processes]
                if self.interfaced:
                    interface_sum += [self.interface_queue.get() for p in processes]

                #Computing average atomic network from list of csr matrices
                chunk_avg = avg_sparse(atomicContacts, self.n_atoms, n_frames=tr.n_frames)
                traj_avg_list.append((chunk_avg, tr.n_frames))
                total.append((chunk_avg, tr.n_frames))

            #Tarball individual frames if applicable
            if self.all_frames == 'tar':
                with tarfile.open('{0}.tar'.format(self.traj_output_folder), "w") as tar:
                    k=0
                    filek = lambda: 'frame{0}.npz'.format(k)
                    while isfile(jn(self.traj_output_folder, filek())):
                        tar.add(jn(self.traj_output_folder, filek()), arcname=filek())
                        k+=1
                shutil.rmtree(self.traj_output_folder)

            #Average single trajectory file
            if self.ponderate:
                traj_avg = inverse_sparse(avg_sparse(traj_avg_list, self.n_atoms))
            else:
                traj_avg = avg_sparse(traj_avg_list, self.n_atoms)
            if output != None:
                save_npz('{0}_{1}.npz'.format(output, j+1), csr_matrix(traj_avg))

        #Average all the trajectories
        if self.ponderate:
            self.atomic_avg = inverse_sparse(avg_sparse(total, self.n_atoms))
        else:
            self.atomic_avg = avg_sparse(total, self.n_atoms)
        if output:
            save_npz(output, csr_matrix(self.atomic_avg))
        #
        if output and self.interfaced:
            contacts = np.array([value for (order, value) in sorted(interface_sum)])
            np.save('{0}_interface_{1}'.format(output, interface[2]), contacts)

    def get_contacts(self, coord, frame):
        tree = cKDTree(coord[0])
        if not self.ponderate:
            pairs = tree.query_pairs(r=self.cutoff/10.) #Cutoff is in Angstrom but mdtraj uses nm
            #Creating sparse CSR matrix
            data = np.ones(len(pairs), dtype=np.bool)
            pairs = np.array(list(pairs))
            contacts = csr_matrix((data, (pairs[:,0], pairs[:,1])), shape=[self.n_atoms, self.n_atoms])
            if self.all_frames:
                save_npz(jn(self.traj_output_folder, 'frame{0}'.format(self.prev+frame)), contacts)
            self.queue.put(contacts)
        else:
            distances = csr_matrix(tree.sparse_distance_matrix(tree, self.cutoff/10.))
            #score = inverse_sparse(distances)
            distances.eliminate_zeros()
            if self.all_frames:
                save_npz(jn(self.traj_output_folder, 'frame{0}'.format(self.prev+frame)), distances)
            self.queue.put(distances)
            contacts = distances
        if self.interfaced:
            to_app = (contacts @ self.topd).transpose() @ self.topg
            self.interface_queue.put((frame+self.prev, np.sum(to_app)))
            
        


# def create_atomic_multi(traj_list, topo_list, name_list, output_folder, selection='protein', cutoff=5, chunk=1000):
#     n_trajs = len(traj_list)
#     n_cpu = mp.cpu_count()
#     pool = mp.Pool(processes=min(n_cpu, len(traj_list)))
#     output_atomic_list = [jn(output_folder, '{0}.anpy'.format(name)) for name in name_list]
#     pool.starmap(AtomicNetMaker, zip(traj_list, topo_list, [selection]*n_trajs, [cutoff]*n_trajs, [chunk]*n_trajs, output_atomic_list))

def create_atomic_multi(traj_list, topo_list, name_list, output_folder, **kwargs):
    output_atomic_list = [jn(output_folder, '{0}.anpy'.format(name)) for name in name_list]
    mkdir(output_folder, exist_ok=True)
    for traj, topo, output_atomic in zip(traj_list, topo_list, output_atomic_list):
        AtomicNetMaker(traj, topo, output=output_atomic, **kwargs)



# def compute_interface(filename, top, expected=True, chain1=None, chain2=None):
#     if chain1 == None and chain2 == None:
#         sels = ['chainid 0', 'chainid 1']
#     else:
#         sels = ['chainid {}'.format(chain) for chain in [chain1, chain2]]
#     top = pkl.load(open(top, 'rb'))
#     topg, topd = [create_top_mat(sel, top) for sel in sels]
#     contact = []
#     with open(filename, 'rb') as fr:
#         try:
#             while True:
#                 atomicContacts = pkl.load(fr)
#                 if expected:
#                     ones = np.ones([topg.shape[0]]*2)
#                     evalue = (ones @ topd).transpose() @ topg
#                 for mat in atomicContacts:
#                     to_app = (mat @ topd).transpose() @ topg
#                     if expected:
#                         to_app = divide_expected(to_app, evalue)
#                     contact.append(np.sum(to_app))
#         except Exception as e:
#             if e == EOFError:
#                 pass
#             else:
#                 print(e) 

#     return contact

# def average_pickle(filename, output=None):
#     if output == None:
#         output = filename.replace('.p', '.anpy')
#     avg = []
#     with open(filename, 'rb') as fr:
#         try:
#             while True:
#                 atomicContacts = pkl.load(fr)
#                 intermediate_avg = np.zeros([atomicContacts[0].shape[0]]*2)
#                 for mat in atomicContacts:
#                     intermediate_avg[mat.nonzero()] += mat.data
#                 avg.append((csr_matrix(intermediate_avg), len(atomicContacts)))
#         except EOFError:
#             pass
#     atomic_avg = np.zeros([atomicContacts[0].shape[0]]*2)
#     n_frames = np.sum([elt[1] for elt in atomic_avg])
#     for mat, fact in avg:
#         atomic_avg[mat.nonzero()] += mat.data*(fact/n_frames)
#     atomic_avg = csr_matrix(atomic_avg)
#     pkl.dump(atomic_avg, open(output, 'wb'))
#     return atomic_avg, n_frames

# def average_list(file_list, output_total, output_list=[None]):
#     avg_list = [] 
#     fact_list = []
#     for f, o in zip_longest(file_list, output_list):
#         a, b  = average_pickle(f, o)
#         avg_list.append(a)
#         fact_list.append(b)
#     tot_frames = np.sum(fact_list)
#     tot_avg = np.zeros([avg_list[0].shape[0]]*2)
#     for mat, fact in zip(avg_list, fact_list):
#         tot_avg[mat.nonzero()] += mat.data*(fact/tot_frames)
#     tot_avg = csr_matrix(tot_avg)
#     pkl.dump(tot_avg, open(output_total, 'wb'))
        

def plot_interfaces(input_list, topo_list, name_list, expected=False, **kwargs):
    n_trajs = len(input_list)
    i2color =  dict(zip(range(n_trajs), sns.color_palette("colorblind", n_trajs)))
    n_cpu = mp.cpu_count()
    pool = mp.Pool(processes=min(n_cpu, len(input_list)))
    contacts = pool.starmap(compute_interface, zip(input_list, topo_list, [expected]*len(input_list)))
    fig, ax = plt.subplots(1, 1)
    for i, contact in enumerate(contacts):
        ax.plot(contact, color=i2color[i], label=name_list[i], **kwargs)
    return contacts

        
        
