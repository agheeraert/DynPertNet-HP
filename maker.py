from os import name
from itertools import zip_longest
import numpy as np
from scipy.spatial import cKDTree
import mdtraj as md
import pickle as pkl
import multiprocessing as mp
import matplotlib.pyplot as plt
from os.path import join as jn
from Bio.PDB.Polypeptide import aa1, aa3
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
warnings.simplefilter('ignore', PDBConstructionWarning)
three2one = dict(zip(aa3, aa1))
t2o = lambda X: three2one[X] if X in three2one else X[0]
label =  lambda X: t2o(X.name)+str(X.index)
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
import seaborn as sns


plt.style.use('default')
import matplotlib as mpl
mpl.rc('figure', fc = 'white')


class AtomicNetMaker():
    def __init__(self, trajs, topo=None, baseSelection='protein', cutoff=5, chunk=10000, output=None):
        """Function creating the atomic contact network with a desired base selection in chunks
        Parameters: traj: str or list of str: path trajectories to load
        topo: str: path of topology to use
        baseSelection: str: base selection on which to compute the atomic network. To save computation time, this should be the 
        smallest selection that includes all the selections in the list.
        """
        self.cutoff = cutoff
        if output != None:
            output = output.split('.')[0]
        firstpass, total = True, []
        for j, traj in enumerate(trajs):
            print('Treating traj {}'.format(traj))
            traj_avg_list = []
            for tr in tqdm(md.iterload(traj, top=topo, chunk=chunk)):
                #Slicing atoms of interest
                if baseSelection != 'all':
                    tr = tr.atom_slice(tr.topology.select(baseSelection))

                if firstpass:
                    self.n_atoms, self.n_residues = tr.topology.n_atoms, tr.topology.n_residues
                    labels = list(map(label, tr.topology.residues))
                    self.id2label = dict(zip(list(range(self.n_residues)), labels))
                    firstpass = False
                    self.atomic_avg = csr_matrix((self.n_atoms, self.n_atoms))

                coords = tr.xyz
                atomicContacts = []
                self.queue = mp.Queue()
                processes = [mp.Process(target=self.get_contacts, args=[np.array(coords[frame])]) for frame in range(tr.n_frames)]
                [p.start() for p in processes]
                atomicContacts = [self.queue.get() for p in processes]
                if output != None:
                    pkl.dump(atomicContacts, open('{0}_{1}.p'.format(output, j), 'ab+'))         
                
                #Computing average atomic network from list of csr matrices
                chunk_avg = np.zeros((self.n_atoms, self.n_atoms)) 
                for mat in atomicContacts:
                    chunk_avg[mat.nonzero()] += mat.data
                chunk_avg = csr_matrix(chunk_avg)
                chunk_avg /= tr.n_frames
                traj_avg_list.append((chunk_avg, tr.n_frames))
                total.append((chunk_avg, tr.n_frames))
            traj_avg = np.zeros((self.n_atoms, self.n_atoms))  
            n_frames = np.sum(elt[1] for elt in traj_avg_list)
            for mat, fact in traj_avg_list:
                traj_avg[mat.nonzero()] += mat.data*(fact/n_frames)
            if output != None:
                save_npz('{0}_{1}.npz'.format(output, j+1), csr_matrix(traj_avg))
                
        self.n_frames = np.sum([elt[1] for elt in total])
        for mat, fact in total:
            self.atomic_avg[mat.nonzero()] += mat.data*(fact/self.n_frames)
        if output:
            save_npz(output, csr_matrix(self.atomic_avg))

    def get_contacts(self, coord):
        tree = cKDTree(coord)
        pairs = tree.query_pairs(r=self.cutoff/10.) #Cutoff is in Angstrom but mdtraj uses nm
        #Creating sparse CSR matrix
        data = np.ones(len(pairs))
        pairs = np.array(list(pairs))
        self.queue.put(csr_matrix((data, (pairs[:,0], pairs[:,1])), shape=[self.n_atoms, self.n_atoms]))

# def create_atomic_multi(traj_list, topo_list, name_list, output_folder, selection='protein', cutoff=5, chunk=1000):
#     n_trajs = len(traj_list)
#     n_cpu = mp.cpu_count()
#     pool = mp.Pool(processes=min(n_cpu, len(traj_list)))
#     output_atomic_list = [jn(output_folder, '{0}.anpy'.format(name)) for name in name_list]
#     pool.starmap(AtomicNetMaker, zip(traj_list, topo_list, [selection]*n_trajs, [cutoff]*n_trajs, [chunk]*n_trajs, output_atomic_list))

def create_atomic_multi(traj_list, topo_list, name_list, output_folder, selection='protein', cutoff=5, chunk=1000):
    n_trajs = len(traj_list)
    output_atomic_list = [jn(output_folder, '{0}.anpy'.format(name)) for name in name_list]
    for traj, topo, output_atomic in zip(traj_list, topo_list, output_atomic_list):
        AtomicNetMaker(traj, topo, selection, cutoff, chunk, output_atomic)




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

def create_top_mat(selection, top, subselection='protein'):
    n_atoms = top.n_atoms
    n_residues = top.subset(top.select(subselection)).n_residues
    selection = selection.replace("not hydrogen", "!(name =~'H.*')")
    indexes = top.select(selection)
    top_mat = np.zeros((n_atoms, n_residues))
    for atom in top.atoms:
        if atom.index in indexes:
            top_mat[atom.index, atom.residue.index] = 1
    top_mat = csr_matrix(top_mat)
    return top_mat


def compute_interface(filename, top, chain1=None, chain2=None):
    if chain1 == None and chain2 == None:
        sels = ['chainid 0', 'chainid 1']
    else:
        sels = ['chainid {}'.format(chain) for chain in [chain1, chain2]]
    top = pkl.load(open(top, 'rb'))
    topg, topd = [create_top_mat(sel, top) for sel in sels]
    contact = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                atomicContacts = pkl.load(fr)
                for mat in atomicContacts:
                    contact.append(np.sum(mat.dot(topd).transpose().dot(topg)))
        except EOFError:
            pass
    return contact

def average_pickle(filename, output=None):
    if output == None:
        output = filename.replace('.p', '.anpy')
    avg = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                atomicContacts = pkl.load(fr)
                intermediate_avg = np.zeros([atomicContacts[0].shape[0]]*2)
                for mat in atomicContacts:
                    intermediate_avg[mat.nonzero()] += mat.data
                avg.append((csr_matrix(intermediate_avg), len(atomicContacts)))
        except EOFError:
            pass
    atomic_avg = np.zeros([atomicContacts[0].shape[0]]*2)
    n_frames = np.sum([elt[1] for elt in atomic_avg])
    for mat, fact in avg:
        atomic_avg[mat.nonzero()] += mat.data*(fact/n_frames)
    atomic_avg = csr_matrix(atomic_avg)
    pkl.dump(atomic_avg, open(output, 'wb'))
    return atomic_avg, n_frames

def average_list(file_list, output_total, output_list=[None]):
    avg_list = [] 
    fact_list = []
    for f, o in zip_longest(file_list, output_list):
        a, b  = average_pickle(f, o)
        avg_list.append(a)
        fact_list.append(b)
    tot_frames = np.sum(fact_list)
    tot_avg = np.zeros([avg_list[0].shape[0]]*2)
    for mat, fact in zip(avg_list, fact_list):
        tot_avg[mat.nonzero()] += mat.data*(fact/tot_frames)
    tot_avg = csr_matrix(tot_avg)
    pkl.dump(tot_avg, open(output_total, 'wb'))
        

def plot_interfaces(input_list, topo_list, name_list):
    n_trajs = len(input_list)
    i2color =  dict(zip(range(n_trajs), sns.color_palette("colorblind", n_trajs)))
    n_cpu = mp.cpu_count()
    pool = mp.Pool(processes=min(n_cpu, len(input_list)))
    contacts = pool.starmap(compute_interface, zip(input_list, topo_list))
    fig, ax = plt.subplots(1, 1)
    for i, contact in enumerate(contacts):
        ax.plot(contact, color=i2color[i], label=name_list[i])
    return fig





        
        
