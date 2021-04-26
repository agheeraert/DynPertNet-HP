from pymol import cmd, stored
from pymol.cgo import *
import networkx as nx
from Bio.PDB.Polypeptide import aa1, aa3
import pickle as pkl
import pickle5 as pkl5
import numpy as np
from pymol.selecting import select
from scipy.sparse import csr_matrix, load_npz
from os.path import basename
three2one = dict(zip(aa3, aa1))
#letter2id = dict(zip([chr(ord('A') + i) for i in range(26)], list(map(str, range(26)))))

def load(path):
    extension = path.split('.')[-1]
    if extension != 'npz':
        try:
            return pkl.load(open(path, 'rb'))
        except ValueError:
            return pkl5.load(open(path, 'rb'))
    else:
        return load_npz(path)
    

t2o = lambda X: three2one[X] if X in three2one else X[0] 
relabel = lambda X: t2o(X[:3])+X[3:-1]+':'+X[-1]
selection = lambda X: " or first (resi "+X[3:-1]+" and elem C and chain "+X[-1]+")"

def create_top(selection, top):
    n_atoms = top.n_atoms
    n_residues = top.subset(top.select('protein')).n_residues
    selection = selection.replace("not hydrogen", "!(name =~'H.*')")        
    indexes = top.select(selection)
    top_mat = np.zeros((n_atoms, n_residues))
    for atom in top.atoms:
        if atom.index in indexes:
            top_mat[atom.index, atom.residue.index] = 1
    top_mat = csr_matrix(top_mat)
    return top_mat

def drawHydroPolar(path1, path2, threshold=0, edge_norm=None, scale_norm=True, norm_expected=False, **kwargs):

    if norm_expected:
        scale_norm = False

    cmd.select("hydrophobic", "(elem C or (all within 2 of elem C) and elem H)")
    cmd.select("polar", "not hydrophobic")
    hydro = [elt[1] for elt in cmd.index(selection="hydrophobic")]
    polar = [elt[1] for elt in cmd.index(selection="polar")]

    normalization_factor = len(hydro)**2/len(polar)**2

    offset = min(hydro[0], polar[0])
    sel_hydro = "index "+" ".join([str(elt-offset) for elt in hydro])
    sel_polar = "index "+" ".join([str(elt-offset) for elt in polar])

    drawNetwork(path1, path2, sele=sel_hydro, edge_color1=(1, 0.86, 0.73), edge_color2=(1, 0.86, 0), 
                name1="hydro1", name2="hydro2", name_nodes="hydrophobic_nodes", threshold=threshold, 
                edge_norm=edge_norm, norm_expected=norm_expected, **kwargs)

    if scale_norm:
        t2 = threshold/normalization_factor
        print("normalization factor polar:", normalization_factor)
        edge_norm2 = None if edge_norm is None else edge_norm/normalization_factor 
    else:
        t2 = threshold
        edge_norm2 = edge_norm

    drawNetwork(path1, path2, sele=sel_polar, edge_color1=(0.68, 0.85, 0.90), edge_color2=(0.25, 0.41, 0.88), 
                name1="polar1", name2="polar2", name_nodes="polar_nodes", threshold=t2, 
                edge_norm=edge_norm2,  norm_expected=norm_expected, keep_previous=True, **kwargs)

    if scale_norm:
        t3 = threshold*len(polar)/len(hydro)
        print("normalization factor mixed:", len(hydro)/len(polar))
        edge_norm3 = None if edge_norm is None else edge_norm*len(polar)/len(hydro)
    else:
        t3 = threshold
        edge_norm3 = edge_norm

    drawNetwork(path1, path2, sele1=sel_polar, sele2=sel_hydro, edge_color1=(0.60, 0.98, 0.60), edge_color2=(0, 0.50, 0), 
                name1="mixed1", name2="mixed2", name_nodes="mixed_nodes", threshold=t3, 
                edge_norm=edge_norm3, norm_expected=norm_expected, keep_previous=True, **kwargs)
    
 

def drawNetwork(path1, path2, sele=None, sele1=None, sele2=None, top1=None, top2=None,
                r=1, edge_norm=None, alpha=0.5, mutations=False, sum=False,
                node_color=(0.6, 0.6, 0.6), edge_color1 = (0, 0, 1), 
                edge_color2 = (1, 0, 0), labelling='0', norm_expected=False,
                threshold=0, topk=None, around=None, keep_previous=False, 
                name1 = None, name2 = None, name_nodes='nodes'):
    '''
    Draws a NetworkX network on the PyMol structure
    '''

    #Initialization of labelling variables and retreieving residue XYZ positions
    userSelection = "all"
    if not keep_previous:
        cmd.delete('*nodes *edges')
        cmd.label(selection=userSelection, expression="")
    # Building position -- name correspondance
    stored.posCA = []
    stored.names = []
    userSelection = userSelection + " and ((n. CA) or n. C)"
    cmd.iterate_state(1, selector.process(userSelection), "stored.posCA.append([x,y,z])")
    cmd.iterate(userSelection, 'stored.names.append(resn+resi+chain)')
    stored.labels = list(map(relabel, stored.names))
    stored.resid = list(map(selection, stored.names))
    node2id = dict(zip(stored.labels, stored.resid))
    node2CA = dict(zip(stored.labels, stored.posCA))

    #Loading external data
    mat1, mat2 = list(map(load, [path1, path2]))
    get_ext = lambda X: X.split('.')[-1]
    ext1, ext2 = list(map(get_ext, [path1, path2]))
    top1 = load(path1.replace('.'+ext1, '.topy')) if top1 == None else load(top1)
    top2 = load(path2.replace('.'+ext2, '.topy')) if top2 == None else load(top2)

    #Handling selections
    if sele != None:
        sele1, sele2 = [sele]*2
    if sele == None and sele1 == None and sele2 == None:
        sele1, sele2 = ['protein && not hydrogen']*2
        print('Default selection protein without hydrogens')
    sels = [sele1, sele2]
    

    #Creating topology matrices for each selection
    topg1, topd1 = [create_top(sel, top1) for sel in sels]
    topg2, topd2 = [create_top(sel, top2) for sel in sels]

    #From atomic to residual contacts and perturbation network computation
    mat1 = (mat1 @ topd1).transpose() @ topg1
    mat2 = (mat2 @ topd2).transpose() @ topg2
    #Apply expected norm if necessary
    if norm_expected:
        ones = csr_matrix(np.ones([topg1.shape[0]]*2))
        mat1 /= (ones @ topd1).transpose() @ topg1
        mat2 /= (ones @ topd2).transpose() @ topg2
        mat1, mat2 = list(map(csr_matrix, [mat1, mat2]))
    pertmat = mat2 - mat1

    pertmat.setdiag(0)
    pertmat.eliminate_zeros()

    
    net = nx.from_scipy_sparse_matrix(pertmat)

    #Creating labeling dictionnary
    if str(next(top1.residues))[-1] == '0':
        offset = 1
    else:
        offset = 0

    chain_names = [chr(ord('A') + i) for i in range(26)]

    t2o = lambda X: three2one[X] if X in three2one else X[0]
    get_chain = lambda X: chain_names[(X.chain.index % len(chain_names))]
    res2str = lambda X: t2o(X.name)+str(X.resSeq+offset)+':'+get_chain(X)
    id2label = {i: res2str(res) for i, res in enumerate(top1.residues)}

    #Output topK if necessary
    if type(topk) == int:
        limit_weight = np.sort([abs(net.edges[(u, v)]['weight']) for u, v in net.edges])[::-1][:topk]     
        threshold = limit_weight

    #Detect mutations
    if mutations:
        cmd.show_as(representation="cartoon", selection="?mutations")
        cmd.color(color="grey80", selection="?mutations")
        cmd.delete("?mutations")
        mutations_list = []
        y = {j: res2str(res) for j, res in enumerate(top2.residues)}
        for resid in id2label:
            if resid in y:
                if id2label[resid] != y[resid]:
                    mutations_list.append((resid, (y[resid][0]+':').join(id2label[resid].split(':'))))
                    cmd.select("mutations", 'resi '+str(id2label[resid].split(':')[0][1:])+ ' and chain '+id2label[resid][-1], merge=1)
            else:
                print('Deletion of ', id2label[resid])
        print('List of mutations: ', ', '.join([elt[1] for elt in mutations_list]))
        cmd.show_as(representation="licorice", selection="?mutations")
        cmd.color(color="magenta", selection="?mutations")

    #Relabelling network
    net = nx.relabel_nodes(net, id2label)

    #Apply threshold
    if threshold !=0:
        net.remove_edges_from([(u, v) for u, v in net.edges() if abs(net[u][v]['weight']) < threshold])

    #Induced perturbation network if needed

    if around !=None:
        net = net.subgraph(nx.node_connected_component(net, around))



    #Setting Pymol parameters
    cmd.set('auto_zoom', 0)
    cmd.set("cgo_sphere_quality", 4)
    print(node2CA)

    #Creating edges
    if edge_norm == None:
        edge_norm = max([net.edges()[(u, v)]['weight'] for u, v in net.edges()])/r
    obj1, obj2, nodelist = [], [], []
    for u, v in net.edges():
        radius = net[u][v]['weight']/edge_norm
        if abs(net[u][v]['weight']) >= threshold:
            if 'color' in net[u][v]: 
                if net[u][v]['color'] == 'r':
                    obj1+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color1, *edge_color1]
                else:
                    obj2+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color2, *edge_color2]
            else:
                if net[u][v]['weight'] <= 0:
                    obj1+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color1, *edge_color1]
                else:
                    obj2+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color2, *edge_color2]

            nodelist+=[u, v]


    #Creating nodes
    obj=[COLOR, *node_color]
    nodelist = set(nodelist)
    selnodes = ''.join([node2id[u] for u in nodelist])[4:]
    for u in nodelist:
        x, y, z = node2CA[u]
        obj+=[SPHERE, x, y, z, r]

    #Labelling
    if labelling==1:
        cmd.label(selection=selnodes, expression="t2o(resn)+resi")
    if labelling==3:
        cmd.label(selection=selnodes, expression="resn+resi")

    def name_edges(name, path):
        if name == None:
            return '.'.join(basename(path).split('.')[:-1])
        return name

    name1, name2 = map(name_edges, [name1, name2], [path1, path2])

    #Drawing nodes and edges
    cmd.load_cgo(obj, name_nodes)
    cmd.load_cgo(obj1, name1+'_edges')
    cmd.load_cgo(obj2, name2+'_edges')

    #Summing
    if sum:
        print('Sum of contacts lost: ', np.sum(pertmat))

cmd.extend("drawNetwork", drawNetwork)
cmd.extend("drawHydroPolar", drawHydroPolar)
cmd.extend("delNet", lambda: cmd.delete('*nodes *edges'))
cmd.extend("t2o", lambda X: three2one[X] if X in three2one else X[0])
