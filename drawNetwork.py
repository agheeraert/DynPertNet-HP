from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.shortest_paths import weighted
from pymol import cmd, stored
from pymol.cgo import *
from pymol.vfont import plain
import networkx as nx
from Bio.PDB.Polypeptide import aa1, aa3
import pickle as pkl
import pickle5 as pkl5
import numpy as np
import seaborn as sns
import pandas as pd
from pymol.selecting import select
from pymol.viewing import label, label2
from operator import itemgetter
from sklearn.cluster import Birch
from scipy.stats import rankdata
from scipy.sparse import csr_matrix, load_npz, dok_matrix
from os.path import basename
three2one = dict(zip(aa3, aa1))
#letter2id = dict(zip([chr(ord('A') + i) for i in range(26)], list(map(str, range(26)))))
IGPS_mapping = {'𝘧β1': '𝘧β1', '𝘧β2': 'Loop1', '𝘧α1': '𝘧α1', '𝘧β3': '𝘧β2', '𝘧α2': '𝘧α2', '𝘧β4': '𝘧β3', '𝘧α3': '𝘧α3', '𝘧β5': '𝘧β4', '𝘧α4': "𝘧α4'", '𝘧α5': '𝘧α4', '𝘧β6': '𝘧β5', '𝘧β7': "𝘧β5'", '𝘧β8': "𝘧β5'-𝘧α5", '𝘧α6': '𝘧α5', '𝘧β9': '𝘧β6', '𝘧α7': '𝘧α6', '𝘧β10': '𝘧β7', '𝘧α8': '𝘧α7', '𝘧β11': '𝘧β8', '𝘧α9': '𝘧β8-fα8', '𝘧α10': '𝘧α8', '𝘩β1': '𝘩β1', '𝘩α1': '𝘩α1', '𝘩β2': '𝘩β2', '𝘩β3': '𝘩β3', '𝘩α2': '𝘩α2', '𝘩α3': "𝘩α2'", '𝘩β4': '𝘩β4', '𝘩α4': '𝘩α3', '𝘩β5': '𝘩β5', '𝘩β6': '𝘩β7', '𝘩β7': "𝘩β8'", '𝘩β8': '𝘩β8', '𝘩α5': '𝘩β8-𝘩β9', '𝘩β9': '𝘩β9', '𝘩β10': '𝘩β10', '𝘩β11': '𝘩β11', '𝘩α6': '𝘩α4', '𝘧Head': '𝘧Head', '𝘩Head': '𝘩Head', '𝘩Tail': '𝘩Tail', '𝘩β11-𝘩α6': '𝘩β11-𝘩α4', '𝘩β10-𝘩β11': '𝘩β10-𝘩β11', '𝘩β9-𝘩β10': '𝘩β9-𝘩β10', '𝘩α5-𝘩β9': '𝘩β8-𝘩β9', '𝘩β8-𝘩α5': '𝘩β8-𝘩β9', '𝘩β7-𝘩β8': '𝘩β8', '𝘩β6-𝘩β7': '𝘩β7-𝘩β8', '𝘩β5-𝘩β6': '𝘩β6-𝘩β7', '𝘩α4-𝘩β5': '𝘩β5-𝘩β6', '𝘩α3-𝘩β4': '𝘩α2-𝘩β4', '𝘩α2-𝘩α3': "𝘩α2-𝘩α2'", '𝘩β3-𝘩α2': 'oxyanion strand', '𝘩β2-𝘩β3': '𝘩β2-𝘩β3', '𝘩α1-𝘩β2': 'Ω-loop', '𝘩β1-𝘩α1': '𝘩β1-𝘩α1', '𝘧Tail': '𝘧Tail', '𝘧α9-𝘧α10': "𝘧β8-𝘧α8'", '𝘧α8-𝘧β11': '𝘧α7-𝘧β8', '𝘧β10-𝘧α8': '𝘧β7-𝘧α7', '𝘧α7-𝘧β10': '𝘧α6-𝘧β7', '𝘧β9-𝘧α7': '𝘧β6-𝘧α6', '𝘧α6-𝘧β9': '𝘧α5-𝘧β6', '𝘧β7-𝘧β8': '𝘧β5-𝘧α5', '𝘧β6-𝘧β7': "𝘧β5-𝘧β5'", '𝘧α5-𝘧β6': '𝘧α4-𝘧β5', '𝘧α4-𝘧α5': "𝘧α4'-𝘧α4", '𝘧α3-𝘧β5': '𝘧α3-𝘧β4', '𝘧β4-𝘧α3': '𝘧β3-𝘧α3', '𝘧α2-𝘧β4': '𝘧α2-𝘧β3', '𝘧β3-𝘧α2': '𝘧β2-𝘧α2', '𝘧α1-𝘧β3': '𝘧α1-𝘧β2', '𝘧β2-𝘧α1': 'Loop1', '𝘧β1-𝘧β2': 'Loop1'}

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

def create_top(selection, top, fromstruct=None):
    n_atoms = top.n_atoms
    n_residues = top.subset(top.select('protein')).n_residues
    selection = selection.replace("not hydrogen", "!(name =~'H.*')")
    if any([subsel in selection for subsel in ['hydrophobic', 'polar']]):
        subsel = [subsel for subsel in ['hydrophobic', 'polar'] if subsel in selection]
        indexes = get_hydro_polar(subsel[0], fromstruct)
        selection = selection.replace(subsel[0], indexes)
    indexes = top.select(selection)
    top_mat = dok_matrix((n_atoms, n_residues))
    for atom in top.atoms:
        if atom.index in indexes:
            top_mat[atom.index, atom.residue.index] = 1
    top_mat = csr_matrix(top_mat)
    return top_mat

def get_hydro_polar(selection, fromstruct=None):
    hydrosel = "(elem C or (all within 2 of elem C) and elem H)"
    if selection == "hydrophobic":
        if fromstruct == None:
            cmd.select("atoms", "{}".format(hydrosel))
        else:
            cmd.load(fromstruct, 'struct')
            cmd.select("atoms", "struct and {}".format(hydrosel))
    elif selection == "polar":
        if fromstruct == None:
            cmd.select("atoms", "not {}".format(hydrosel))
        else:
            cmd.load(fromstruct, 'struct')
            cmd.select("atoms", "struct and not {}".format(hydrosel))

    if fromstruct == None:
        cmd.select("f", "first all")
    else:
        cmd.select("f", "first struct")
    offset = cmd.index(selection="f")[0][1]
    indexes = [elt[1] for elt in cmd.index(selection="atoms")]
    selection = "index "+" ".join([str(elt-offset) for elt in indexes])
    cmd.delete('struct or atoms or f')
    return selection

def get_expected_matrix(atom_mat, top_d, top_g):
    mat = (atom_mat @ top_d).transpose() @ top_g
    #Apply expected norm if necessary
    expected = (top_d.sum(axis=1).transpose() @ top_d).transpose() @ (top_g.sum(axis=1).transpose() @ top_g)
    mat = dok_matrix(divide_expected(mat, expected))
    return mat


def get_expected_matrices(atom_mat, top, fromstruct):
    top_h, top_p = [create_top(sel, top, fromstruct) for sel in ["hydrophobic", "polar"]]
    mat_h = get_expected_matrix(atom_mat, top_h, top_h)
    mat_p = get_expected_matrix(atom_mat, top_p, top_p)
    mat_m = get_expected_matrix(atom_mat, top_p, top_h)
    return [mat_h, mat_p, mat_m]


def get_expected_type(atom_mat1, atom_mat2, top1, top2, fromstruct):
    expected_matrices_1 = get_expected_matrices(atom_mat1, top1, fromstruct)
    expected_matrices_2 = get_expected_matrices(atom_mat2, top2, fromstruct)
    return expected_matrices_1+expected_matrices_2
  

# def divide_expected2(mat1, mat2):
#     mat1, mat2 = list(map(dok_matrix, [mat1, mat2]))
#     res = dok_matrix(mat1.shape)
#     res[mat2.nonzero()] = mat1[mat2.nonzero()] / mat2[mat2.nonzero()]
#     return res

def divide_expected(mat1, mat2):
    res = mat1 / mat2
    res[np.isnan(res)] = 0
    return res

def get_connected_components(pertmat):
    mat = np.abs(pertmat.copy())
    net = nx.from_scipy_sparse_matrix(mat)
    net.remove_nodes_from(list(nx.isolates(net)))
    connected_components = [nx.number_connected_components(net)]
    while mat.nnz !=0:
        mat.data[np.argmin(mat.data)] = 0
        mat.eliminate_zeros()
        net = nx.from_scipy_sparse_matrix(mat)
        net.remove_nodes_from(list(nx.isolates(net)))
        connected_components.append(nx.number_connected_components(net))
    return connected_components[:-1]

def get_hubs(c, method):
    try:
        if method == 'eigenvector':
            centralities = nx.eigenvector_centrality(c, weight='weight', max_iter=1000)
        elif method == 'pagerank':
            centralities = nx.pagerank(c, max_iter=1000)
        elif method == 'hits_hub':
            centralities = nx.hits(c, max_iter=1000)[1]
        elif method == 'hits_authority':
            centralities = nx.hits(c, max_iter=1000)[0]
        elif method == 'betweenness':
            centralities = nx.betweenness_centrality(c, weight='weight')
        elif method == 'katz':
            centralities = nx.katz_centrality(c, weight='weight', max_iter=10000)
        else:
            raise NameError
    except nx.PowerIterationFailedConvergence:
        return None

    max_hub = max(centralities.items(), key=itemgetter(1))[1]
    hubs = [node for node in centralities if abs(centralities[node]-max_hub)<=0.001]
    if len(hubs) == len(c.nodes()):
        return 'all nodes'
    else:
        return (', '.join(hubs))

def drawHydroPolar(path1, path2, threshold=0, edge_norm=None, scale_norm=True, norm_expected=False, fromstruct=None, **kwargs):

    if norm_expected:
        scale_norm = False

    if fromstruct == None:
        cmd.select("hydrophobic", "(elem C or (all within 2 of elem C) and elem H)")
        cmd.select("polar", "not hydrophobic")
    else:
        cmd.load(fromstruct, 'struct')
        cmd.select("hydrophobic", "struct and (elem C or (all within 2 of elem C) and elem H)")
        cmd.select("polar", "not hydrophobic and struct")

    hydro = [elt[1] for elt in cmd.index(selection="hydrophobic")]
    polar = [elt[1] for elt in cmd.index(selection="polar")]

    if fromstruct != None:
        cmd.delete('struct')

    normalization_factor = len(hydro)**2/len(polar)**2

    offset = min(hydro[0], polar[0])
    sel_hydro = "index "+" ".join([str(elt-offset) for elt in hydro])
    sel_polar = "index "+" ".join([str(elt-offset) for elt in polar])

    drawNetwork(path1, path2, sele=sel_hydro, edge_color1=(1, 0.86, 0.73), edge_color2=(1, 0.86, 0), 
                name1="hydro1", name2="hydro2", name_nodes="hydrophobic_nodes", threshold=threshold, 
                edge_norm=edge_norm, norm_expected=norm_expected, **kwargs)

    if scale_norm:
        t2 = threshold/normalization_factor
        print("Normalization factor polar:", normalization_factor)
        edge_norm2 = None if edge_norm is None else edge_norm/normalization_factor 
    else:
        t2 = threshold
        edge_norm2 = edge_norm

    drawNetwork(path1, path2, sele=sel_polar, edge_color1=(0.68, 0.85, 0.90), edge_color2=(0.25, 0.41, 0.88), 
                name1="polar1", name2="polar2", name_nodes="polar_nodes", threshold=t2, 
                edge_norm=edge_norm2,  norm_expected=norm_expected, keep_previous=True, **kwargs)

    if scale_norm:
        t3 = threshold*len(polar)/len(hydro)
        print("Normalization factor mixed:", len(hydro)/len(polar))
        edge_norm3 = None if edge_norm is None else edge_norm*len(polar)/len(hydro)
    else:
        t3 = threshold
        edge_norm3 = edge_norm

    drawNetwork(path1, path2, sele1=sel_polar, sele2=sel_hydro, edge_color1=(0.60, 0.98, 0.60), edge_color2=(0, 0.50, 0), 
                name1="mixed1", name2="mixed2", name_nodes="mixed_nodes", threshold=t3, 
                edge_norm=edge_norm3, norm_expected=norm_expected, keep_previous=True, **kwargs)
    
 

def drawNetwork(path1, path2, sele=None, sele1=None, sele2=None, top1=None, top2=None,
                r=1, edge_norm=None, alpha=0.5, mutations=False, align_with = None, 
                node_color=(0.6, 0.6, 0.6), edge_color1 = (0, 0, 1), palette="colorblind",
                edge_color2 = (1, 0, 0), labeling='0', norm_expected=False,
                threshold=0, topk=None, max_compo=None, mean_vp=None, strong_compo=None, 
                around=None, keep_previous=False, compo_size=None, save_cc=None, load_cc=None,
                compos_to_excel = None, force_binary_color=False, compo_radius=None, compo_diam=None,
                label_compo='', auto_patch=True, printall=False, sum=False, n_clusters=None,
                color_by_compo=False, color_by_group=False, show_top_group=None,
                name1 = None, name2 = None, name_nodes='nodes', userSelection='all',
                fromstruct=None, color_by_contact_type=False, standard_and_expected=None):
    '''
    Draws a NetworkX network on the PyMol structure
    '''

    #Initialization of labeling variables and retreieving residue XYZ positions
    if not keep_previous:
        cmd.delete('*nodes *edges Component* Group*')
        cmd.label(selection=userSelection, expression="")
        cmd.hide("licorice", "?mutations")
    # Building position -- name correspondance
    stored.posCA = []
    stored.names = []
    stored.ss = []
    userSelection = userSelection + " and ((n. CA) or n. C)"
    cmd.iterate_state(1, selector.process(userSelection), "stored.posCA.append([x,y,z])")
    cmd.iterate(userSelection, "stored.ss.append(ss)")
    cmd.iterate(userSelection, 'stored.names.append(resn+resi+chain)')
    stored.labels = list(map(relabel, stored.names))
    stored.resid = list(map(selection, stored.names))
    node2id = dict(zip(stored.labels, stored.resid))
    node2CA = dict(zip(stored.labels, stored.posCA))

    #Secondary Structure labels
    prevSS, prevChain = None, None
    counters = {'': 0, 'H': 0, 'S': 0, 'L': 0}
    node2SS = dict(zip(stored.labels, stored.ss))
    SS2nodelist = {}
    putflag = lambda X: 'U' if X in ['', 'L'] else X
    for label in node2SS:
        ss = node2SS[label]
        chain = label[-1]
        if prevChain != chain:
            for counter in counters: counters[counter] = 0
        if prevSS != ss:
            counters[ss] +=1
        labss = putflag(ss)+str(counters[ss])+':'+chain
        if labss in SS2nodelist:
            SS2nodelist[labss].append(label)
        else:
            SS2nodelist[labss] = [label]
        prevSS = ss
        prevChain = chain

    prevkey, prevChain = None, None
    order = []
    keys = list(SS2nodelist.keys())

    for key in keys:
        if prevChain != key.split(':')[-1]:
            prevkey = None
        if key[0] == 'U':
            if prevkey == None:
                newkey = 'Head:'+key.split(':')[-1]
            else:
                newkey = 'U'+prevkey
            SS2nodelist[newkey] = SS2nodelist.pop(key)
            order.append(newkey)
        else:
            order.append(key)
        prevkey = key
        prevChain = key.split(':')[-1]
    prevkey = None
    final = []
    for key in order[::-1]:
        if prevChain != key.split(':')[-1]:
            prevkey = None
        if key[0] == 'U':
            if prevkey == None:
                newkey = 'Tail:'+key.split(':')[-1]
            else:
                newkey = '{}-{}'.format(key[1:], prevkey)
            SS2nodelist[newkey] = SS2nodelist.pop(key)
            final.append(newkey)
        else:
            final.append(key)
        prevkey = key
        prevChain = key.split(':')[-1]
    # ss_dict = dict(zip(keys, final[::-1]))
    mapss = {}
    for key in final:
        newkey = key.replace('S', 'β').replace('H', 'α').replace('αead', 'Head')
        if 'IGPS' in str(label_compo):
            _ = []
            for elt in newkey.split('-'):    
                if elt.split(':')[1] in ['A', 'C', 'E']:
                    _.append('𝘧{}'.format(elt.split(':')[0]))
                elif elt.split(':')[1] in ['B', 'D', 'F']:
                    _.append('𝘩{}'.format(elt.split(':')[0]))
            newkey = '-'.join(_)
            mapss[key] = IGPS_mapping[newkey]      
        else:
            mapss[key] = newkey     

    for ss in SS2nodelist:
        for node in SS2nodelist[ss]:
            node2SS[node] = mapss[ss]


    #Loading external data
    atom_mat1, atom_mat2 = list(map(load, [path1, path2]))
    get_ext = lambda X: X.split('.')[-1]
    ext1, ext2 = list(map(get_ext, [path1, path2]))
    top1 = load(path1.split('_')[0]+'.topy') if top1 == None else load(top1)
    top2 = load(path2.split('_')[0]+'.topy') if top2 == None else load(top2)

    #Handling selections
    if sele != None:
        sele1, sele2 = [sele]*2
    if sele == None and sele1 == None and sele2 == None:
        sele1, sele2 = ['protein && not hydrogen']*2
        print('Default selection protein without hydrogens')
    
    sels = [sele1, sele2]

    #Creating topology matrices for each selection
    topg1, topd1 = [create_top(sel, top1, fromstruct) for sel in sels]
    topg2, topd2 = [create_top(sel, top2, fromstruct) for sel in sels]
    #From atomic to residual contacts and perturbation network computation
    mat1 = (atom_mat1 @ topd1).transpose() @ topg1
    mat2 = (atom_mat2 @ topd2).transpose() @ topg2
    #Apply expected norm if necessary
    if norm_expected:
        exp1 = (topd1.sum(axis=1).transpose() @ topd1).transpose() @ (topg1.sum(axis=1).transpose() @ topg1)
        exp2 = (topd2.sum(axis=1).transpose() @ topd2).transpose() @ (topg2.sum(axis=1).transpose() @ topg2)
        mat1 = divide_expected(mat1, exp1)
        mat2 = divide_expected(mat2, exp2)
        mat1, mat2 = list(map(csr_matrix, [mat1, mat2]))

    if align_with != None:
        cmd.align(align_with, userSelection, object='aln')
        raw_aln = cmd.get_raw_alignment('aln')
        cmd.hide('cgo', 'aln')
        order_string = [idx[0] for idx in raw_aln[-1]][::-1]
        trans_mat = dok_matrix(tuple([cmd.count_atoms(X) for X in order_string]))
        for idx1, idx2 in raw_aln:
            trans_mat[idx2[1]-1, idx1[1]-1] = 1
        trans_mat = csr_matrix(trans_mat)
        top_t1, top_t2 = [create_top('name CA', top) for top in [top1, top2]]
        trans_res = (trans_mat @ top_t1).transpose() @ top_t2
        mat2 = trans_res @ (mat2 @ trans_res.transpose())

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
    # if 'IGPS' in label_compo:
    #     igps_label = {}
    #     for elt in id2label.items():
    #         if elt.split(':')[1] in ['A', 'C', 'E']:
    #             rerelabel[elt] = '𝘧{}'.format(elt.split(':')[0])
    #         elif elt.split(':')[1] in ['B', 'D', 'F']:
    #             rerelabel[elt] = '𝘩{}'.format(elt.split(':')[0])
    #Relabeling network
    net = nx.relabel_nodes(net, id2label)

    label2id = {res2str(res): i for i, res in enumerate(top1.residues)}



    #Auto_patching network labels
    if not all(elem in node2CA for elem in net.nodes()):
        print('PDB structure and topology labeling not matching.')
        if auto_patch:
            print('Attempting to auto-patch residue names. (this can be disabled with auto_patch=False)')
            if len(node2CA.keys()) == len(net.nodes()):
                remap = dict(zip(net.nodes(), node2CA.keys()))
                net = nx.relabel_nodes(net, remap)
                label2id = dict(zip(node2CA.keys(), range(top1.n_residues)))
            else:
                print("Auto-patching not working, please try on different PDB file")


    #Output topK if necessary
    if type(topk) == int:
        limit_weight = np.sort([abs(net.edges[(u, v)]['weight']) for u, v in net.edges])[::-1][topk] 
        threshold = limit_weight

    if type(standard_and_expected) == int:
        limit_weight = np.sort([abs(net.edges[(u, v)]['weight']) for u, v in net.edges])[::-1][standard_and_expected]
        relabel_net2 = dict(enumerate(net.nodes()))
        threshold = limit_weight


    if max_compo or mean_vp or any(np.array([compo_size, compo_diam, compo_radius, strong_compo])!= None): 
        color_by_compo = True
        if load_cc != None:
            cc = np.load(load_cc)
        else:
            cc = get_connected_components(pertmat)
            if save_cc != None:
                np.save(save_cc, cc)
        if max_compo:
            threshold = np.sort(np.abs(pertmat.data))[::-1][np.argmax(cc[::-1])]
        else:
            lastmax = np.sort(np.abs(pertmat.data))[::-1][np.argmax(cc[::-1])]
            print('last maximum: {}'.format(np.round(lastmax, 2)))
            net.remove_edges_from([(u, v) for u, v in net.edges() if abs(net[u][v]['weight']) < lastmax])
            net.remove_nodes_from(list(nx.isolates(net)))
            components_list = [net.subgraph(c).copy() for c in nx.connected_components(net)] 
            if mean_vp:
                vanishing_points = [np.max([abs(net[u][v]['weight']) for u, v in c.edges()]) for c in components_list]
                threshold = np.median(vanishing_points)
            elif compo_size !=None:
                robust = [list(c.nodes()) for c in components_list if len(c.edges())>=float(compo_size)]
                net = net.subgraph([x for robust in list(robust) for x in robust])
                threshold = 0
            elif compo_diam !=None:
                robust = [list(c.nodes()) for c in components_list if nx.diameter(c)>=float(compo_diam)]
                net = net.subgraph([x for robust in list(robust) for x in robust])
                threshold = 0
            elif compo_radius !=None:
                robust = [list(c.nodes()) for c in components_list if nx.radius(c)>=float(compo_radius)]
                net = net.subgraph([x for robust in list(robust) for x in robust])
                threshold = 0
            elif strong_compo !=None:
                vanishing_points = [np.max([abs(net[u][v]['weight']) for u, v in c.edges()]) for c in components_list]
                edges_len = [len(c.edges()) for c in components_list]
                percentile = float(strong_compo)*len(components_list)/100
                vani_ranks = len(vanishing_points)+1-rankdata(vanishing_points, method='max')
                size_ranks = len(edges_len)+1-rankdata(edges_len, method='max')
                vani_nodes = [list(c.nodes()) for i, c in enumerate(components_list) if vani_ranks[i]<percentile]
                size_nodes = [list(c.nodes()) for i, c in enumerate(components_list) if size_ranks[i]<percentile]
                vani_nodes = [x for vani_nodes in list(vani_nodes) for x in vani_nodes]
                size_nodes = [x for size_nodes in list(size_nodes) for x in size_nodes]
                strong = list(set(vani_nodes) & set(size_nodes))
                net = net.subgraph(strong)


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


    #Apply threshold
    if threshold !=0:
        print('Applying threshold {}'.format(threshold))
        net.remove_edges_from([(u, v) for u, v in net.edges() if abs(net[u][v]['weight']) < threshold])
        net.remove_nodes_from(list(nx.isolates(net)))

    #Induced perturbation network if needed

    if around !=None:
        net = net.subgraph(nx.node_connected_component(net, around))

    #Setting Pymol parameters
    cmd.set('auto_zoom', 0)
    cmd.set("cgo_sphere_quality", 4)
    if len(net.edges()) == 0:    
        raise ValueError('Computations give empty network')

    #Norm edges
    if edge_norm == None:
        edge_norm = max([net.edges()[(u, v)]['weight'] for u, v in net.edges()])/r

    elif edge_norm == True:
        tot_atoms_in_sel = np.sum([np.sum(elt) for elt in [topd1, topd2, topg1, topg2]])
        tot_atoms = np.sum([max(elt.shape) for elt in [topd1, topd2, topg1, topg2]])
        norm_fact = tot_atoms_in_sel**2/tot_atoms**2
        edge_norm = norm_fact*30
        print('Global normalization factor: {}'.format(1/norm_fact))


    #Function to name edges
    def name_edges(name, path):
        if name == None:
            return '.'.join(basename(path).split('.')[:-1])
        return name

    if type(standard_and_expected) == int:
        exp1 = (topd1.sum(axis=1).transpose() @ topd1).transpose() @ (topg1.sum(axis=1).transpose() @ topg1)
        exp2 = (topd2.sum(axis=1).transpose() @ topd2).transpose() @ (topg2.sum(axis=1).transpose() @ topg2)
        mat1 = divide_expected(mat1, exp1)
        mat2 = divide_expected(mat2, exp2)
        mat1, mat2 = list(map(csr_matrix, [mat1, mat2]))
        net2 = nx.from_scipy_sparse_matrix(mat2-mat1)
        net2 = nx.relabel_nodes(net2, relabel_net2)
        limit_weight = np.sort([abs(net2.edges[(u, v)]['weight']) for u, v in net2.edges])[::-1][standard_and_expected] 
        net2.remove_edges_from([(u, v) for u, v in net2.edges() if abs(net2[u][v]['weight']) < limit_weight])
        net2.remove_nodes_from(list(nx.isolates(net2)))
        colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
        objs_inboth = []
        objs_instd = []
        objs_inexp = []
        nodes = []
        for u, v in net.edges():
            radius = net[u][v]['weight']/edge_norm
            if (u, v) in list(net2.edges()):
                objs_inboth += [CYLINDER, *node2CA[u], *node2CA[v], radius, *colors[0], *colors[0]]
            else:
                objs_instd += [CYLINDER, *node2CA[u], *node2CA[v], radius, *colors[1], *colors[1]]
            nodes += [u, v]
        edge_norm2 = max([net2.edges()[(u, v)]['weight'] for u, v in net2.edges()])/r
        for u, v in net2.edges():
            radius = net2[u][v]['weight']/edge_norm2
            if (u, v) not in list(net.edges()):
                objs_inexp += [CYLINDER, *node2CA[u], *node2CA[v], radius, *colors[2], *colors[2]]
            nodes += [u, v]

        nodelist = set(nodes)
        objs_nodes = [COLOR, *node_color]
        for u in nodelist:
                x, y, z = node2CA[u]
                objs_nodes += [SPHERE, x, y, z, r]
        selnodes = ''.join([node2id[u] for u in nodelist])[4:]
        cmd.load_cgo(objs_inboth, 'in_both_edges') 
        cmd.load_cgo(objs_instd, 'in_std_edges')
        cmd.load_cgo(objs_inexp, 'in_exp_edges')
        cmd.load_cgo(objs_nodes, 'nodes') 



    elif color_by_contact_type:
        expected_matrices = get_expected_type(atom_mat1, atom_mat2, top1, top2, fromstruct)
        name1, name2 = list(map(name_edges, [name1, name2], [path1, path2]))
        names = ['{0}_{1}'.format(name1, sel) for sel in ['hydro', 'polar', 'mixed']] + ['{0}_{1}'.format(name2, sel) for sel in ['hydro', 'polar', 'mixed']]
        nodes_dict = {i: [] for i in range(len(names))}
        objs_dict = {i: [] for i in range(len(names))}
        colors = [(1, 0.86, 0.73), (0.68, 0.85, 0.90), (0.60, 0.98, 0.60), (1, 0.86, 0), (0.25, 0.41, 0.88), (0, 0.50, 0)]
        for u, v in net.edges():
            radius = net[u][v]['weight']/edge_norm
            id_u, id_v = label2id[u], label2id[v]
            values = list(map(lambda _mat: _mat[id_v, id_u], expected_matrices))
            type_of_contact = np.argmax(values)
            objs_dict[type_of_contact] += [CYLINDER, *node2CA[u], *node2CA[v], radius, *colors[type_of_contact], *colors[type_of_contact]]
            nodes_dict[type_of_contact] += [u, v]
        selnodes = ''
        for toc in nodes_dict:
            nodelist = set(nodes_dict[toc])
            objs_dict[toc]+=[COLOR, *node_color]
            for u in nodelist:
                x, y, z = node2CA[u]
                objs_dict[toc]+=[SPHERE, x, y, z, r]
            selnodes += ''.join([node2id[u] for u in nodelist])[4:]

        for i, name in zip(objs_dict.keys(), names):
            cmd.load_cgo(objs_dict[i], '{}_edges'.format(name))         
    
    #Coloring by components
    elif color_by_compo:
        components_list = [net.subgraph(c).copy() for c in nx.connected_components(net)]
        diameters = [nx.diameter(c) for c in components_list]
        ranking = np.argsort(diameters)[::-1]
        colors = sns.color_palette(palette, n_colors=len(components_list)+1)
        for i, c in enumerate(colors):
            if c[0] == c[1] == c[2]:
                print(c)
                colors.pop(i)
                break
        selnodes = ''
        for i, rank in enumerate(ranking):
            color, compo = colors[rank], components_list[rank]
            _obj, nodelist = [], []
            for u, v in compo.edges():
                radius = net[u][v]['weight']/edge_norm
                if abs(net[u][v]['weight']) >= threshold:
                    if not force_binary_color:
                        _obj+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *color, *color]
                    else:
                        if net[u][v]['weight'] <= 0:
                            _obj+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color1, *edge_color1]
                        else:
                            _obj+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color2, *edge_color2]
                    nodelist += [u, v]
#            cmd.load_cgo(_obj, 'Component{}_edges'.format(i+1))
            _obj+=[COLOR, *node_color]
            nodelist = set(nodelist)
            selnodes += ''.join([node2id[u] for u in nodelist])[4:]
            for u in nodelist:
                x, y, z = node2CA[u]
                _obj+=[SPHERE, x, y, z, r]
            cmd.load_cgo(_obj, 'Component{}'.format(i+1)) 

    #Color by group of relevance  
    elif color_by_group:
        weights = np.array([abs(net[u][v]['weight']) for u, v in net.edges()]).reshape(-1, 1)
        birch = Birch(n_clusters=n_clusters).fit(weights)
        labels = birch.predict(weights)
        ordered_labels = labels[np.argsort(pertmat.data)]
        _, idx = np.unique(ordered_labels, return_index=True)
        mapping = dict(zip(ordered_labels[np.sort(idx)], np.sort(np.unique(ordered_labels))))
        i2color =  dict(zip(ordered_labels[np.sort(idx)], sns.color_palette(palette, len(np.unique(ordered_labels)))[::-1]))
        selnodes = ''
        if show_top_group == None:
            show_top_group = len(mapping.keys())
        
        for j, i in enumerate(list(mapping.keys())[:show_top_group]):
            _obj, nodelist = [], []
            _net = net.copy()
            to_remove_edges = [(u, v) for j, (u, v) in enumerate(net.edges()) if labels[j] != i]
            _net.remove_edges_from(to_remove_edges)
            _net.remove_nodes_from(list(nx.isolates(_net)))
            for u, v in _net.edges():
                radius = net[u][v]['weight']/edge_norm
                if abs(net[u][v]['weight']) >= threshold:
                    _obj+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *i2color[j], *i2color[j]]
                    nodelist += [u, v]
#            cmd.load_cgo(_obj, 'Component{}_edges'.format(i+1))
            _obj+=[COLOR, *node_color]
            nodelist = set(nodelist)
            selnodes += ''.join([node2id[u] for u in nodelist])[4:]
            for u in nodelist:
                x, y, z = node2CA[u]
                _obj+=[SPHERE, x, y, z, r]
            cmd.load_cgo(_obj, 'Group{}'.format(j+1)) 

    #Default edge coloring   
    else:
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
        name1, name2 = map(name_edges, [name1, name2], [path1, path2])
        cmd.load_cgo(obj1, name1+'_edges')
        cmd.load_cgo(obj2, name2+'_edges')

        #Drawing nodes 
        obj=[COLOR, *node_color]
        nodelist = set(nodelist)
        selnodes = ''.join([node2id[u] for u in nodelist])[4:]
        for u in nodelist:
            x, y, z = node2CA[u]
            obj+=[SPHERE, x, y, z, r]

        cmd.load_cgo(obj, name_nodes)


    #Creating text for labeling components
    if label_compo != '' or compos_to_excel !=None:
        if compos_to_excel != None:
            rows_list = []
        objtxt = []
        axes = -np.array(cmd.get_view()[:9]).reshape(3,3)
        components_list = [net.subgraph(c).copy() for c in nx.connected_components(net)]
        diameters = [nx.diameter(c) for c in components_list]
        for i, j in enumerate(np.argsort(diameters)[::-1]):
            row_dict = {}
            c = components_list[j]
            sses = sorted(list(set([node2SS[node] for node in c])))
            if compos_to_excel !=None:
                row_dict['Secondary structure elements'] = ','.join(sses)
                row_dict['Vanishing point'] = np.max([abs(net[u][v]['weight']) for u, v in c.edges()])
                row_dict['Diameter'] = nx.diameter(c)
                row_dict['Size'] = len(c.edges())
                row_dict['Size rank'] = i+1

            else:
                print('Component {}\n'.format(i+1), ', '.join(sses))
                print('Size (number of edges) {}'.format(len(c.edges())))
                print('Vanishing point: {}'.format(np.max([abs(net[u][v]['weight']) for u, v in c.edges()])))
            if 'h' in str(label_compo):
                methods = ['eigenvector', 'hits_hub', 'hits_authority', 'pagerank', 'betweenness', 'katz']
                hubs = [get_hubs(c, method) for method in methods]
                if compos_to_excel !=None:
                    row_dict.update(dict(zip(methods, hubs)))
                else:
                    print(dict(zip(methods, hubs)))
            if 'c' in str(label_compo):
                pos = np.array(node2CA[next(c.__iter__())]) + (axes[0])
                cyl_text(objtxt, plain, pos, 'Component {}'.format(i+1), radius=0.1, color=[0, 0, 0], axes=axes)
            if compos_to_excel:
                rows_list.append(row_dict)
        if compos_to_excel:
            df = pd.DataFrame(rows_list)
            df.to_excel(compos_to_excel)
        if 's' in str(label_compo):
            for ss in SS2nodelist:
                nodelist = SS2nodelist[ss] 
                print(mapss[ss], ': ', ('{}--{}'.format(nodelist[0], nodelist[-1]) if len(nodelist)>1 else nodelist[0]))

#        print(objtxt)
        cmd.set("cgo_line_radius", 0.03)
        cmd.load_cgo(objtxt, 'txt')

    #labeling
    if labeling==1:
        cmd.label(selection=selnodes, expression="t2o(resn)+resi")
    if labeling==3:
        cmd.label(selection=selnodes, expression="resn+resi")

    #Summing
    if sum:
        print('Sum of contacts lost: ', np.sum(pertmat))

    if printall:
        print([(u,v, net[u][v]) for u, v in net.edges()])


cmd.extend("drawNetwork", drawNetwork)
cmd.extend("drawHydroPolar", drawHydroPolar)
cmd.extend("delNet", lambda: cmd.delete('*nodes *edges'))
cmd.extend("t2o", lambda X: three2one[X] if X in three2one else X[0])
