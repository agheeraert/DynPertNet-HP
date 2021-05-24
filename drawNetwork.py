from pymol import cmd, stored
from pymol.cgo import *
from pymol.vfont import plain
import networkx as nx
from Bio.PDB.Polypeptide import aa1, aa3
import pickle as pkl
import pickle5 as pkl5
import numpy as np
import seaborn as sns
from pymol.selecting import select
from pymol.viewing import label
from sklearn.cluster import Birch
from scipy.sparse import csr_matrix, load_npz, dok_matrix
from os.path import basename
three2one = dict(zip(aa3, aa1))
#letter2id = dict(zip([chr(ord('A') + i) for i in range(26)], list(map(str, range(26)))))
IGPS_mapping = {'𝘧β1': '𝘧β1', '𝘧β2': 'Loop1', '𝘧α1': '𝘧α1', '𝘧β3': '𝘧β2', '𝘧α2': '𝘧α2', '𝘧β4': '𝘧β3', '𝘧α3': '𝘧α3', '𝘧β5': '𝘧β4', '𝘧α4': "𝘧α4'", '𝘧α5': '𝘧α4', '𝘧β6': '𝘧β5', '𝘧β7': "𝘧β5'", '𝘧β8': "𝘧β5'-𝘧α5", '𝘧α6': '𝘧α5', '𝘧β9': '𝘧β6', '𝘧α7': '𝘧α6', '𝘧β10': '𝘧β7', '𝘧α8': '𝘧α7', '𝘧β11': '𝘧β8', '𝘧α9': '𝘧β8-fα8', '𝘧α10': '𝘧α8', '𝘨β1': '𝘨β1', '𝘨α1': '𝘨α1', '𝘨β2': '𝘨β2', '𝘨β3': '𝘨β3', '𝘨α2': '𝘨α2', '𝘨α3': "𝘨α2'", '𝘨β4': '𝘨β4', '𝘨α4': '𝘨α3', '𝘨β5': '𝘨β5', '𝘨β6': '𝘨β7', '𝘨β7': "𝘨β8'", '𝘨β8': '𝘨β8', '𝘨α5': '𝘨β8-𝘨β9', '𝘨β9': '𝘨β9', '𝘨β10': '𝘨β10', '𝘨β11': '𝘨β11', '𝘨α6': '𝘨α4', '𝘧Head': '𝘧Head', '𝘨Head': '𝘨Head', '𝘨Tail': '𝘨Tail', '𝘨β11-𝘨α6': '𝘨β11-𝘨α4', '𝘨β10-𝘨β11': '𝘨β10-𝘨β11', '𝘨β9-𝘨β10': '𝘨β9-𝘨β10', '𝘨α5-𝘨β9': '𝘨β8-𝘨β9', '𝘨β8-𝘨α5': '𝘨β8-𝘨β9', '𝘨β7-𝘨β8': '𝘨β8', '𝘨β6-𝘨β7': '𝘨β7-𝘨β8', '𝘨β5-𝘨β6': '𝘨β6-𝘨β7', '𝘨α4-𝘨β5': '𝘨β5-𝘨β6', '𝘨α3-𝘨β4': '𝘨α2-𝘨β4', '𝘨α2-𝘨α3': "𝘨α2-𝘨α2'", '𝘨β3-𝘨α2': 'oxyanion strand', '𝘨β2-𝘨β3': '𝘨β2-𝘨β3', '𝘨α1-𝘨β2': 'Ω-loop', '𝘨β1-𝘨α1': '𝘨β1-𝘨α1', '𝘧Tail': '𝘧Tail', '𝘧α9-𝘧α10': "𝘧β8-𝘧α8'", '𝘧α8-𝘧β11': '𝘧α7-𝘧β8', '𝘧β10-𝘧α8': '𝘧β7-𝘧α7', '𝘧α7-𝘧β10': '𝘧α6-𝘧β7', '𝘧β9-𝘧α7': '𝘧β6-𝘧α6', '𝘧α6-𝘧β9': '𝘧α5-𝘧β6', '𝘧β7-𝘧β8': '𝘧β5-𝘧α5', '𝘧β6-𝘧β7': "𝘧β5-𝘧β5'", '𝘧α5-𝘧β6': '𝘧α4-𝘧β5', '𝘧α4-𝘧α5': "𝘧α4'-𝘧α4", '𝘧α3-𝘧β5': '𝘧α3-𝘧β4', '𝘧β4-𝘧α3': '𝘧β3-𝘧α3', '𝘧α2-𝘧β4': '𝘧α2-𝘧β3', '𝘧β3-𝘧α2': '𝘧β2-𝘧α2', '𝘧α1-𝘧β3': '𝘧α1-𝘧β2', '𝘧β2-𝘧α1': 'Loop1', '𝘧β1-𝘧β2': 'Loop1'}

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
    top_mat = dok_matrix((n_atoms, n_residues))
    for atom in top.atoms:
        if atom.index in indexes:
            top_mat[atom.index, atom.residue.index] = 1
    top_mat = csr_matrix(top_mat)
    return top_mat

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
    connected_components = [nx.number_connected_components(net)]
    while mat.nnz !=0:
        mat.data[np.argmin(mat.data)] = 0
        mat.eliminate_zeros()
        net = nx.from_scipy_sparse_matrix(mat)
        net.remove_nodes_from(list(nx.isolates(net)))
        connected_components.append(nx.number_connected_components(net))
    return connected_components[:-1]


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
                r=1, edge_norm=None, alpha=0.5, mutations=False, align_with = None, 
                node_color=(0.6, 0.6, 0.6), edge_color1 = (0, 0, 1),
                edge_color2 = (1, 0, 0), labeling='0', norm_expected=False,
                threshold=0, topk=None, max_compo=None, mean_vp=None, 
                around=None, keep_previous=False, robust_compo=False,
                label_compo=None, auto_patch=True, printall=False, sum=False,
                color_by_compo=False, color_by_group=False, show_top_group=None,
                name1 = None, name2 = None, name_nodes='nodes', userSelection='all'):
    '''
    Draws a NetworkX network on the PyMol structure
    '''

    #Initialization of labeling variables and retreieving residue XYZ positions
    if not keep_previous:
        cmd.delete('*nodes *edges Component* Group*')
        cmd.label(selection=userSelection, expression="")
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
        if label_compo == 'IGPS':
            _ = []
            for elt in newkey.split('-'):    
                if elt.split(':')[1] in ['A', 'C', 'E']:
                    _.append('𝘧{}'.format(elt.split(':')[0]))
                elif elt.split(':')[1] in ['B', 'D', 'F']:
                    _.append('𝘨{}'.format(elt.split(':')[0]))
            newkey = '-'.join(_)
            mapss[key] = IGPS_mapping[newkey]            
        else:
            mapss[key] = newkey     

    for ss in SS2nodelist:
        for node in SS2nodelist[ss]:
            node2SS[node] = mapss[ss]


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
        exp1 = (topd1.sum(axis=1).transpose() @ topd1).transpose() @ (topg1.sum(axis=1).transpose() @ topg1)
        exp2 = (topd2.sum(axis=1).transpose() @ topd2).transpose() @ (topg2.sum(axis=1).transpose() @ topg2)
        mat1 = divide_expected(mat1, exp1)
        mat2 = divide_expected(mat2, exp2)
        mat1, mat2 = list(map(csr_matrix, [mat1, mat2]))

    if align_with != None:
        cmd.align(userSelection, align_with, object='aln')
        raw_aln = cmd.get_raw_alignment('aln')
        order_string = [idx[0] for idx in raw_aln[-1]]
        mat_shapes = [X.shape[0] for X in [mat1, mat2]]
        if order_string[0] == align_with:
            mat_shapes = mat_shapes[::-1]
        trans_mat = dok_matrix(tuple([cmd.count_atoms(X) for X in order_string]))
        for idx1, idx2 in raw_aln:
            trans_mat[idx1[1]-1, idx2[1]-1] = 1
        trans_mat = csr_matrix(trans_mat)
        top_t1, top_t2 = [create_top('name CA', top) for top in [top1, top2]]
        trans_res = (trans_mat @ top_t1).transpose() @ top_t2
        mat2 = trans_res @ (mat2 @ trans_res.transpose())
        print(mat2.shape, mat1.shape)

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
    #Relabeling network
    net = nx.relabel_nodes(net, id2label)

    #Auto_patching network labels
    if not all(elem in node2CA for elem in net.nodes()):
        print('PDB structure and topology labeling not matching.')
        if auto_patch:
            print('Attempting to auto-patch residue names. (this can be disabled with auto_patch=False)')
            if len(node2CA.keys()) == len(net.nodes()):
                remap = dict(zip(net.nodes(), node2CA.keys()))
                net = nx.relabel_nodes(net, remap)
            else:
                print("Auto-patching not working, please try on a different .pdb")


    #Output topK if necessary
    if type(topk) == int:
        limit_weight = np.sort([abs(net.edges[(u, v)]['weight']) for u, v in net.edges])[::-1][topk] 
        threshold = limit_weight

    if max_compo or mean_vp or robust_compo:
        cc = get_connected_components(pertmat)
        if max_compo:
            threshold = np.sort(np.abs(pertmat.data))[::-1][np.argmax(cc[::-1])]
        else:
            lastmax = np.sort(np.abs(pertmat.data))[::-1][np.argmax(cc[::-1])]
            net.remove_edges_from([(u, v) for u, v in net.edges() if abs(net[u][v]['weight']) < lastmax])
            net.remove_nodes_from(list(nx.isolates(net)))
            components_list = [net.subgraph(c).copy() for c in nx.connected_components(net)] 
            if mean_vp:
                vanishing_points = [np.max([abs(net[u][v]['weight']) for u, v in c.edges()]) for c in components_list]
                threshold = np.median(vanishing_points)
            elif robust_compo:
                robust = [list(c.nodes()) for c in components_list if len(c.edges())>=4]
                net = net.subgraph([x for robust in list(robust) for x in robust])
                threshold = 0

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


    #Norm edges
    if edge_norm == None:
        edge_norm = max([net.edges()[(u, v)]['weight'] for u, v in net.edges()])/r

    #Function to name edges
    def name_edges(name, path):
        if name == None:
            return '.'.join(basename(path).split('.')[:-1])
        return name
    
    #Default edge coloring   
    if not color_by_compo and not color_by_group:
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
    
    #Coloring by components
    elif color_by_compo:
        components_list = [net.subgraph(c).copy() for c in nx.connected_components(net)]
        compo_size = [len(c.edges()) for c in components_list]
        ranking = np.argsort(compo_size)[::-1]
        colors = sns.color_palette("husl", n_colors=len(components_list))
        selnodes = ''
        for i, rank in enumerate(ranking):
            color, compo = colors[rank], components_list[rank]
            _obj, nodelist = [], []
            for u, v in compo.edges():
                radius = net[u][v]['weight']/edge_norm
                if abs(net[u][v]['weight']) >= threshold:
                    _obj+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *color, *color]
                    nodelist += [u, v]
#            cmd.load_cgo(_obj, 'Component{}_edges'.format(i+1))
            _obj+=[COLOR, *node_color]
            nodelist = set(nodelist)
            selnodes += ''.join([node2id[u] for u in nodelist])[4:]
            for u in nodelist:
                x, y, z = node2CA[u]
                _obj+=[SPHERE, x, y, z, r]
            cmd.load_cgo(_obj, 'Component{}'.format(i+1))   
    else:
        weights = np.array([abs(net[u][v]['weight']) for u, v in net.edges()]).reshape(-1, 1)
        birch = Birch(n_clusters=None).fit(weights)
        labels = birch.predict(weights)
        ordered_labels = labels[np.argsort(pertmat.data)]
        _, idx = np.unique(ordered_labels, return_index=True)
        mapping = dict(zip(ordered_labels[np.sort(idx)], np.sort(np.unique(ordered_labels))))
        i2color =  dict(zip(ordered_labels[np.sort(idx)], sns.color_palette("husl", len(np.unique(ordered_labels)))))
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

        

    #Creating text for labeling components
    if label_compo != None:
        objtxt = []
        axes = -np.array(cmd.get_view()[:9]).reshape(3,3)
        components_list = [net.subgraph(c).copy() for c in nx.connected_components(net)]
        edges_len = [len(c.edges()) for c in components_list]
        for i, j in enumerate(np.argsort(edges_len)[::-1]):
            c = components_list[j]
            sses = sorted(list(set([node2SS[node] for node in c])))
            print('Component {}\n'.format(i+1), ', '.join(sses))
            print('Vanishing point: {}'.format(np.max([abs(net[u][v]['weight']) for u, v in c.edges()])))
            pos = np.array(node2CA[next(c.__iter__())]) + (axes[0])
            cyl_text(objtxt, plain, pos, 'Component {}'.format(i+1), radius=0.1, color=[0, 0, 0], axes=axes)
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
