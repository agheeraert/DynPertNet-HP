from maker import *
from os.path import join as jn


INPUT_FOLDER = '/home/aria/landslide/MDRUNS/MUTANTS_COV'
OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/MUTANTS_COV/500NS'

mutants = ['OR', 'SA', 'BR', 'CA', 'UK']
traj_list = [[jn(INPUT_FOLDER, 'complex_ACE2_free_RBDmute{}_500ns_stride100_nowat.dcd'.format(mutant))] for mutant in mutants]
topo_list = [jn(INPUT_FOLDER, 'complex_ACE2_free_RBDmute{}_nowat.psf'.format(mutant)) for mutant in mutants]
output_top_list = [jn(OUTPUT_FOLDER, name) for name in mutants]

#create_atomic_multi(traj_list, topo_list, mutants, OUTPUT_FOLDER, chunk=100, save_top=True, interface=['chainid 0', 'chainid 1', 'interface_expected.npy', 'expected'])
create_atomic_multi(traj_list, topo_list, mutants, OUTPUT_FOLDER, chunk=20, save_top=True, interface=['chainid 0', 'chainid 1', 't'])
# def save_pdb_last(traj_list, topo_list, name_list, output_list, selection='protein'):
#     for i in range(len(name_list)):
#         t = md.load_frame(traj_list[i][0], 1000, top=topo_list[i])
#         t = t.atom_slice(t.topology.select(selection))
#         topo = t.topology
#         if str(next(topo.residues))[-1] == '0':
#             for res in t.topology.residues:
#                 res.resSeq += 1
#         t.save('{0}_100.pdb'.format(output_list[i]))


# save_pdb_last(traj_list, topo_list, mutants, output_top_list)






