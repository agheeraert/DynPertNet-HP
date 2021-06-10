from maker import *
from os.path import join as jn


INPUT_FOLDER = '/home/aria/landslide/MDRUNS/MUTANTS_COV'
OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/MUTANTS_COV/500NS'

mutants = ['BR']
traj_list = [[jn(INPUT_FOLDER, 'complex_ACE2_free_RBDmute{}_500ns_stride100_nowat.dcd'.format(mutant))] for mutant in mutants]
topo_list = [jn(INPUT_FOLDER, 'complex_ACE2_free_RBDmute{}_nowat.psf'.format(mutant)) for mutant in mutants]
output_top_list = [jn(OUTPUT_FOLDER, name) for name in mutants]

create_atomic_multi(traj_list, topo_list, mutants, OUTPUT_FOLDER, chunk=20, save_top=True, interface=['chainid 0', 'chainid 1', 'interface.npy'])












