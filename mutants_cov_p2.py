from maker import *
from os.path import join as jn

INPUT_FOLDER = '/home/aria/landslide/MDRUNS/MUTANTS_COV/TRIMER/'
OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/MUTANTS_COV/TRIMER/'

# name_list= ['OR', 'SA']
# traj_list = [[jn(INPUT_FOLDER, 'OR_nowat_1.dcd')],[jn(INPUT_FOLDER, 'SA_nowat_{}.dcd'.format(i)) for i in range(1, 3)]]
# topo_list = [jn(INPUT_FOLDER, 'OR_nowat.psf'), jn(INPUT_FOLDER, 'SA_nowat.psf')]

name_list = ['SA']
traj_list = [[jn(INPUT_FOLDER, 'SA_nowat_1.dcd')]]
topo_list = [jn(INPUT_FOLDER, 'SA_nowat.psf')]

#test zone
# INPUT_FOLDER = '/home/aria/landslide/MDRUNS/MUTANTS_COV/TRIMER/10frames'
# OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/MUTANTS_COV/TRIMER/10frames'

# name_list= ['OR', 'SA']
# traj_list = [[jn(INPUT_FOLDER, '{}.dcd'.format(name))] for name in name_list]
# traj_list[-1]+=[jn(INPUT_FOLDER, 'SA_2.dcd')]
# topo_list = [jn(INPUT_FOLDER, '{}.psf'.format(name)) for name in name_list]



create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=11, 
                     interface=['chainid 0 1 2', 'chainid 3 4 5', 't'])

# create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=10, 
#                     interface=[['chainid 0', 'chainid 1', 'chainid 2'], ['chainid 3', 'chainid 4', 'chainid 5']])









