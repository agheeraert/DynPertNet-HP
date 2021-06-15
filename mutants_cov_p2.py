from maker import *
from os.path import join as jn

INPUT_FOLDER = '/home/aria/landslide/MDRUNS/MUTANTS_COV/TRIMER'
OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/MUTANTS_COV/TRIMER'

name_list= ['OR', 'SA']
traj_list = [[jn(INPUT_FOLDER, 'spikeOR_ACE2_trimer_bound_300ns_stride100_nowat.dcd')], [jn(INPUT_FOLDER, 'spikeSA_ACE2_trimer_bound_440ns_stride100_nowat.dcd')]]
topo_list = [jn(INPUT_FOLDER, 'spikeOR_ACE2_trimer_bound_nowat.psf'), jn(INPUT_FOLDER, 'spikeSA_ACE2_trimer_bound_nowat.psf')]
output_top_list = [jn(OUTPUT_FOLDER, name) for name in name_list]

create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=8, save_top=True)

# create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=8, interface=['chainid 1', 'chainid 3', 'interface_13.npy'])
# create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=8, interface=['chainid 1', 'chainid 4', 'interface_14.npy'])
# create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=8, interface=['chainid 0', 'chainid 3', 'interface_03.npy'])
# create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=8, interface=['chainid 0', 'chainid 4', 'interface_04.npy'])


# input_list = [jn(OUTPUT_FOLDER, '{0}_0.p'.format(name)) for name in name_list]
# output = jn(OUTPUT_FOLDER, 'comparative_plots.png')

# input_top = [top+'.topy' for top in output_top_list]
# fig = plot_interfaces(input_list, input_top, name_list)
# plt.style.use('default')
# import matplotlib as mpl
# mpl.rc('figure', fc = 'white')
# plt.legend(ncol=2, loc='upper right')
# plt.savefig(output)











