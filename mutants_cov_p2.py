from maker import *
from os.path import join as jn
import matplotlib.pyplot as plt
import pickle as pkl
import mdtraj as md
import numpy as np
import MDAnalysis
from MDAnalysis.analysis.rms import RMSF 
from MDAnalysis.analysis import align
import seaborn as sns


INPUT_FOLDER = '/home/aria/landslide/MDRUNS/MUTANTS_COV/TRIMER'
OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/MUTANTS_COV/TRIMER'

name_list= ['OR', 'SA']
traj_list = [[jn(INPUT_FOLDER, 'spike{0}_ACE2_trimer_bound_test_chainABCDEF.dcd'.format(mutant))] for mutant in name_list]
topo_list = [jn(INPUT_FOLDER, 'spike{0}_ACE2_trimer_bound_test_chainABCDEF.psf'.format(mutant)) for mutant in name_list]
output_top_list = [jn(OUTPUT_FOLDER, name) for name in name_list]

#create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=20)
#save_top(traj_list, topo_list, name_list, output_top_list)


input_list = [jn(OUTPUT_FOLDER, '{0}_0.p'.format(name)) for name in name_list]
output = jn(OUTPUT_FOLDER, 'comparative_plots.png')

input_top = [top+'.topy' for top in output_top_list]
fig = plot_interfaces(input_list, input_top, name_list)
plt.style.use('default')
import matplotlib as mpl
mpl.rc('figure', fc = 'white')
plt.legend(ncol=2, loc='upper right')
plt.savefig(output)











