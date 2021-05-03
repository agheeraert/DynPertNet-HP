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


INPUT_FOLDER = '/home/aria/landslide/MDRUNS/MUTANTS_COV'
OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/MUTANTS_COV'

name_list= ['OR', 'BR', 'CA', 'UK', 'SA']
mutants = ['_RBD_OR_', '_RBDmuteBR_', '_RBDmuteCA_', '_RBDmuteUK_', '_RBDmuteSA_']
traj_list = [[jn(INPUT_FOLDER, 'complex_ACE2_free{0}200ns_stride100.dcd'.format(mutant))] for mutant in mutants]
topo_list = [jn(INPUT_FOLDER, 'complex_ACE2_free{0}.psf'.format(mutant[:-1])) for mutant in mutants]
output_top_list = [jn(OUTPUT_FOLDER, name) for name in name_list]


input_list = [jn(OUTPUT_FOLDER, '{0}_0.p'.format(name)) for name in name_list]
output = jn(OUTPUT_FOLDER, 'comparative_plots.png')

input_top = [top+'.topy' for top in output_top_list]
fig = plot_interfaces(input_list, input_top, name_list)
plt.style.use('default')
import matplotlib as mpl
mpl.rc('figure', fc = 'white')
# fig = pkl.load(open(output[:-2], 'rb'))
# plt.legend(ncol=3, loc='upper right')
# plt.savefig(output)











