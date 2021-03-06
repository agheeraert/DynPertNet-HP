from maker import *
from os.path import join as jn
import matplotlib.pyplot as plt
import pickle as pkl
import mdtraj as md
import numpy as np


INPUT_FOLDER = '/home/aria/landslide/MDRUNS/IVAN_IGPS'
OUTPUT_FOLDER = '/home/aria/landslide/RESULTS/GUIDELINES/SCORING'

name_list = ['apo', 'prfar']
traj_list = [[jn(INPUT_FOLDER, 'prot_{0}_sim{1}_s10.dcd'.format(name, i)) for i in range(1, 5)] for name in name_list]
topo_list = [jn(INPUT_FOLDER, 'prot.prmtop')]*2
output_top_list = [jn(OUTPUT_FOLDER, name) for name in name_list]

create_atomic_multi(traj_list, topo_list, name_list, OUTPUT_FOLDER, chunk=20, save_top=True, ponderate=True)










