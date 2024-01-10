# For climate data from 
# Prof. Dr. Jakob Runge (he/him)
# climateinformaticslab.com
 
# Chair of Climate Informatics
# Technische Universität Berlin | Electrical Engineering and Computer Science | Institute of Computer Engineering and Microelectronics
 
# Group Lead Causal Inference
# German Aerospace Center | Institute of Data Science

import numpy as np
import pickle

DATA_PATH = './data/CESM2_pacific_SST.pkl'

def load_data(args):
    with open(DATA_PATH, 'rb') as file:
        data = pickle.load(file)
    
    data = data[:, :args.d_X]
    # reset args.num
    args.num = data.shape[0]
    
    m_true = generate_band_bin_matrix(args.num, args.d_X, args.distance)
    
    return data, m_true

def generate_band_bin_matrix(n, d, bandwidth):
    matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(max(0, i - bandwidth), min(d, i + bandwidth + 1)):
            if i != j:
                matrix[i, j] = 1
    return np.tile(matrix, (n, 1, 1))