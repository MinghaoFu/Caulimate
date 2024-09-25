import numpy as np
import networkx as nx

from Caulimate.Data.SimDAG import simulate_time_vary_weight, simulate_random_dag
from Caulimate.Data.SimCoords import simulate_uniform_coords
from Caulimate.Utils.GraphUtils import eudistance_mask
from Caulimate.Data.DataUtils import non_linear_case_2
from Caulimate.Utils.GraphUtils import bin_mat

class GPCDM:
    def __init__(self, 
                 T, 
                 D, 
                 L, 
                 degree,
                 graph_type,
                 t_period,
                 B_ranges,
                 extent = [-180, 180, -90, 90],
                 max_eud = 100,
                 noise_std=0.1,
                 seed=1) -> None:
        """
            zt = f(z_{t-1}) + e_t
            x_t = g(x_t, z_t) + e_t
        """
        self.T = T
        self.D = D
        self.L = L
        self.noise_std = noise_std
        self.rs = np.random.RandomState(seed)
        self.coords = simulate_uniform_coords(D, extent, seed)
        self.mask = eudistance_mask(self.coords, max_eud)
        self.B_bin = simulate_random_dag(D, degree, graph_type, seed) * self.mask
        self.Bs = simulate_time_vary_weight(self.B_bin, T, np.cos, t_period, B_ranges)
        
        self.Z = self.sim_transition(self.T, self.L, self.noise_std)
        self.X = self.sim_generation(self.Z, self.Bs, self.noise_std)
        

    def sim_transition(self, n, dim, noise_std):
        Z = np.zeros([n, dim])
        for i in range(n - 1):
            Z[i, :] = (Z[i - 1 if i != 0 else 0] + self.rs.normal(size=(dim)))
        return Z
    
    def sim_single_equation(self, X_t, B_i, Z_t, noise_std):
        if len(X_t) == 0:
            input_data = Z_t
        else:
            input_data = np.concatenate([X_t @ B_i, Z_t], axis=0)

        return non_linear_case_2(input_data, 1, noise_std)

    def sim_generation(self, Zs, Bs, noise_std):
        n = Bs.shape[0]
        dim = Bs.shape[-1]
        X = np.zeros([n, dim])
        for i in range(n):
            Bi = Bs[i]
            Zi = Zs[i]
            G = nx.DiGraph(bin_mat(Bi))
            ordered_vertices = list(nx.topological_sort(G))
            assert len(ordered_vertices) == dim
            for j in ordered_vertices:
                parents = list(G.predecessors(j))
                X[i:i+1, j] = self.sim_single_equation(X[i:i+1, parents], Bi[parents, j], Zi, noise_std)
        
        return X
    
    def save(self, path):
        np.savez(path, 
                 X=self.X, 
                 Z=self.Z, 
                 Bs=self.Bs, 
                 B_bin=self.B_bin)
    
if __name__ == "__main__":
    cgpdm = GPCDM(6000, 1000, 9, 3, "ER", 10, ((-0.15, -0.05),(0.05, 0.15)))
    cgpdm.save("test.npz")
    