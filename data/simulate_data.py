import numpy as np



################################ Hardcode ##############################
def generate_six_nodes_DAG(random_state, x_size=1000):
    x3 = random_state.uniform(size=x_size)
    x0 = 3.0 * x3 + random_state.uniform(size=x_size)
    x2 = 6.0 * x3 + random_state.uniform(size=x_size)
    x1 = 3.0 * x0 + 2.0 * x2 + random_state.uniform(size=x_size)
    x5 = 4.0 * x0 + random_state.uniform(size=x_size)
    x4 = 8.0 * x0 - 1.0 * x2 + random_state.uniform(size=x_size)
    X = np.array([x0, x1, x2, x3, x4, x5]).T
    return X
