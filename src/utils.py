import numpy as np

# Function to extract station number from variable name
def get_index(var_name):
    return int(var_name.split('_')[2])

def get_distance(A):
    ground_truth = np.load('misc/adjacency_matrix.npy')
    return np.sum(np.abs(A - ground_truth))

def adj_padder(A, min_size=98):
    current_size = A.shape[0]
    if current_size < min_size:
        larger_matrix = np.zeros((min_size, min_size), dtype=int)
        larger_matrix[:current_size, :current_size] = A
        return larger_matrix
    return A