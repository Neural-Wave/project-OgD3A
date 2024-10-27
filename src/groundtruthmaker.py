import json
import numpy as np
from utils import get_index

def process_adjacency():
    with open('misc/ground_truth.json', 'r') as file:
        data = json.load(file)

    adjacency_matrix = np.zeros((98,98))
        
    adjacency_list = data.get('adjacency', [])
    
    for i, item in enumerate(adjacency_list):
        # Process each item in the adjacency list
        for var in item:
            id = var.get('id')
            adjacency_matrix[i][get_index(id)] = 1

    np.save('misc/adjacency_matrix.npy', adjacency_matrix)

    return adjacency_matrix

if __name__ == "__main__":
    adjacency_matrix = process_adjacency()
    print(adjacency_matrix)