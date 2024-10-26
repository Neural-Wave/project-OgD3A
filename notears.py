import pandas as pd
import numpy as np
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
from causalnex.structure.notears import from_pandas_lasso
#from causalnex.structure.structuremodel import StructureModel
import networkx as nx
import pickle
import matplotlib.pyplot as plt

# Load the datasets
high = pd.read_csv("/teamspace/studios/this_studio/dataset/high_scrap.csv")
low = pd.read_csv("/teamspace/studios/this_studio/dataset/low_scrap.csv")

# Normalize the data
low = (low - low.mean()) / low.std()
high = (high - high.mean()) / high.std()
print("Data Normalized")

# concat the two datasets
data = pd.concat([low, high], axis=0)
print("Data Concatenated")

# print("Data ", data.head())

print(data.describe())

# Define tabu edges based on the station sequence rule
tabu_edges = []
measurements = data.columns

# Extract station numbers from the column names
station_numbers = [int(col.split('_')[0][7:]) for col in measurements]

for i, measurement_i in enumerate(measurements):
    station_i = station_numbers[i]
    for j in range(i + 1, len(measurements)):
        measurement_j = measurements[j]
        station_j = station_numbers[j]
        if station_j < station_i:
            # Prevent edges from later stations to earlier stations
            tabu_edges.append((measurement_j, measurement_i))

print("Tabu Edges Added")

# Build a causal graph using the NOTEARS algorithm with the tabu edges
sm = from_pandas_lasso(
    data,
    beta=0.01,
    max_iter=100,
    w_threshold=0.015,
    tabu_edges=tabu_edges
)
print("Causal Graph Built")

# Save the causal model to a pickle file
with open("/teamspace/studios/this_studio/causal_model.pkl", "wb") as f:
    pickle.dump(sm, f)
print("Causal Model Saved as Pickle")

# # Remove weaker edges for a cleaner graph
# sm.remove_edges_below_threshold(0.02)
# print("Weak Edges Removed")

graph = nx.DiGraph([(int(measurement1.split('_')[2]), int(measurement2.split('_')[2])) for (measurement1, measurement2) in sm.edges()])

# Get the adjacency matrix of the causal graph
adj_matrix = nx.adjacency_matrix(graph).todense()

# Ensure the adjacency matrix has at least 98 rows and columns
min_size = 98
current_size = adj_matrix.shape[0]
if current_size < min_size:
    # Create a larger matrix filled with zeros
    larger_matrix = np.zeros((min_size, min_size), dtype=int)
    # Copy the existing adjacency matrix into the larger matrix
    larger_matrix[:current_size, :current_size] = adj_matrix
    adj_matrix = larger_matrix
adj_matrix_int = adj_matrix.astype(int)
np.save("/teamspace/studios/this_studio/oli/adjacency_matrix.npy", adj_matrix_int)
np.savetxt("/teamspace/studios/this_studio/oli/adjacency_matrix.txt", adj_matrix_int)
print(f"Adjacency Matrix: \n{adj_matrix_int}")

print("Number of edges: ", adj_matrix.sum())

head15 = np.loadtxt("/teamspace/studios/this_studio/head15.txt")

num_diff = np.sum(np.abs(head15 - adj_matrix_int[:15,:]))

print("Difference: ", num_diff)

# Define the target variable
target_node = 85

# Recursively get all nodes with a path to the target node using a reverse DFS
def get_all_ancestors(graph, target):
    ancestors = set()
    stack = [target]
    
    while stack:
        node = stack.pop()
        for predecessor in graph.predecessors(node):
            if predecessor not in ancestors:
                ancestors.add(predecessor)
                stack.append(predecessor)
    return ancestors

# Get all ancestors of the target node
ancestor_nodes = get_all_ancestors(graph, target_node)
# Include the target node itself
ancestor_nodes.add(target_node)

# Create a subgraph with the target node, its ancestors, and the edges between them
subgraph = graph.subgraph(ancestor_nodes).copy()  # Copy to ensure subgraph is editable

# Plot the subgraph with directed edges
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(subgraph)
nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color='lightblue')
nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold')
nx.draw_networkx_edges(subgraph, pos, arrowstyle='->', arrows=True, connectionstyle="arc3,rad=0.2")
print("Directed edges plotted")

# Save the directed subgraph as a PDF
plt.savefig("/teamspace/studios/this_studio/causal_graph.pdf", format="pdf")
print("Causal Graph Saved as PDF")

# Show the plot
plt.show()
