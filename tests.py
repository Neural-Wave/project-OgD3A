import pandas as pd
import numpy as np
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
from causalnex.structure.notears import from_pandas_lasso
#from causalnex.structure.structuremodel import StructureModel
import networkx as nx
import matplotlib.pyplot as plt

# Load the datasets
high = pd.read_csv("/teamspace/studios/this_studio/dataset/high_scrap.csv")
low = pd.read_csv("/teamspace/studios/this_studio/dataset/low_scrap.csv")

print("High Scrap Data")
#print(high.head())
print("Low Scrap Data")
#print(low.head())

# Normalize the data
low = (low - low.mean()) / low.std()
high = (high - high.mean()) / high.std()
print("Data Normalized")

# Use 'low' dataset for causal discovery
# data = low

# Add a column to differentiate between the two datasets
high['high'] = 1.0
low['high'] = 0.0
# concat the two datasets
data = pd.concat([low, high], axis=0)
print("Data Concatenated")

print("Data ", data.head())

print(data.describe())

# Define tabu edges based on the station sequence rule
tabu_edges = []
measurements = data.columns.drop('high') if 'high' in data.columns else data.columns
for i, measurement in enumerate(measurements):
    for j in range(i + 1, len(measurements)):
        if int(measurement.split('_')[0][7:]) != int(measurements[j].split('_')[0][7:]):
            tabu_edges.append((measurements[j], measurement))
print("Tabu Edges Added")

# Build a causal graph using the `NOTEARS` algorithm with the tabu edges
sm = from_pandas_lasso(data, 0.1, max_iter=1000, w_threshold=0.0, tabu_edges=tabu_edges)
#sm = from_pandas(data, max_iter=1000, w_threshold=0.0, tabu_edges=tabu_edges)
print("Causal Graph Built")

# # Remove weaker edges for a cleaner graph
# sm.remove_edges_below_threshold(0.02)
# print("Weak Edges Removed")

graph = nx.DiGraph([(int(measurement1.split('_')[0][7:]), int(measurement2.split('_')[0][7:])) for (measurement1, measurement2) in sm.edges()])

# Get the adjacency matrix of the causal graph
adj_matrix = nx.adjacency_matrix(graph).todense()
np.save("/teamspace/studios/this_studio/oli/adjacency_matrix.npy", adj_matrix)
np.savetxt("/teamspace/studios/this_studio/oli/adjacency_matrix.txt", adj_matrix)
print(f"Adjacency Matrix: \n{adj_matrix}")

# Define the target variable
target_node = '85'

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
ancestor_nodes = get_all_ancestors(sm, target_node)
# Include the target node itself
ancestor_nodes.add(target_node)

# Create a subgraph with the target node, its ancestors, and the edges between them
subgraph = graph.subgraph(ancestor_nodes).copy()  # Copy to ensure subgraph is editable
print("Subgraph Nodes:", subgraph.nodes())
print("Subgraph Edges:", subgraph.edges())

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
