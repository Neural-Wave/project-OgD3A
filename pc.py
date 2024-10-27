import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
import networkx as nx

# Load the datasets
high = pd.read_csv("/teamspace/studios/this_studio/dataset/high_scrap.csv")
low = pd.read_csv("/teamspace/studios/this_studio/dataset/low_scrap.csv")

# Normalize the data
low = (low - low.mean()) / low.std()
high = (high - high.mean()) / high.std()
print("Data Normalized")

# Concatenate the two datasets
data = pd.concat([low, high], axis=0)
print("Data Concatenated")

print(data.describe())

# Convert your DataFrame to a numpy array
data_np = data.values

# Get variable names from the DataFrame columns
labels = list(data.columns)

# Define the significance level for the independence tests
alpha = 0.05  # You can adjust this value as needed

# Run the PC algorithm with the Fisher's Z conditional independence test
cg = pc(data_np, alpha=alpha, indep_test_func=fisherz, stable=True)
print("PC Algorithm Executed")

# Convert the learned graph to a NetworkX graph with labels
G = cg.nx_graph
print("Causal Graph Converted to NetworkX Graph")

print("Nodes: ", G.nodes())
print("Edges: ", G.edges())

# Plot the graph with directed edges
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='gray')
print("Directed edges plotted")

# Save the graph as a PDF
plt.savefig("/teamspace/studios/this_studio/pc_causal_graph.pdf", format="pdf")
print("Causal Graph Saved as PDF")

# Show the plot
plt.show()
