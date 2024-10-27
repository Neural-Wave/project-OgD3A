import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

# Load the datasets
high_scrap = pd.read_csv("/teamspace/studios/this_studio/dataset/high_scrap.csv")
low_scrap = pd.read_csv("/teamspace/studios/this_studio/dataset/low_scrap.csv")

# Combine the datasets
data = pd.concat([low_scrap, high_scrap], axis=0).reset_index(drop=True)

# Normalize the data
data = (data - data.mean()) / data.std()
print("Data normalized and combined.")

# Discretize the data
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
data_discrete = pd.DataFrame(discretizer.fit_transform(data), columns=data.columns)
data_discrete = data_discrete.astype(int)
print("Data discretized.")

# Extract variable names and station numbers
variables = data.columns.tolist()

# Function to extract station number from variable name
def get_station(var_name):
    return int(var_name.split('_')[0][7:])  # Assumes format 'StationX_...'

# Create a list of forbidden edges based on temporal constraints
forbidden_edges = []
for var1 in variables:
    station1 = get_station(var1)
    for var2 in variables:
        station2 = get_station(var2)
        if station2 > station1:
            # Prevent edges from later stations to earlier stations
            forbidden_edges.append((var2, var1))

# Initialize the scoring method and estimator
score_discrete = BicScore(data_discrete)
hc_discrete = HillClimbSearch(data_discrete)

# Estimate the structure (returns a DAG object)
dag_discrete = hc_discrete.estimate(
    scoring_method=score_discrete,
    black_list=forbidden_edges
)
print("Structure learning with discretized data completed.")

# Convert the DAG to a BayesianNetwork object
model_discrete = BayesianNetwork(dag_discrete.edges())

# Fit the model parameters using Bayesian Estimation
model_discrete.fit(data_discrete, estimator=BayesianEstimator, prior_type="BDeu")
print("Parameter learning completed.")

# Initialize inference
infer = VariableElimination(model_discrete)

# Identify all variables causally upstream of the target variable
def get_ancestors(graph, target):
    ancestors = set()
    predecessors = list(graph.predecessors(target))
    for predecessor in predecessors:
        if predecessor not in ancestors:
            ancestors.add(predecessor)
            ancestors.update(get_ancestors(graph, predecessor))
    return ancestors


# Print the CPDs for all nodes
for cpd in model_discrete.get_cpds():
    print(f"CPD of {cpd.variable}:")
    print(cpd)
    print("\n")
    
    

target_variable = 'Station5_mp_85'
ancestors = get_ancestors(model_discrete, target_variable)

# Calculate the influence of each ancestor on the target variable
influences = {}
for var in ancestors:
    # Set the variable to its low and high states
    low_state = data_discrete[var].min()
    high_state = data_discrete[var].max()
    
    # Compute the probability distribution of the target variable
    # when the ancestor variable is at high and low states
    evidence_high = {var: high_state}
    evidence_low = {var: low_state}
    
    # Marginal distribution of the target variable
    target_dist_high = infer.query(variables=[target_variable], evidence=evidence_high)
    target_dist_low = infer.query(variables=[target_variable], evidence=evidence_low)
    
    # Compute the expected value of the target variable
    target_states = np.arange(data_discrete[target_variable].nunique())
    expected_high = np.sum(target_dist_high.values * target_states)
    expected_low = np.sum(target_dist_low.values * target_states)
    
    influence = expected_high - expected_low
    influences[var] = abs(influence)

# Rank the root causes based on their influence
sorted_influences = sorted(influences.items(), key=lambda item: item[1], reverse=True)

# Save the sorted influences to a text file
with open("sorted_influences.txt", "w") as file:
    for var, influence in sorted_influences:
        file.write(f"{var}: {influence}\n")
print("Sorted influences saved to 'sorted_influences.txt'.")

print("\nRoot causes ranked by influence on the target variable:")
for var, influence in sorted_influences:
    print(f"{var}: {influence}")

nodes = list(model_discrete.nodes())
print("Nodes:", nodes)

edges = list(model_discrete.edges())
print("Edges:", edges)

G = nx.DiGraph(edges)

# Remove self-loops if any
G.remove_edges_from(nx.selfloop_edges(G))
print("Self-loops removed.")

# Create a subgraph with the target node, its ancestors, and the edges between them
ancestors.add(target_variable)
subgraph = G.subgraph(ancestors).copy()  # Copy to ensure subgraph is editable
subgraph_nodes = list(subgraph.nodes())

# Visualize the Bayesian Network
pos = nx.spring_layout(subgraph)
print("Positions computed for nodes.")

# Generate node colors
node_colors = []
for node in subgraph_nodes:
    if node == target_variable:
        node_colors.append('red')
    elif node in [var for var, _ in sorted_influences[:5]]:
        node_colors.append('orange')
    else:
        node_colors.append('lightblue')

print("Node colors assigned.")

# Ensure the length of node_colors matches the number of nodes
print("Number of node colors:", len(node_colors))

# Plot the graph
plt.figure(figsize=(15, 10))
try:
    # nx.draw_networkx(subgraph, pos, with_labels=True, node_size=300, node_color=node_colors, arrows=True, arrowsize=20)
    nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color=node_colors)
    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(subgraph, pos, arrowstyle='->', arrows=True, connectionstyle="arc3,rad=0.2")
    print("Bayesian Network with root causes highlighted.")
except Exception as e:
    print("Error during drawing:")
    print(e)

# Save the graph
plt.savefig("bayesian_network.png")
print("Graph saved as 'bayesian_network.png'.")

plt.show()

# Get the adjacency matrix
adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
print("Adjacency matrix:", adj_matrix)

# Create a DataFrame for better readability
adj_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)

# Display the adjacency matrix
print("Adjacency Matrix:")
print(adj_df)

print("Number of edges:", adj_matrix.sum())

# Save the adjacency matrix to a CSV file
adj_df.to_csv("adjacency_matrix.csv")
print("Adjacency matrix saved to 'adjacency_matrix.csv'.")

# Get distance between the learned adjacency matrix and the ground truth
from utils import get_distance, adj_padder
adj_matrix = adj_padder(adj_matrix)
distance = get_distance(adj_matrix)
print("Distance from ground truth:", distance)
