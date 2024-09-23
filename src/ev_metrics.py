from collections import Counter
import networkx as nx
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import numpy as np

def calculate_conductance(graph, communities):
  conductance_total = 0
  conductance = []
  for i, community in enumerate(communities):
      conductance_i = nx.algorithms.conductance(graph, community)
      conductance_total += conductance_i
      conductance.append(conductance_i)
      print(f"Community {i + 1} Conductance: {conductance_i}")

  print(f"Total Conductance: {conductance_total}")
  return conductance

def community_purity(graph, communities):
    purity_scores = []
    j = 0
    for community in communities:
        label_counts = Counter()
        i = 0
        j += 1
        # Iterate through nodes in the community
        for node in community:
            # Check if the node is present in the labels_df DataFrame
            if node in graph.nodes and graph.nodes[node]['stance'] != 0:
                # Get the label of the current node from the DataFrame
                i += 1
                node_label = graph.nodes[node]['stance']
                # Update the label counts
                label_counts[node_label] += 1
        # Calculate the purity of the community
        if not label_counts:
            purity_scores.append(0.0)  # Avoid division by zero
        else:
            # Find the most frequent label within the community
            max_label_count = max(label_counts.values())
            # Calculate the purity using the formula
            purity = max_label_count / i
            purity_scores.append(purity)
            print(f"Purity distribution for community {j}:{label_counts}")
    for i, purity in enumerate(purity_scores):
        print(f"Community {i + 1}: {purity:.2f}")
    return purity_scores

def cluster_measures_community(G, communities):
    # Map nodes to community IDs
    node_to_com = {}
    for i, com in enumerate(communities):
      for node in com:
          node_to_com[node] = i

    labels = [node_to_com[node] for node in G.nodes()]
    A = nx.to_scipy_sparse_array(G)

    # Assuming you have a list of feature vectors named 'feature_vectors'
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(A.todense())

    sil_score = silhouette_score(reduced_data, labels)
    ch_score = calinski_harabasz_score(reduced_data, labels)
    db_score = davies_bouldin_score(reduced_data, labels)

    return (sil_score, ch_score, db_score)

def community_measures(graph, communities):
  mod = nx.community.modularity(graph, communities)
  con = calculate_conductance(graph, communities)
  return mod, con

def overlapping(partitions):

    """Check for overlapping nodes between two partitions or subgraphs.
    Input: Communities list"""

    try:
        # Attempt to get the node lists from subgraphs, if they exist
        nodes_for_partition_1 = set(partitions[1].nodes())
        nodes_for_partition_0 = set(partitions[0].nodes())
    except AttributeError:
        # If the partitions are not subgraphs, they are already sets of nodes
        nodes_for_partition_1 = set(partitions[1])
        nodes_for_partition_0 = set(partitions[0])

    # Find overlapping nodes
    overlapping_nodes = nodes_for_partition_1.intersection(nodes_for_partition_0)

    # Check if there are any overlapping nodes and print the result
    if overlapping_nodes:
        print(f"There are overlapping nodes between the two partitions.")
    else:
        print(f"There are no overlapping nodes between the two partitions.")

    # Print the count of overlapping nodes
    count_of_overlapping_nodes = len(overlapping_nodes)
    print(f"Number of overlapping nodes: {count_of_overlapping_nodes}")

    # If you want to print the overlapping nodes themselves:
    if count_of_overlapping_nodes > 0:
        print(f"Overlapping nodes: {overlapping_nodes}")

def measures(reduced_data, cluster_labels):

  silhouette_avg = silhouette_score(reduced_data, cluster_labels)
  print(f"Silhouette Score: {silhouette_avg}")

  # Calculate the Calinski-Harabasz Index
  ch_score = calinski_harabasz_score(reduced_data, cluster_labels)
  print(f"Calinski-Harabasz Index: {ch_score}")

  # Calculate the Davies-Bouldin Index
  db_score = davies_bouldin_score(reduced_data, cluster_labels)
  print(f"Davies-Bouldin Index: {db_score}")

  return silhouette_avg, ch_score, db_score

def community_measures_cluster(cluster_labels, df):
    # Sample partition assignment list
    partition = cluster_labels

    # Initialize a dictionary to store communities
    communities_dict = {}

    # Iterate through the partition assignment list
    for i, community_id in enumerate(partition):
        if community_id not in communities_dict:
            communities_dict[community_id] = set()
        communities_dict[community_id].add(i)  # Add node index to the corresponding community

    # Convert the dictionary values to a list of sets
    communities_list = list(communities_dict.values())

    # Create an undirected graph from the dataframe
    nx_G = nx.from_pandas_edgelist(df, 0, 1)
    nx_G = nx_G.to_undirected()

    # Generate the communities list based on cluster labels
    communities_list = [set(np.where(cluster_labels == i)[0]) for i in range(len(set(cluster_labels)))]
    #print("Communities List:", communities_list)

    # Compute modularity
    mod = nx.community.modularity(nx_G, communities_list, resolution = 1)
    print(f'Modularity: {mod}')

    # Function to calculate conductance for each community
    def calculate_conductance(graph, communities):
        conductance_result = []
        total_conductance = 0

        for i, community in enumerate(communities):
            conductance_i = nx.algorithms.conductance(graph, community)
            print(f"Community {i + 1} Conductance: {conductance_i}")
            conductance_result.append(conductance_i)
            total_conductance += conductance_i

        #print(f"Total Conductance: {total_conductance}")
        return conductance_result

    # Calculate and return the conductance values
    conductance_result = calculate_conductance(nx_G, communities_list)

    # Return modularity and conductance results
    return mod, conductance_result