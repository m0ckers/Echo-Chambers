import ev_metrics as ev
import networkx as nx
#import metis
import utils as ut
import numpy as np
import plots as pf

def louvain(graph, weight=2, res=1, i=1, seed=10, mod=0, plot_subgraph_bool=False, max_iterations=15):
    print(f"Graph: {graph.name if hasattr(graph, 'name') else 'Unnamed'}")
    print(f"Iteration: {i}")

    # Run Louvain algorithm
    communities = list(nx.community.louvain_communities(graph, weight=weight, resolution=res, seed=seed))
    mod1 = nx.community.modularity(graph, communities, resolution=1)
    print(f"Louvain with resolution={res}, weight={weight}, modularity={mod1}, communities={len(communities)}")

    # If only one community is found, retry with higher resolution
    if len(communities) == 1 and i < max_iterations:
        return louvain(graph, weight, res + res / 2, i + 1, seed, mod1, plot_subgraph_bool, max_iterations)

    # Return the final communities if stopping condition met
    elif len(communities) == 2 or i >= max_iterations:
        print("Stopping condition met or max iterations reached")
        return process_final_communities(graph, communities, plot_subgraph_bool)

    # Recur with lower resolution
    elif len(communities) > 2:
        return louvain(graph, weight, res / 2, i + 1, seed, mod1, plot_subgraph_bool, max_iterations)
    
def process_final_communities(graph, communities, plot_subgraph_bool=False):
    cluster_sizes = [len(community) for community in communities]
    print(f"Largest communities sizes: {cluster_sizes}")

    # Calculate modularity and conductance
    final_mod, con = ev.community_measures(graph, communities)
    print(f"Modularity: {final_mod}, Conductance: {con}")

    # Calculate clustering scores
    score = ev.cluster_measures_community(graph, communities)
    print(f'Clustering scores: silhouette: {score[0]}, Calinski-Harabasz index: {score[1]}, Davies-Bouldin index: {score[2]}')

    # Optionally, plot the subgraph
    if plot_subgraph_bool:
        print('Plotting subgraph')
        pf.plot_subgraph(graph, communities)

    return cluster_sizes, score, [final_mod, con]

def metis_fun(graph, seed=10, plot_subgraph_bool = False):
        
        print(f"Graph Name: {graph.name}")

        # Convert NetworkX graph to a METIS graph
        metis_graph = metis.networkx_to_metis(graph)

        # Perform graph partitioning using METIS
        (edgecuts, parts) = metis.part_graph(metis_graph, nparts=2, seed=seed, recursive=True)

        # Print the results
        print("Edge Cuts:", edgecuts)
        print("Partition Assignment:", parts)

        # Create community list
        communities_dict = {i: set() for i in set(parts)}
        for node, community_id in enumerate(parts):
            communities_dict[community_id].add(node)

        # Convert the dictionary values to a list of sets
        communities_list = list(communities_dict.values())
        cluster_sizes = [len(communities_list[0]), len(communities_list[1])]
        print(f"Largest communities sizes: {cluster_sizes}")
        
        # Calculate final modularity
        final_mod, con = ev.community_measures(graph, communities_list)
        print(f"Modularity: {final_mod}")
        print(f"Conductance: {con}")

        print('Computing cluster measures')
        score = ev.cluster_measures_community(graph, communities_list)
        print(f'Clustering scores: silouhette: {score[0]}, Calinski-Harabasz index :{score[1]}, Davies-Bouldin index:{score[2]}')
        
        if plot_subgraph_bool:
            print('Plotting subgraph')
            pf.plot_subgraph(graph, communities_list)

        return cluster_sizes, score, [final_mod] + [con]

def async_fluid(graph, plot_subgraph_bool=False):
        print(f"Graph Name: {graph.name}")

        # Run asynchronous fluid communities
        com = nx.community.asyn_fluidc(graph, 2)
        communities_list = [set(community) for community in com]

        cluster_sizes = [len(community) for community in communities_list]
        print(f"Largest communities sizes: {cluster_sizes}")

        # Calculate modularity and conductance
        final_mod, con = ev.community_measures(graph, communities_list)
        print(f"Modularity: {final_mod}, Conductance: {con}")

        # Calculate clustering scores
        score = ev.cluster_measures_community(graph, communities_list)
        print(f'Clustering scores: silhouette: {score[0]}, Calinski-Harabasz index: {score[1]}, Davies-Bouldin index: {score[2]}')

        # Optionally, plot the subgraph
        if plot_subgraph_bool:
            print('Plotting subgraph')
            pf.plot_subgraph(graph, communities_list)

        return cluster_sizes, score, [final_mod, con]


