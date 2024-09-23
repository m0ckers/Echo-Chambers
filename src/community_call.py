import community_detection as cd
import utils as ut

def community(input_file_edges, type, plot_subgraph_bool = False):
    
    # Load the dataset and build the graph
    df = ut.load_dataframe(input_file_edges)
    graph = ut.build_graph(df)

    if type is None:

        louvain_results = cd.louvain(graph)
        async_fluid_results = cd.async_fluid(graph)

    if type == 'louvain':
        print('Louvain...')
        cd.louvain(graph, plot_subgraph_bool = plot_subgraph_bool)

    elif type == 'metis':
        print('Metis...')
        cd.metis_fun(graph, plot_subgraph_bool = plot_subgraph_bool)    

    elif type == 'async':
        print('Async clustering...')
        cd.async_fluid(graph, plot_subgraph_bool = plot_subgraph_bool)

    else:
        print(f"Error: Unknown clustering type '{type}'. Please use 'kmeans', 'spectralclustering', or 'agglomerative'.")