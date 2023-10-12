def build_graph(df, df1):
    import pandas as pd
    
    edge_weights = {}
    # Iterate through the DataFrame rows and update edge weights
    for _, row in df.iterrows():
        source = row[0]
        target = row[1]

        # Check if the edge already exists in the dictionary
        if (source, target) in edge_weights:
            edge_weights[(source, target)] += 1
        else:
            edge_weights[(source, target)] = 1

    #Add weights based on the stance
    for key, value in edge_weights.items():
        if key[0] in df1['user'].values:
            edge_weights[key] =  value + abs(df1[df1['user'] == key[0]]['stance'].iloc[0])
        else:
            edge_weights[key] = value


    # Create a list of edges with weights
    edges_with_weights = [(source, target, weight) for (source, target), weight in edge_weights.items()]
    df1 = pd.DataFrame(edges_with_weights, columns = ['source', 'target', 'weight'])
        
    import networkx as nx

    # Add edges with weights to the graph
    graph = nx.from_pandas_edgelist(df1, 'source', 'target', 'weight')
    graph = graph.to_undirected()
    # Print the list of edges with weights

    print(graph)
    print(f"Cluster is weighted : {nx.is_weighted(graph)}")