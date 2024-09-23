from clustering import kmeans, spectral, agglomerative
import utils as ut

def clustering(model=None, node_embeddings=None, df=None, num_clusters=2, plot_type=None, type=None):
    
    node_embeddings_c = ut.load_embeddings(node_embeddings)
    df_c = ut.load_dataframe(df)

    if type is None:

        kmeans(model_c=model, node_embeddings_c=node_embeddings_c, df_c=df_c, num_clusters_c=num_clusters, plot_type_c=plot_type)
        spectral(model_c=model, node_embeddings_c=node_embeddings_c, df_c=df_c, num_clusters_c=num_clusters, plot_type_c=plot_type)
        agglomerative(model_c=model, node_embeddings_c=node_embeddings_c, df_c=df_c, num_clusters_c=num_clusters, plot_type_c=plot_type)

    if type == 'kmeans':
        print('KMeans clustering...')
        kmeans(model_c = model, node_embeddings_c = node_embeddings_c, df_c=df_c, num_clusters_c=num_clusters, plot_type_c=plot_type)

    elif type == 'spectralclustering':
        print('Spectral clustering...')
        spectral(model_c=model, node_embeddings_c=node_embeddings_c, df_c=df_c, num_clusters_c=num_clusters, plot_type_c=plot_type)
        
    elif type == 'agglomerative':
        print('Hierarchical clustering...')
        agglomerative(model_c=model, node_embeddings_c=node_embeddings_c, df_c=df_c, num_clusters_c=num_clusters, plot_type_c=plot_type)

    else:
        print(f"Error: Unknown clustering type '{type}'. Please use 'kmeans', 'spectralclustering', or 'agglomerative'.")
