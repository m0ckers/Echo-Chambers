import utils
import sys
import argparse
import pathlib
import os
import clustering_call
import community_call

def main():
    # Argument Parser Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_edges', type=str, help='Input edges file name', default='brexit_edges.csv')
    parser.add_argument('--input_file_text', type=str, help='Input node file name', default='brexit_stances.tsv')
    parser.add_argument('--input_node_embeddings', type=str, help='Input node embeddings', default='node_embeddings_brexit_unweighted.npy')
    parser.add_argument('--output', type=str, default='output.txt', help='Output file name (default: output.txt)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--N', type=int, default=256, help="Number of components (default: 256)")
    parser.add_argument('--plot_subgraph', type=str, default= False, 
                        help='want a plot?')
    parser.add_argument('--type_community', type=str, default='louvain', 
                        help='Choose community detection technique [louvain, metis, async]')
    parser.add_argument('--type_clustering', type=str, default='kmeans', 
                        help='Choose clustering technique [kmeans, spectralclustering, agglomerative]')

    args = parser.parse_args()

    # Path Setup
    parent_directory = os.path.dirname(pathlib.Path.cwd())
    input_file_edges = os.path.join(parent_directory, 'Datasets', args.input_file_edges)
    input_file_text = os.path.join(parent_directory, 'Datasets', args.input_file_text)
    input_node_embeddings = os.path.join(parent_directory, 'Datasets', args.input_node_embeddings)

    # Verbose Logging
    if args.verbose:
        print(f"Input Text File: {input_file_text}")
        print(f"Input Edges File: {input_file_edges}")
        print(f"Clustering Type: {args.type}")
        print(f"Output File: {args.output}")

    try:

        louvain_result = community_call.community(input_file_edges, 
                                                  type = args.type_community, 
                                                  plot_subgraph_bool = args.plot_subgraph) 

        # Clustering (pass the clustering type and other parameters)
        clustering_result = clustering_call.clustering(node_embeddings=input_node_embeddings, 
                                                       type=args.type_clustering, 
                                                       df=input_file_edges)
        print(clustering_result)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
