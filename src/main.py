import utils
import sys
import pandas as pd
import argparse
import pathlib
import os

def preprocessing_operation(input, input_edges):
    if input:
        df = pd.read_csv(input, sep='\t', header=0, usecols=['user', 'tweets', 'stance'])
        df = utils.preprocess(df, 'tweets', 'stance', "|")
        input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/brexit_edges.csv'
        df_edge = pd.read_csv(input_edges, sep= ',', header=None, usecols= [0,1])
        df1 = utils.sentiment_roberta(df, df_edge)
        from utils import build_graph 
        graph, link_adj, edg_adj = build_graph(df1, df)
        print("PREPROCESSING DONE")
        print("")
        #def community_detection():
        utils.louvain(graph)
    elif input =='vaccination2.csv':
        df = pd.read_csv(input, sep= ',', usecols=['user_id','username', 'tweet', 'reply_to'])
        df = utils.preprocess_vaccination(df)
        df1 = utils.sentiment_roberta_vaccination(df)
        from utils import build_graph 
        graph, link_adj, edg_adj = build_graph(df1, df)
        print("PREPROCESSING DONE")
        print("")

def clustering(type):
    import utils
    from gensim.models import Word2Vec
    model = Word2Vec.load("/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/embeddings_weight.model")
    node_embeddings = model.wv
    feature_vectors = utils.feature_vector(node_embeddings)
    utils.clustering(feature_vectors, 2, type)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_edges', type=str, help='Input edges file name', default='brexit_edges.csv')
    parser.add_argument('input_file', type=str, help='Input node file name', default='brexit_stances.tsv')
    parser.add_argument('--output', type=str, default='output.txt', help='Output file name (default: output.txt)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--seed', type=int, nargs='*', default=[325, 350, 375, 400, 450, 500, 525, 550, 575, 601])
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--type', type=str, default=None, help='Choose clustering technique [kmeans,spectralclustering,agglomerative]')
    args = parser.parse_args()

    input_file = args.input_file
    parent_directory = os.path.dirname(pathlib.Path.cwd())
    input = os.path.join(parent_directory, 'Datasets', input_file)
    input_file_edges = args.input_file_edges
    parent_directory = os.path.dirname(pathlib.Path.cwd())
    input_edges = os.path.join(parent_directory, 'Datasets', input_file_edges)
    #input = f'{pathlib.Path.cwd()}/{input_file}'
    preprocessing_operation(input, input_edges)
    clustering(args.type)
if __name__ == '__main__':
    main()
    #community_detection()
    