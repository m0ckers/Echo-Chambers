import utils
import sys
import pandas as pd

def preprocessing_operation():
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/brexit_stances.tsv'
    df = pd.read_csv(input_file, sep='\t', header=0, usecols=['user', 'tweets', 'stance'])
    df1 = utils.preprocess(df, 'tweets', 'stance', "|")
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/brexit_edges.csv'
    df_edge = pd.read_csv(input_file, sep= ',', header=None, usecols= [0,1])
    df1 = utils.sentiment_roberta(df1, df_edge)
    from utils import build_graph_brexit 
    graph, link_adj, edg_adj = build_graph_brexit(df1)
    print("PREPROCESSING DONE")
    print("")

#def community_detection():
    from utils import louvain 
    louvain(graph, df)

def clustering():
    import utils
    from gensim.models import Word2Vec
    model = Word2Vec.load("/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/embeddings_weight.model")
    node_embeddings = model.wv
    feature_vectors = utils.feature_vector(node_embeddings)
    utils.clustering(feature_vectors, 2)


if __name__ == '__main__':
    preprocessing_operation()
    #community_detection()
    clustering()