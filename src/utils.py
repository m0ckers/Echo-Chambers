import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from collections import Counter
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import networkx as nx
import os
import pathlib
from node2vec import Node2Vec
import numpy as np
import ev_metrics
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import logging

#The stances
#1 = Leave = Positive
#-1 = 2 = Remain = Negative
global column_text
global column_stance
global delimiter
# Set the log level to ERROR
logging.basicConfig(level=logging.ERROR)
import warnings
# Filter out FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
#Import datasets
def ds():
    #Import VaxNoVax
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/Vaxnovax_edges.txt'
    df_edge_vax = pd.read_csv(input_file, sep= ' ', header=None, usecols= [0,1])
    #Import brexit edges
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/Brexit_edges.csv'
    df_edge_brexit = pd.read_csv(input_file, sep= ',', header=None, usecols= [0,1,3])
    #Import brexit stances
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/brexit_stances.tsv'
    df_stance_brexit = pd.read_csv(input_file, sep='\t', header=0, usecols=['tweets', 'stance','user'])
    #Import vaccination2
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/data_vaccination_sentiment.csv'
    df_vax2 = pd.read_csv(input_file, sep= ',')
    df_vax2['new_weights'] = df_vax2['new_weights'].astype(float)
    # Covid-19
    csv_file_path = "/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/Final_data.csv"
    df_covid = pd.read_csv(csv_file_path, usecols = ['username', 'favorites', 'retweets', '@mentions', 'geo', 'text_con_hashtag'])
    pd.set_option('display.max_colwidth', 100)
    graph_covid = nx.read_gml("/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/Final_Graph_Covid.gml")
    #NET SOTA
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/retweet_graph_nepal_threshold_largest_CC.txt'
    df_net = pd.read_csv(input_file, header=None)
    return(df_edge_brexit, df_stance_brexit, df_edge_vax, df_vax2, graph_covid, df_net)

#Build graphs
def graphs():
    #Brexit no weight
    datasets = ds()
    graph = nx.from_pandas_edgelist(datasets[0], 0, 1)
    graph1 = graph.to_undirected()
    graph1.name = 'Brexit no weight'
    # Print the list of edges with weights

    print(graph)
    print(f"Graph is weighted : {nx.is_weighted(graph)}")

    #Brexit
    df = preprocess(datasets[1], 'tweets', 'stance', "|")
    df1 = sentiment_roberta(df, datasets[0])
    from utils import build_graph 
    graph2, link_adj, edg_adj = build_graph(df1, df)
    graph2 = graph.to_undirected()
    graph2.name = 'Brexit weight'

    #VaxnoVax
    graph = nx.from_pandas_edgelist(datasets[2], 0, 1)
    graph3 = graph.to_undirected()
    graph3.name = 'VaxnoVax'

    # Vax2
    graph = nx.from_pandas_edgelist(datasets[3], 'source', 'target', edge_attr='new_weights')
    graph4 = graph.to_undirected()
    graph4.name = 'Vaccination Kaggle'

    #Covid 
    graph = datasets[4]
    graph5 = nx.convert_node_labels_to_integers(graph)
    df_net = datasets[5].rename(columns={2: "edge_weight_attr"})
    graph5.name = 'Covid'

    #Nethanyau
    graph = nx.from_pandas_edgelist(df_net, 0, 1, "edge_weight_attr")
    graph = nx.convert_node_labels_to_integers(graph)
    graph6 = graph.to_undirected()
    graph6.name = 'Nethanyau Soa'

    #graph_list = [globals()[f'graph{i}'] for i in range(1, 6)]
    return(graph1, graph2 , graph3 , graph4 , graph5 , graph6)

def emb():
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/Embeddings/embeddings_weight_brexit.model'

#Splitting tweets for every user
def split_dataframe(df, column_name, delimiter = None):
    new_rows = []

    for _, row in df.iterrows():
        segments = row[column_name].split(delimiter)

        for segment in segments:
            new_row = row.copy()
            new_row[column_name] = segment
            new_rows.append(new_row)

    split_df = pd.DataFrame(new_rows)
    split_df = split_df.reset_index(drop=True)

    return split_df

def preprocess(df, column_text, column_stance, delimiter = None):
    #Library to handle tweets
    def text_processor():
        text_processor = TextPreProcessor(
            segmenter="twitter",
            corrector="twitter",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize)
        return text_processor
    #Main pre process function
    def preprocess_tweet(tweet, min_length=4, max_length=10):
        # Remove URLs, mentions, special characters, numbers, and stuff between ()
        tweet = re.sub(r"http\S+|www\S+|https\S+|@\w+|[^a-zA-Z']|\(.*?\)", " ", tweet, flags=re.MULTILINE)
        # Tokenize the tweet
        tokens = word_tokenize(tweet)
        # Convert to lowercase
        tokens = [token.lower() for token in tokens]
        # Join the tokens back into a single string
        preprocessed_tweet = " ".join(tokens)
        return preprocessed_tweet
    
    if delimiter is not None:
        print('Need to split dataframe')
        df = split_dataframe(df, column_text, delimiter)
    else: 
        print('No need to split dataframe')

    preprocessed_texts = []
    text_processor_instance = text_processor()
    print('Now we preprocess')
    for text in df[column_text]:
        preprocessed_text = ' '.join(text_processor_instance.pre_process_doc(text))
        preprocessed_text = preprocess_tweet(preprocessed_text)
        preprocessed_texts.append(preprocessed_text)

    labels = list(df[column_stance])
    labels = [2 if label == -1 else label for label in labels]
    #Create new columns for labels and text and labels Text
    df['labels'] = labels
    df['preprocessed'] = preprocessed_texts
    df['labels text'] = ['Negative' if label == 2 else 'Neutral' if label == 0 else 'Positive' for label in df['labels']]
    df = df[['user', 'tweets', 'preprocessed', 'labels', 'labels text']]
    df = df.reset_index(drop=True)
    return df

def sentiment_roberta(df_stance, df_edge = None):
    users = df_stance.columns[0]
    tweets = df_stance.columns[1]
    labels = df_stance.columns[3]
    labels_text = df_stance.columns[4]
    delimiter = "|"
    def preprocess_sentiment(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    print(f'Text is in column: {tweets}, stance in column: {labels}')
    if any(delimiter in tweet for tweet in df_stance[tweets]):
        df = split_dataframe(df_stance, tweets, delimiter)
    else: 
        print('Feeling good..;)')
    #Prepare model
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    #PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    #Calculate sentiment for each user
    unique_users = df_stance[users].unique()
    sentiments = []
    # Iterate over users to get sentiment
    print('Calculate sentiment for each user...')

    for user in unique_users:
        user_df = df_stance[df_stance[users] == user]
        sentiment = 0
        for elem in user_df[tweets]:
            elem = preprocess_sentiment(elem)
            result = sentiment_pipeline(elem)

            if result[0]['label'] == 'positive':
                sentiment += result[0]['score']
            elif result[0]['label'] == 'negative':
                sentiment += -(result[0]['score'])
            else:
                sentiment += 0

        sentiment = sentiment/len(user_df)
        sentiments.append(sentiment)

    data = {'user': unique_users, 'sentiment': sentiments}
    df = pd.DataFrame(data)
    print('Done!')
    #Put weights on node pairs
    print('Creating weights')

    if df_edge is not None:
        edge_weights = {}
        # Iterate through the DataFrame rows and update edge weights
        for _, row in df_edge.iterrows():
            source = row[0]
            target = row[1]
            # Check if the edge already exists in the dictionary
            if (source, target) in edge_weights:
                edge_weights[(source, target)] += 1
            else:
                edge_weights[(source, target)] = 1

        #Add weights based on the stance

        for key, value in edge_weights.items():
            if key[0] in df[users].values or key[1] in df[users].values:
                #Both nodes have text
                if key[0] in df[users].values and key[1] in df[users].values:
                    s = df[df[users] == key[0]]['sentiment'].iloc[0]
                    t = df[df[users] == key[1]]['sentiment'].iloc[0]
                    edge_weights[key] +=  value + abs(s + t)
                #Just one node has text
                elif key[0] in df[users].values and key[1] not in df[users].values:
                    edge_weights[key] +=  value + abs(df[df[users] == key[0]]['sentiment'].iloc[0])
                elif key[1] in df[users].values and key[0] not in df[users].values:
                    edge_weights[key] +=  value + abs(df[df[users] == key[1]]['sentiment'].iloc[0])
                #No node has text    
                else:
                    edge_weights[key] = value

        # Create a list of edges with weights
        edges_with_weights = [(source, target, weight) for (source, target), weight in edge_weights.items()]
        df1 = pd.DataFrame(edges_with_weights, columns = ['source', 'target', 'weight'])
        print('Done!')

    else:
        print('Some Viz!')
        sns.histplot(data=df1, y="weight", bins=50)
        print()
        sns.histplot(data=df1[df1['weight']!= 1], y="weight", bins=50)

    return(df1)

def preprocess_sentiment(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def sentiment_roberta_vaccination(df1):
    # Prepare model
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    if torch.cuda.is_available():
        model = model.to('cuda')

    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Calculate sentiment for each user
    from tqdm import tqdm
    print('Calculate sentiment for each user...')
    sentiments = []
    sentiment = 0  # Initialize sentiment score
    prev_tweet = None  # Initialize previous tweet as None

    for _, row in tqdm(df1.iterrows(), total=len(df1)):
        tweet = row['Preprocessed']

        if tweet != prev_tweet:
          sentiment = 0
          prev_tweet = tweet  # Update previous tweet
          result = sentiment_pipeline(tweet)
          if result[0]['label'] == 'positive':
            sentiment += result[0]['score']
          elif result[0]['label'] == 'negative':
            sentiment -= result[0]['score']
          else:
            sentiment = 0

          sentiments.append(sentiment)
        else:
            sentiments.append(sentiment)

        df1['sentiments'] = sentiments
        df1 = df1.groupby(['source', 'target'])[['weight', 'sentiments']].sum()
        df1.reset_index(inplace=True)
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()

        df1['sentiments'] = scaler.fit_transform(df1[['sentiments']])

        first = [float(x) for x in df1['weight'].tolist()]
        second = [float(y) for y in df1['sentiments'].tolist()]
        sum_result = [x * y if y != 0 else x + y for x, y in zip(first, second)]

        df1['new_weights'] = sum_result
        df1['unordered_pair'] = df1.apply(lambda row: tuple(sorted([row['source'], row['target']])), axis=1)

        # Group by the unordered pair column and sum the other columns
        grouped = df1.groupby('unordered_pair').agg({
            'source': 'first',  # Just keep one of the source values (or use 'first' or 'last' as per your need)
            'target': 'first',   # Just keep one of the target values (or use 'first' or 'last' as per your need)
            'new_weights': 'sum'  # Sum the 'value' column
        }).reset_index()

        # Drop the 'unordered_pair' column if not needed
        grouped = grouped.drop(columns='unordered_pair')
        df1 = grouped
        df1.head()

def node2vec(graph):
    def visualize():
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # Assuming you have a list of node embeddings (e.g., node_embeddings_list) and their corresponding node IDs
        node_ids = node_embeddings.index_to_key
        node_vectors = [node_embeddings[node_id] for node_id in node_ids]

        # Apply PCA to reduce the dimensionality to 2D
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(node_vectors)

        # Create a scatter plot of the reduced embeddings
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title('PCA Visualization of Node Embeddings')
        plt.show()
    # Generate random walks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    node2vec = Node2Vec(graph, dimensions=128, walk_length=15, num_walks=50, p=0.5, q=2, workers=2)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Access the node embeddings
    node_embeddings = model.wv
    visualize()

def networkx_to_metis(graph):
    metis_data = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        metis_data.append(neighbors)

    return metis_data

def filter_graph(graph, threshold = 2):
  G = graph.copy()

  average_degree = sum(dict(G.degree()).values()) / len(G)

  # Filter edges with weights greater than or equal to the threshold
  filtered_edges = [(u, v, data) for u, v, data in G.edges(data=True) if 
  G.degree(u) > average_degree * threshold or G.degree(v) > average_degree * threshold]

  # Create a subgraph containing only the filtered edges
  filtered_G = nx.Graph()
  filtered_G.add_edges_from(filtered_edges)

  # If you want to add the same node attributes or other data, copy them from the original graph if needed
  filtered_G.add_nodes_from((n, G.nodes[n]) for n in filtered_G.nodes)
  H = filtered_G

  components = nx.connected_components(H)
  largest_component = max(components, key=len)
  H = G.subgraph(largest_component)
  H.number_of_nodes(), H.number_of_edges()

  mapping = {node: idx for idx, node in enumerate(H.nodes())}

  # Rename nodes to integer labels
  H = nx.relabel_nodes(H, mapping)

  # Print transformed nodes
  print("Transformed nodes:", list(H.nodes()))

  for u, v, data in H.edges(data=True):
      data[2] = float(round(data[2]))  # Convert weight to integer

  # Print transformed edge weights
  print("Transformed edge weights:", nx.get_edge_attributes(H, 'new_weights'))
  print(H.number_of_nodes(), H.number_of_edges())

  return H

def average_degree(graph):
  average_degree = sum(dict(graph.degree()).values()) / len(graph)
  return average_degree

def df_info_function(cluster_sizes, measures_result, community_result, method):
    # Initialize an empty DataFrame with column names
    df_info = pd.DataFrame(columns= ["Method"] + ["Community " + str(i) for i in range(2)] + ["Modularity", "Silhouette", "Calinski-Harabasz Index",
                                                                                "David-Bouldain Index","Conductance"])
    # Prepare data for appending
    data_to_append = {
        "Method": method,
        **{f"Community {i}": cluster_sizes.get(i, 0) for i in range(2)},  # Dynamically handle community sizes
        "Modularity": round(community_result[0], ndigits=2) if community_result else None,
        "Silhouette": round(measures_result[0], ndigits=2) if measures_result else None,
        "Calinski-Harabasz Index": round(measures_result[1], ndigits=2) if measures_result else None,
        "David-Bouldain Index": round(measures_result[2], ndigits=2) if measures_result else None,
        "Conductance": community_result[1] if community_result and len(community_result) > 1 else None
        }

    # Append to the DataFrame
    df_info = pd.concat([df_info, pd.DataFrame([data_to_append])], ignore_index=True)

    return df_info

def load_embeddings(path):
    from gensim.models import Word2Vec
    return np.load(path)

def load_dataframe(file_path, sep=',', dtype=None):
    file_type = os.path.splitext(file_path)[1].lower()

    if file_type in ['.csv']:
        # Load the file into a DataFrame
        df = pd.read_csv(file_path, sep=',', header = None, dtype=dtype)
        df = df.rename(columns={df.columns[0]: 0, df.columns[1]: 1,df.columns[2]: 2})
    elif file_type in ['.tsv']:
        df = pd.read_csv(file_path, sep=' ', header = None, dtype=dtype)
        df = df.rename(columns={df.columns[0]: 0, df.columns[1]: 1,df.columns[2]: 2})
    elif file_type in ['.txt']:
        df = pd.read_csv(file_path, sep=' ', header = None, dtype=dtype)
        df = df.rename(columns={df.columns[0]: 0, df.columns[1]: 1,df.columns[2]: 2})

    elif file_type == 'gml':
        # Load a graph from a GML file
        graph = nx.read_gml(file_path)
        return graph

    else:
        raise ValueError(f"Unsupported file type '{file_type}'. Use 'csv', 'tsv', 'txt', or 'gml'.")
    return df
    
def build_graph(df, edge_columns=(0, 1), df_stance=None, name=None):
    # Ensure the dataframe has the expected columns for edges
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two columns for edges.")
    
    # Create a graph from the DataFrame
    graph = nx.from_pandas_edgelist(df, source=0, target=1)
    graph = graph.to_undirected()

    # Optional: Set the graph name if provided
    if name:
        graph.name = name
    
    return graph



















