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
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#The stances
#1 = Leave = Positive
#-1 = 2 = Remain = Negative
global column_text
global column_stance
global delimiter
def load(type):
    curr_dir = pathlib.Path.cwd()
    data_dir = curr_dir.parent / 'Datasets' / type
    data_dir

#Splitting tweets for every user
def split_dataframe(df, column_name, delimiter):
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

def preprocess(df, column_text, column_stance, delimiter):
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

def preprocess_vaccination(df):
    import json
    #Extract edges between nodes
    origine = []
    dest = []
    for item in range(len(df['reply_to'])):
        data_str = df['reply_to'].iloc[item]
        data_str = data_str.replace("'", '"')
        data_list = json.loads(data_str)
        user_ids = [entry['user_id'] for entry in data_list]
        i=0
        for elem in range(len(user_ids)-1):
            origine.append(user_ids[i])
            dest.append(user_ids[elem+1])
    #Creat elist of weighted edges
    source  = []
    target = []
    weight = []
    column_tuples = [(val1, val2) for val1, val2 in zip(origine, dest)]

    from collections import Counter
    tuple_counter = Counter(column_tuples)
    for tuple_value, count in tuple_counter.items():
        source.append(int(tuple_value[0]))
        target.append(int(tuple_value[1]))
        weight.append(count)
    df1 = pd.DataFrame(data = np.column_stack([source,target,weight]), columns= ['source','target','weight'])
    #Assign new nodes numbers
    # Extract unique nodes from both columns
    unique_nodes = pd.concat([df1['source'], df1['target']]).unique()
    # Create a mapping from old node IDs to new node IDs
    node_mapping = {old_node: new_node for new_node, old_node in enumerate(unique_nodes)}
    # Apply the mapping to the 'Source' and 'Target' columns
    df1['source'] = df1['source'].map(node_mapping)
    df1['target'] = df1['target'].map(node_mapping)
    print(f'New dataframe with shape:{df1.shape}')
    sns.histplot(data=df1, y="weight", bins=50, kde=True)
    sns.histplot(df1[df1['weight'] != 1], y="weight", bins=50, kde=True)
    return(df1)

def split_train_test(df):
    # On pre-processed tweets
    from sklearn.model_selection import train_test_split
    labels = df['Stance']
    train_text, test_text, train_labels, test_labels = train_test_split(df['Preprocessed'], df['stance'], random_state=2018,
                                                                        test_size=0.2,
                                                                        stratify=labels)
    train_labels = train_labels.tolist()
    test_labels = test_labels.tolist()
    # Get the lists of sentences and their labels.
    print(len(train_labels), len(train_text))
    return(train_text, train_labels)
#Calculate sentiment for each user
def sentiment_roberta(df_stance, df_edge):
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
    print('Some Viz!')
    sns.histplot(data=df1, y="weight", bins=50)
    print()
    sns.histplot(data=df1[df1['weight']!= 1], y="weight", bins=50)
    return(df1)
def stance_bert():
    ...
#Build graph
def build_graph(df):
    # Add edges with weights to the graph
    graph = nx.from_pandas_edgelist(df, 'source', 'target', 'weight')
    graph = graph.to_undirected()
    link_adj = nx.to_numpy_array(graph)
    edge_list = np.array(list(map(lambda x: list(x), list(graph.edges()))))
    # Print the list of edges with weights
    print(graph)
    print(f"Graph is weighted : {nx.is_weighted(graph)}")
    return(graph, link_adj, edge_list)

def calculate_conductance(graph, communities):
  conductance = 0
  for i, community in enumerate(communities):
      conductance_i = nx.algorithms.conductance(graph, community)
      conductance += conductance_i
      print(f"Community {i + 1} Conductance: {conductance_i}")

  print(f"Total Conductance: {conductance}")

def networkx_to_metis(graph):
    metis_data = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        metis_data.append(neighbors)

    return metis_data

def community_purity(df_stance, communities):
  purity_scores = []
  j = 0
  for community in communities:
      # Create a Counter object to count label frequencies within the community
      label_counts = Counter()
      i = 0
      j += 1
      # Iterate through nodes in the community
      for node in community:
          # Check if the node is present in the labels_df DataFrame
          if node in df_stance['user'].values:
              # Get the label of the current node from the DataFrame
              i += 1
              node_label = df_stance.loc[df_stance['user'] == node, 'stance'].values[0]
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

def louvain(graph, df, resolution):
    communities = list(nx.community.louvain_communities(graph, weight = 'weight', resolution=0.5))
    print(f'Modularity:{nx.community.modularity(graph, communities, weight="weight",resolution=1)}')
    print('Conductance____')
    calculate_conductance(graph, communities)
    print('Purity____')
    community_purity(df, communities)

def feature_vector(node_embeddings):
    from gensim.models import KeyedVectors
    # Initialize an empty list to store the feature vectors
    feature_vectors = []
    # Iterate through the keys (nodes) in the KeyedVectors object
    for node in node_embeddings.index_to_key:
        vector = node_embeddings[node]  # Access the vector for the current node
        feature_vectors.append(vector)  # Append the vector to the list
    return(feature_vectors)

def cluster_measures(feature_vectors, cluster_labels):
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.metrics import davies_bouldin_score
    silhouette_avg = silhouette_score(feature_vectors, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg}")
    # Calculate the Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(feature_vectors, cluster_labels)
    print(f"Calinski-Harabasz Index: {ch_score}")
    # Calculate the Davies-Bouldin Index
    db_score = davies_bouldin_score(feature_vectors, cluster_labels)
    print(f"Davies-Bouldin Index: {db_score}")

def clustering(feature_vectors, num_clusters, type):
    if type == 'kmeans' or type is None:
    # Create a KMeans clustering model
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        # Fit the model to your feature vectors
        cluster_labels = kmeans.fit_predict(feature_vectors)
        cluster_labels
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(feature_vectors)
        # Assuming 'cluster_labels' contains the cluster assignments for each data point
        plt.figure(figsize=(10, 6))

        for cluster_id in set(cluster_labels):
            # Filter data points belonging to the current cluster
            cluster_data = reduced_data[cluster_labels == cluster_id]
            # Plot data points of the current cluster with a unique color
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_id}', alpha=0.7)
        #Visualize and measure
        plt.title('Cluster Visualization')
        plt.legend()
        plt.show()
        print('Some measures...')
        cluster_measures(feature_vectors, cluster_labels)
        
    elif type == 'spectralclustering' or type is None:
        print('Spectral clustering...')
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_similarity_matrix = cosine_similarity(feature_vectors)
        sc = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='discretize')
        # Assuming you have cluster assignments and cosine similarity matrix
        cluster_labels = sc.fit_predict(cosine_similarity_matrix)
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(cosine_similarity_matrix)
        # Create a DataFrame with cluster labels and reduced data
        df_spectral = pd.DataFrame({'Cluster': cluster_labels, 'Feature_0': reduced_data[:, 0], 'Feature_1': reduced_data[:, 1]})
        # Plot the clusters in the reduced space
        sns.scatterplot(x='Feature_0', y='Feature_1', hue='Cluster', data=df_spectral, palette='Set1')
        plt.title('Spectral Clustering Results (t-SNE Visualization)')
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.show()
        print('Some measures...')
        cluster_measures(cosine_similarity_matrix, cluster_labels)
        
    elif type == 'agglomerative' or type is None:
        print('Hierarchical clustering...')
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering().fit(feature_vectors)
        cluster_labels = clustering.labels_
        print('Some measures...')
        cluster_measures(feature_vectors, cluster_labels)
    