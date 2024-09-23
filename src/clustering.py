from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ev_metrics as ev
import utils
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import plots


def kmeans(model_c = None, node_embeddings_c = None, df_c = None, num_clusters_c = 2, plot_type_c = None):
  
  if model_c is not None:
      try:
          model_c.eval()
          with torch.no_grad():
              node_embeddings_c = model_c.embedding.weight.cpu().numpy()
      except:
          print("Error: Failed to extract node embeddings from the model.")

  if node_embeddings_c is None:
    print("Error: No node embeddings provided.")
    return

  # Create a KMeans clustering model
  kmeans_0 = KMeans(n_clusters = num_clusters_c, random_state = 0)
  cluster_labels = kmeans_0.fit_predict(node_embeddings_c)
  
  # Calculate the size of each cluster
  unique, counts = np.unique(cluster_labels, return_counts = True)
  cluster_sizes = dict(zip(unique, counts))
  print(f"Cluster sizes: {cluster_sizes}")

  if plot_type_c is not None:
    reduced_data = plots.plot_embeddings(node_embeddings_c, cluster_labels, plot_type_c)
  else:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(node_embeddings_c)

  return cluster_sizes, ev.measures(reduced_data, cluster_labels), ev.community_measures_cluster(cluster_labels, df_c) 

def spectral(model_c = None, node_embeddings_c = None, df_c = None, num_clusters_c = 2, plot_type_c = None):

  sc = SpectralClustering(num_clusters_c, affinity='precomputed', n_init=100, assign_labels='discretize')

  if model_c is not None:
      try:
          model_c.eval()
          with torch.no_grad():
              node_embeddings_c = model_c.embedding.weight.cpu().numpy()
      except:
          print("Error: Failed to extract node embeddings from the model.")

  if node_embeddings_c is None:
    print("Error: No node embeddings provided.")
    return

  #Cosine similarity matrix
  cosine_similarity_matrix = cosine_similarity(node_embeddings_c)
  cosine_similarity_matrix = np.nan_to_num(cosine_similarity_matrix)
  
  min_value = np.min(cosine_similarity_matrix)
  # Add the absolute value of the minimum value to make all values non-negative
  cosine_similarity_matrix += np.abs(min_value/2)
  
  svd = TruncatedSVD(n_components = 5)
  reduced_cosine_matrix = svd.fit_transform(cosine_similarity_matrix)

  # Assuming you have cluster assignments and cosine similarity matrix
  cluster_labels = sc.fit_predict(cosine_similarity_matrix)

  unique, counts = np.unique(cluster_labels, return_counts=True)
  cluster_sizes = dict(zip(unique, counts))
  print(f"Cluster sizes: {cluster_sizes}")

  if plot_type_c is not None:
    reduced_data = plots.plot_embeddings(node_embeddings_c, cluster_labels, plot_type_c)
  else:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(node_embeddings_c)

  return cluster_sizes, ev.measures(reduced_data, cluster_labels), ev.community_measures_cluster(cluster_labels, df_c), 'Spectral'

def agglomerative(model_c = None, node_embeddings_c = None, df_c = None, num_clusters_c = 2, plot_type_c = None):
  
  if model_c is not None:
      try:
          model_c.eval()
          with torch.no_grad():
              node_embeddings_c = model_c.embedding.weight.cpu().numpy()
      except:
          print("Error: Failed to extract node embeddings from the model.")

  if node_embeddings_c is None:
    print("Error: No node embeddings provided.")
    return

  pca = PCA(n_components=2)
  reduced_data = pca.fit_transform(node_embeddings_c)

  clustering = AgglomerativeClustering(n_clusters=num_clusters_c, linkage='ward').fit(reduced_data)
  cluster_labels = clustering.labels_

  # Calculate the size of each cluster
  unique, counts = np.unique(cluster_labels, return_counts=True)
  cluster_sizes = dict(zip(unique, counts))
  print(f"Cluster sizes: {cluster_sizes}")

  if plot_type_c is not None:
    reduced_data = plots.plot_embeddings(node_embeddings_c, cluster_labels, plot_type_c)

  else:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(node_embeddings_c)

  return cluster_sizes, ev.measures(reduced_data, cluster_labels), ev.community_measures_cluster(cluster_labels, df_c), 'Agglomerative'
