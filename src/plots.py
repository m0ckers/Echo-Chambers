import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_subgraph(Gp, lpc=None):
    G = Gp.copy()

    # Calculate average degree
    average_degree = sum(dict(G.degree()).values()) / len(G)
    print(f'Average degree is: {average_degree}')

    # Remove low-degree nodes
    low_degree = [n for n, d in G.degree() if d < average_degree]
    G.remove_nodes_from(low_degree)

    # Largest connected component
    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    H = G.subgraph(largest_component)

    # Compute centrality
    centrality = nx.betweenness_centrality(H)

    # Compute community structure
    if lpc is None:
        print('Calculating communities for you')
        lpc = nx.community.kernighan_lin_bisection(H)
        community_index = {n: i for i, com in enumerate(lpc) for n in com}
    else:
        print('You passed some communities')
        community_index = {n: i for i, com in enumerate(lpc) for n in com}

    #### Draw graph ####
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(H, k=0.15, seed=4572321)
    node_color = [community_index[n] for n in H]
    node_size = [v * 20000 for v in centrality.values()]

    # Draw the network
    nx.draw_networkx(
        H,
        pos=pos,
        with_labels=False,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
        cmap = plt.cm.jet
    )

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    ax.set_title("Graph Subsample Structure", font)

    # Add custom legend for node sizes
    size_legend_labels = ['Low Centrality', 'High Centrality']
    size_legend_patches = [patches.Patch(color='grey', label=label) for label in size_legend_labels]
    size_legend_patches[0].set_edgecolor('none')
    size_legend_patches[1].set_edgecolor('none')
    ax.legend(handles=size_legend_patches, loc='lower right', fontsize=12, title="Node Size", title_fontsize='13', shadow=True, fancybox=True)

    # Add custom legend for node colors
    color_legend_labels = [f'Community {i}' for i in set(node_color)]
    color_legend_patches = [patches.Patch(color=plt.cm.jet(i / max(node_color)), label=label) for i, label in enumerate(color_legend_labels)]
    ax.legend(handles=color_legend_patches, loc='upper right', fontsize=12, title="Communities", title_fontsize='13', shadow=True, fancybox=True)

    # Resize figure for label readability
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.show()