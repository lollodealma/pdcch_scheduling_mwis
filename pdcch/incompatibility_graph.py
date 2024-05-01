from pdcch.search_space import search_space_start_cce
from mwis.utils import do_intersect
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def get_incompatibility_graph(users, mode='DL', do_plot=False):
    color = plt.cm.rainbow(np.linspace(0, 1, len(users)))
    G = nx.Graph()

    # add NODES
    node_id = 0
    color_per_node = []
    for i_ue, ue in enumerate(users):
        for ii, start_cce in enumerate(ue.search_space):
            G.add_node(f'{ue.rnti}-{start_cce}-{ue.al}', rnti=ue.rnti, mode=mode, start_cce=start_cce,
                       al=ue.al, weight=ue.pdcch_weight)
            node_id += 1
            color_per_node.append(color[i_ue])

    # add EDGES
    rnti_per_node = nx.get_node_attributes(G, 'rnti')
    mode_per_node = nx.get_node_attributes(G, 'mode')
    start_cce_per_node = nx.get_node_attributes(G, 'start_cce')
    al_per_node = nx.get_node_attributes(G, 'al')
    for node1 in G.nodes():
        rnti_node1 = rnti_per_node[node1]
        mode_node1 = mode_per_node[node1]
        cces_node1 = np.arange(start_cce_per_node[node1], start_cce_per_node[node1] + al_per_node[node1])
        for node2 in G.nodes():
            if node1 != node2:
                if not(G.has_edge(node1, node2)):
                    if (rnti_node1 == rnti_per_node[node2]) & (mode_node1 == mode_per_node[node2]):
                        # add edge between nodes belonging to the same (RNTI, mode)
                        G.add_edge(node1, node2)
                    else:
                        # add edge if the candidates overlap on at least one CCE
                        cces_node2 = np.arange(start_cce_per_node[node2], start_cce_per_node[node2] + al_per_node[node2])
                        if do_intersect(cces_node1, cces_node2):
                            G.add_edge(node1, node2)
    if do_plot:
        nx.draw(G, with_labels=True, node_color=color_per_node)
    return G

