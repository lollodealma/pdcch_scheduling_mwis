import networkx as nx
import pandas as pd
import numpy as np


def setdiff_keepsort(b, a):
    # computes b setminus a while keeping the sorting of b
    return np.array(b)[~np.in1d(b, a)]


def get_degree(G):
    """
    :param G: networkx graph
    :return: ndegree for each node
    """
    deg_per_node = pd.Series(index=list(G.nodes()), dtype=int)
    for node in G.nodes():
        deg_per_node[node] = G.degree[node]
    return deg_per_node


def get_weights(G):
    weight_per_node = pd.Series(index=list(G.nodes()), dtype=int)
    for d in G.nodes.data():
        if 'weight' in d[1].keys():
            weight_per_node.loc[d[0]] = d[1]['weight']
        else:
            weight_per_node.loc[d[0]] = 1
    return weight_per_node


def has_nodes(G):
    return len(list(G.nodes())) > 0


def get_rand_node(G):
    return np.random.choice(list(G.nodes()))


def get_node_weight(G, node):
    return G.nodes.data()[node]['weight']


def get_hist(vec):
    vec = np.array(vec)
    hist = pd.Series(index=np.unique(vec), dtype=int)
    for v in hist.index:
        hist[v] = np.sum(vec==v)
    return hist


def is_independent_set(G, nodes):
    """
    check if nodes form an independent set
    :param G:
    :param nodes:
    :return:
    """
    return nx.is_empty(G.subgraph(nodes))


def do_intersect(a, b):
    # check if intersection of lists a, b is non empty
    return bool(set(b) & set(a))


def get_leaf_nodes(G):
    if nx.is_directed(G):
        return [x for x in G.nodes() if (G.out_degree(x)==0 and G.in_degree(x)==1) | (G.out_degree(x)==0 and G.in_degree(x)==0)]
    else:
        return [x for x in G.nodes() if G.degree(x) == 1]


def get_source(G):
    if nx.is_directed(G):
        return [x for x in G.nodes() if G.in_degree(x)==0]
    else:
        return None


def get_parents(G, children):
    return np.unique([par for c in children for par in G.predecessors(c)])


def get_exp_degree_per_node(exp_degree_per_al, G):
    al_per_node = nx.get_node_attributes(G, 'al')
    al_per_node_val = [al_per_node[k] for k in al_per_node]
    exp_degree_per_node = pd.Series(exp_degree_per_al.loc[al_per_node_val].values, index=list(al_per_node.keys()))
    return exp_degree_per_node


# turn an undirected forest into a list of directed trees
def undirected_forest_to_directed_trees(G_forest):
    di_forest = []
    for comp in nx.connected_components(G_forest):
        undi_tree = G_forest.subgraph(comp)
        di_tree = nx.bfs_tree(undi_tree, get_rand_node(undi_tree))
        di_tree.add_nodes_from((i, undi_tree.nodes[i]) for i in di_tree.nodes)
        di_forest.append(di_tree)
    return di_forest


def nodes2ue_ind(G, ue_list, nodes_sel):
    # find indexes of "ue_list" corresponding to "nodes_sel" nodes
    rnti_list = nx.get_node_attributes(G, 'rnti')
    rnti_list_sel = [rnti_list[s] for s in nodes_sel]
    ind_ue_sel = []
    for ii, ue in enumerate(ue_list):
        if ue.rnti in rnti_list_sel:
            ind_ue_sel.append(ii)
    return ind_ue_sel
