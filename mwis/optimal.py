
from mwis.utils import get_degree, get_weights, get_node_weight
from copy import deepcopy
import networkx as nx


def optimal_recursive(G):

    if nx.is_empty(G):  # no edges
        return get_weights(G).sum(), list(G.nodes())

    # choose a node
    # node_sel = get_rand_node(G)
    node_sel = get_degree(G).idxmax()
    node_sel_weight = get_node_weight(G, node_sel)

    # compute its neighbors
    neighs = G.neighbors(node_sel)

    # do not include node_sel in independent set
    G1 = deepcopy(G)
    G1.remove_node(node_sel)

    val1, nodes1 = optimal_recursive(G1)

    # include node_sel in independent set
    G2 = deepcopy(G)
    G2.remove_node(node_sel)
    G2.remove_nodes_from(list(neighs))

    val2, nodes2 = optimal_recursive(G2)
    val2 += node_sel_weight
    nodes2.append(node_sel)
    if val1 > val2:
        return val1, nodes1
    else:
        return val2, nodes2

