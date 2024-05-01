import networkx as nx
from mwis.utils import get_weights, is_independent_set, has_nodes
from mwis.node_sort import sort_nodes, best_node
from copy import deepcopy
import time
import pandas as pd
import numpy as np


def greedy_mwis_static_sort(G, sort_node_type='weight/deg', exp_degree_per_al=None,
                            clutter_candidates=True, n_pdcch_candidates_per_al=None, initial_set=None):
    # Select nodes iteratively, according to the chosen criterion.
    # Upon selection, the selected node and its neighbors are removed from the graph
    # nodes are sorted only ONCE, at the beginning
    if initial_set is None:
        initial_set = []
    start_time = time.time()
    nodes_sort = sort_nodes(G, type=sort_node_type, exp_degree_per_al=exp_degree_per_al,
                            clutter_candidates=clutter_candidates,
                            n_pdcch_candidates_per_al=n_pdcch_candidates_per_al)
    nodes_sort = pd.Series(np.arange(len(nodes_sort)), index=nodes_sort)
    independent_set = []  # output independent set
    node_sel_pos = []  # position in the sorted list of the selected nodes

    it = 0
    while len(nodes_sort) > 0:
        # select the first node in line
        if it < len(initial_set):
            node_sel = initial_set[it]
        else:
            node_sel = nodes_sort.index[0]
        independent_set.append(node_sel)
        node_sel_pos.append(nodes_sort.loc[node_sel])

        # eliminate nodes_sel and its neighbors
        nodes_sort.drop(labels=node_sel, inplace=True)
        neighs = np.intersect1d(list(G.neighbors(node_sel)), nodes_sort.index)
        nodes_sort.drop(labels=neighs, inplace=True)
        it += 1

    exec_time_sec = time.time() - start_time

    # compute sum of weights
    weights_G = get_weights(G)
    val = weights_G[independent_set].sum()
    assert is_independent_set(G, independent_set), 'the output set is not independent'
    return independent_set, val, node_sel_pos, exec_time_sec


def greedy_mwis_dynamic_sort(G, sort_node_type='weight/deg', initial_set=None):
    # Select nodes iteratively, according to the chosen criterion.
    # Upon selection, the selected node and its neighbors are removed from the graph
    # Then, node SORTING is UPDATED (at each iteration!)
    start_time = time.time()

    G1 = deepcopy(G)
    independent_set = []
    if initial_set is not None:
        independent_set = deepcopy(initial_set)
        for node_sel in initial_set:
            # remove the node selected and its neighbors
            G1.remove_nodes_from(list(G1.neighbors(node_sel)))
            G1.remove_node(node_sel)

    while has_nodes(G1):
        # select best node according to selected criterion
        node_sel = best_node(G1, type=sort_node_type)
        independent_set.append(node_sel)

        # remove the node selected and its neighbors
        G1.remove_nodes_from(list(G1.neighbors(node_sel)))
        G1.remove_node(node_sel)

    exec_time_sec = time.time() - start_time

    # compute sum of weights
    weights_G = get_weights(G)
    val = weights_G[independent_set].sum()
    assert is_independent_set(G, independent_set)
    return independent_set, val, exec_time_sec



#
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from mwis.optimal import optimal_recursive
#
#     G = nx.Graph()
#
#     G.add_nodes_from([(0, {'weight': 1}),
#                       (1, {'weight': 1}),
#                       (2, {'weight': 1}),
#                       (3, {'weight': 1}),
#                       (4, {'weight': 1}),
#                       (5, {'weight': 1})])
#
#     G.add_edges_from([(0, 1),
#                       (0, 2),
#                       (1, 3),
#                       (1, 4),
#                       (0, 5)])
#
#     indep_set_greedy = greedy_mwis_static_sort(G)
#     print(indep_set_greedy)
#
#     for ii in range(20):
#         indep_set_greedy_rand = greedy_mwis_static_sort(G, sort_node_type='random')
#         print(indep_set_greedy_rand)
#
#     val_opt, indep_set_opt = optimal_recursive(G)
#     print(f'optimal val: {val_opt}')
#     print(f'optimal set: {indep_set_opt}')
#
#     nx.draw(G, with_labels=True)
#     plt.show()
