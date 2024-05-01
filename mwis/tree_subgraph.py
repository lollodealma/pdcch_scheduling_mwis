import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mwis.node_sort import sort_nodes


def get_tree_subgraph(G, method='components', sort_node_type='weight/deg', initial_set=None,
                      exp_degree_per_al=None, clutter_candidates=True, do_plot=False):
    # Loops over nodes and add a node only if at most one of its neighbors has already been seen.
    # Used for Feige-Reichmann MWIS algorithm
    # method :
    # - 'components': maintains set of connected components. Accepts new node only if does not create a loop,
    #                 i.e., if the node has at most on neighbor in each component
    # - '1neighbor': accepts a new node only if at most one of its neighbors has already been seen

    G_forest = nx.Graph()
    if initial_set is not None:
        for node in initial_set:
            G_forest.add_node(node, attr_dict=None, **G.nodes()[node])

    nodes_sort = sort_nodes(G, type=sort_node_type, exp_degree_per_al=exp_degree_per_al,
                            clutter_candidates=clutter_candidates)

    nodes_sort = np.setdiff1d(nodes_sort, list(G_forest.nodes()))

    if method == 'components':

        if initial_set is not None:
            assert nx.is_empty(G.subgraph(initial_set)), 'initial_set must be an independent set'
            # connected components
            node2component = pd.Series(np.arange(len(initial_set)), index=initial_set, dtype=int)
        else:
            node2component = pd.Series(dtype=int)

        for node in nodes_sort:
            neigh_already_seen = np.intersect1d(list(G.neighbors(node)), list(G_forest.nodes()))
            # check if node is isolated
            if neigh_already_seen.size == 0:
                # if so, then accept node...
                G_forest.add_node(node, attr_dict=None, **G.nodes()[node])
                # ...and create new component
                node2component.loc[node] = node2component.max() + 1
            else:
                components_neigh = node2component.loc[neigh_already_seen].values
                # check if neighbors of new node all belong to different components
                if np.unique(components_neigh).size == components_neigh.size:
                    # if so, then no loop is introduced --> accept the node...
                    G_forest.add_node(node, attr_dict=None, **G.nodes()[node])
                    # ... and merge the components into one new component
                    merged_component = min(components_neigh)
                    for cc in components_neigh:
                        node2component.loc[node2component == cc] = merged_component
                    node2component.loc[node] = merged_component
                    for neigh in neigh_already_seen:
                        G_forest.add_edge(node, neigh)

    elif method == '1neighbor':

        for node in nodes_sort:
            neigh_already_seen = np.intersect1d(list(G.neighbors(node)), list(G_forest.nodes()))
            # add new node if at mnost one of its neighbors has already been seen
            if len(neigh_already_seen) < 2:
                G_forest.add_node(node, attr_dict=None, **G.nodes()[node])
            # add corresponding edge (if any)
            if len(neigh_already_seen) == 1:
                G_forest.add_edge(node, neigh_already_seen[0])

    else:

        raise ValueError('\'method\' must be either \'components\' or \'1neighbor\'')

    if do_plot:
        nx.draw(G_forest)
        plt.show()
    return G_forest
