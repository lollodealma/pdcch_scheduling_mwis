import numpy as np
import networkx as nx
from mwis.utils import get_node_weight, get_leaf_nodes, get_parents, get_source, get_weights, \
    is_independent_set, undirected_forest_to_directed_trees
from mwis.tree_subgraph import get_tree_subgraph
import time


def feige_reichmann_mwis(G=None, sort_node_type='weight/deg', method_subtree='components',
                         initial_set=None, exp_degree_per_al=None, clutter_candidates=True, do_plot=False):
    start_time = time.time()

    # extract an undirected forest-like subgraph from G
    G_forest = get_tree_subgraph(G, method=method_subtree, sort_node_type=sort_node_type,
                                 initial_set=initial_set, exp_degree_per_al=exp_degree_per_al,
                                 clutter_candidates=clutter_candidates, do_plot=do_plot)
    assert nx.is_forest(G_forest), 'G_forest graph must be a forest (i.e., it must not contain cycles)'

    # turn G_forest into a list of directed trees
    di_forest = undirected_forest_to_directed_trees(G_forest)

    mwis = []
    val = 0
    for di_tree in di_forest:
        # start from leaf nodes
        leaves = get_leaf_nodes(di_tree)

        mwis_root_IN, mwis_root_OUT = {}, {}
        for leaf in leaves:
            # exclude leaf in MWIS
            mwis_root_OUT[leaf] = {'val': 0,
                                   'nodes': []}
            # include leaf in MWIS
            mwis_root_IN[leaf] = {'val': get_node_weight(di_tree, leaf),
                                  'nodes': [leaf]}
        source = get_source(di_tree)[0]
        parents = get_parents(di_tree, leaves)
        while len(parents) > 0:
            parents_evaluated = []
            for parent in parents:
                # if all children have already been evaluated
                if np.all([c in mwis_root_IN.keys() for c in di_tree.successors(parent)]):
                    parents_evaluated.append(parent)
                    # Include parent node. Children nodes must be excluded
                    mwis_root_IN[parent] = {'val': get_node_weight(di_tree, parent),
                                            'nodes': [parent]}
                    for child in di_tree.successors(parent):
                        mwis_root_IN[parent]['nodes'].extend(mwis_root_OUT[child]['nodes'])
                        mwis_root_IN[parent]['val'] += mwis_root_OUT[child]['val']

                    # Exclude parent node. For each child, decide whether to include or exclude it
                    mwis_root_OUT[parent] = {'val': 0,
                                             'nodes': []}
                    for child in di_tree.successors(parent):
                        if mwis_root_OUT[child]['val'] > mwis_root_IN[child]['val']:
                            nodes_best = mwis_root_OUT[child]['nodes']
                            val_best = mwis_root_OUT[child]['val']
                        else:
                            nodes_best = mwis_root_IN[child]['nodes']
                            val_best = mwis_root_IN[child]['val']
                        mwis_root_OUT[parent]['nodes'].extend(nodes_best)
                        mwis_root_OUT[parent]['val'] += val_best
            parents = get_parents(di_tree, parents_evaluated)
        if mwis_root_OUT[source]['val'] > mwis_root_IN[source]['val']:
            mwis.extend(mwis_root_OUT[source]['nodes'])
            val += mwis_root_OUT[source]['val']
        else:
            mwis.extend(mwis_root_IN[source]['nodes'])
            val += mwis_root_IN[source]['val']

    exec_time_sec = time.time() - start_time
    assert abs(get_weights(G)[mwis].sum() - val) < 1e-10, 'incorrect value for MWIS value'
    assert is_independent_set(G, mwis)
    return mwis, val, exec_time_sec
