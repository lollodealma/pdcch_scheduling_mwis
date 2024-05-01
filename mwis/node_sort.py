import pandas as pd
import networkx as nx
import numpy as np
from mwis.utils import get_degree, get_exp_degree_per_node


def best_node(G, type='weight/deg'):
    # find best node according to a certain metric
    if type == 'weight':
        coef = pd.Series(nx.get_node_attributes(G, 'weight'))
        node_best = coef.idxmax()

    elif type == 'weight/AL':
        coef = pd.Series(nx.get_node_attributes(G, 'weight')) / pd.Series(nx.get_node_attributes(G, 'al'))
        node_best = coef.idxmax()

    elif (type == 'weight/deg') | (type == 'GWMIN'):
        coef = pd.Series(nx.get_node_attributes(G, 'weight')) / (get_degree(G) + 1)
        node_best = coef.idxmax()

    elif type == 'GWMAX':
        coef = pd.Series(nx.get_node_attributes(G, 'weight')) / (get_degree(G) * (get_degree(G) + 1))
        node_best = coef.idxmin()

    elif type == 'GGWMIN':
        weights = pd.Series(nx.get_node_attributes(G, 'weight'))
        coef = pd.Series(nx.get_node_attributes(G, 'weight')) / (get_degree(G) + 1)
        nodes_sort = np.array(coef.sort_values(ascending=False).index)

        node = None
        for node in nodes_sort:
            neighs_plus = list(G.neighbors(node))
            neighs_plus.append(node)
            if coef.loc[neighs_plus].sum() <= weights.loc[node]:
                break
        node_best = node

    elif type == 'weight/sum_weights':
        weights = pd.Series(nx.get_node_attributes(G, 'weight'))
        coef = pd.Series(0., index=list(G.nodes()))
        for node in G.nodes():
            weights_neigh = weights.loc[list(G.neighbors(node))].sum()
            coef.loc[node] = weights.loc[node] / (weights.loc[node] + weights_neigh)
        node_best = coef.idxmax()

    else:
        raise ValueError('input \'type\' not recognized')

    return node_best


def sort_nodes(G, type='weight/deg', exp_degree_per_al=None, clutter_candidates=True, n_pdcch_candidates_per_al=None):
    nodes_sort = None

    # sort nodes of G according to a certain criterion
    if type == 'random':

        nodes_sort = np.random.permutation(list(G.nodes()))

    elif (type == 'weight/deg') | (type == 'GWMIN'):

        coef = pd.Series(nx.get_node_attributes(G, 'weight')) / (get_degree(G) + 1)
        nodes_sort = np.array(coef.sort_values(ascending=False).index)

    elif type == 'weight':

        coef = pd.Series(nx.get_node_attributes(G, 'weight'))
        nodes_sort = np.array(coef.sort_values(ascending=False).index)

    elif type == 'weight/E[deg]':

        assert (exp_degree_per_al is not None), \
            'if sort_node_type==\'weight/E[deg]\' then \'exp_degree_per_al\' must be specified'
        exp_degree_per_node = get_exp_degree_per_node(exp_degree_per_al, G)
        start_cce_per_node = nx.get_node_attributes(G, 'start_cce')
        coef_ = (pd.Series(nx.get_node_attributes(G, 'weight')) / exp_degree_per_node).rename('coef')
        id_rand = pd.Series(np.random.rand(len(coef_)), index=coef_.index).rename('id_rand')
        df = pd.concat([coef_, pd.Series(start_cce_per_node).rename('start_cce'), id_rand], axis=1)
        # sort nodes by weight/E[deg] (-> all nodes for the same RNTI have same coef) first,
        if clutter_candidates:
            # as a second criterion, sort nodes by start_CCE (-> candidates are cluttered)
            df.sort_values(by=['coef', 'start_cce'], ascending=False, inplace=True)
        else:
            # else, sort nodes randomly for the same RNTI
            df.sort_values(by=['coef', 'id_rand'], ascending=False, inplace=True)
        nodes_sort = df.index

    elif type == 'weight/AL':

        al_per_node = pd.Series(nx.get_node_attributes(G, 'al'))
        start_cce_per_node = nx.get_node_attributes(G, 'start_cce')

        coef_ = (pd.Series(nx.get_node_attributes(G, 'weight')) / al_per_node).rename('coef')
        id_rand = pd.Series(np.random.rand(len(coef_)), index=coef_.index).rename('id_rand')
        df = pd.concat([coef_, pd.Series(start_cce_per_node).rename('start_cce'), id_rand], axis=1)
        # sort nodes by weight/AL (-> all nodes for the same RNTI have same coef) first,
        if clutter_candidates:
            # as a second criterion, sort nodes by start_CCE (-> candidates are cluttered)
            df.sort_values(by=['coef', 'start_cce'], ascending=False, inplace=True)
        else:
            # else, sort nodes randomly for the same RNTI
            df.sort_values(by=['coef', 'id_rand'], ascending=False, inplace=True)
        nodes_sort = df.index

    elif type == 'suresh':

        if n_pdcch_candidates_per_al is None:
            raise ValueError('if type==\'suresh\' then \'n_pdcch_candidates_per_al\' must be specified as input')
        al_per_node = pd.Series(nx.get_node_attributes(G, 'al'))
        weights = pd.Series(nx.get_node_attributes(G, 'weight'))
        weights_mod = pd.Series(index=list(G.nodes()), dtype=float)
        for node in list(G.nodes()):
            n_pddch_candidates_node = n_pdcch_candidates_per_al.loc[al_per_node.loc[node]]

            neighs = list(G.neighbors(node))
            al_neighs = al_per_node.loc[neighs]
            n_pddch_candidates_neighs = n_pdcch_candidates_per_al.loc[al_neighs.values]

            # weight_mod_denominator: sum_{neighbors n} 1/n_pddch_candidates[n],
            #                         where the neighbors do NOT include candidates belonging to the same RNTI as ' node'
            weight_mod_denominator = np.sum(1 / n_pddch_candidates_neighs.values) - \
                                     (n_pddch_candidates_node - 1) / n_pddch_candidates_node
            weights_mod.loc[node] = weights.loc[node] / weight_mod_denominator

            nodes_sort = np.array(weights_mod.sort_values(ascending=False).index)

    else:
        raise ValueError('sort_node_type must be \'random\', \'weight/deg\' or \'weight\'')
    return nodes_sort

