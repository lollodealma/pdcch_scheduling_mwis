import pandas as pd
from mwis.utils import get_hist
from mwis.utils import get_degree
import networkx as nx
import numpy as np


def maskCce(startCce, n_al):
    # Allocates n_al resources starting from the startCce+1-th
    # In binary, maskCce(3, 2) = 11000
    out = (((1 << n_al) - 1) << startCce)  # (2**n_al-1) * 2**startCce
    return out


def al_to_al_idx(al):
    return int(np.log2(al))


def get_exp_degree_per_al(al_ue_vec, n_pdcch_candidates_per_al, n_cce):
    """
    Computes expected degree of a node with a certain AL in the incompatibility graph

    :param al_ue_vec:
    :param n_pdcch_candidates_per_al:
    :param n_cce:
    :return:
    """
    al_hist = get_hist(al_ue_vec)
    exp_degree_per_al = pd.Series(index=al_hist.index, dtype=float)
    for al in al_hist.index:
        s = 0
        for al1 in al_hist.index:
            n = (al_hist[al1] - 1) if al1 == al else al_hist[al1]
            s += n * n_pdcch_candidates_per_al.loc[al1] * max(al, al1) / n_cce
        exp_degree_per_al[al] = s + n_pdcch_candidates_per_al.loc[al] - 1
    return exp_degree_per_al


def get_avg_degree_per_al(G):
    al_node = pd.Series(nx.get_node_attributes(G, 'al'), name='al')
    deg_node = get_degree(G).rename('deg')
    df = pd.concat([al_node, deg_node], axis=1)
    return df.groupby('al')['deg'].mean(), df


def check_cce_allocation(G, nodes, n_cce):
    cce_alloc = np.zeros(n_cce)
    for node in nodes:
        start_cce = G.nodes[node]['start_cce']
        al = G.nodes[node]['al']
        cce_alloc[start_cce:start_cce+al] = cce_alloc[start_cce:start_cce+al] + 1
    assert (np.all(cce_alloc<=1)), 'There are conflicts among PDCCH candidates'
    return cce_alloc

