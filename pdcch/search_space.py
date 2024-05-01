
import numpy as np

def Yp(coreset_id, n_slot, n_rnti):
    # used in the hashing function for Search Space
    pmod3 = coreset_id % 3
    if pmod3 == 0:
        Ap = 39827
    elif pmod3 == 1:
        Ap = 39829
    else:
        Ap = 39839
    D = 65537

    if n_slot == -1:
        output = n_rnti
    else:
        output = np.int64(Ap) * Yp(coreset_id, n_slot - 1, n_rnti) % D
    return np.int64(output)



def search_space_start_cce(n_cce, coreset_id, slot, rnti, al, n_pdcch_candidates, coreset_cce_start_idx=0, n_ci=0):
    """
    Computes the starting CCE of the set of possible candidates for a given RNTI in a given slot
    :param n_cce: n. CCE in the CORESET
    :param coreset_id: Coreset ID (0,1,2,...)
    :param slot: n. slot (0,1,2,...)
    :param rnti:
    :param al: Aggregation Level of the RNTI
    :param n_pdcch_candidates: n. candidates for the specific AL
    :param coreset_cce_start_idx: index of the first CCE of the Coreset
    :param n_ci:
    :return: starting CCE of the set of possible candidates for a given RNTI in a given slot
    """
    # hashing function to determine start of CCEs
    start_cce_vec = []
    for ii in range(n_pdcch_candidates):
        rand_alloc = Yp(coreset_id, slot, rnti) + (ii * n_cce) // (al * n_pdcch_candidates) + n_ci
        start_cce = al * (rand_alloc % (n_cce // al)) + coreset_cce_start_idx
        start_cce_vec.append(start_cce)
    return start_cce_vec

