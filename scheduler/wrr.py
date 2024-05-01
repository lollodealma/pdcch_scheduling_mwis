
import numpy as np
from copy import deepcopy


def round_robin(requests, budget):
    # allocate requests in round robin fashion until budget is met
    requests = np.array(requests, dtype=float)
    requests[requests > budget] = budget
    n_ues = len(requests)
    alloc = np.zeros(n_ues)
    budget_left = budget
    # ROUND ROBIN
    ind_sort = list(np.argsort(requests))
    for n in range(n_ues):
        alloc[ind_sort[n]] = min(requests[ind_sort[n]],
                                 budget_left / (n_ues - n))
        budget_left -= alloc[ind_sort[n]]
    return alloc


def weighted_round_robin(requests, budget, weights=None, tol=1e-10):
    # allocate requests in weighted round robin fashion until budget is met
    requests = np.array(requests, dtype=float)
    requests[requests > budget] = budget
    n_ues = len(requests)
    alloc = np.zeros(n_ues)
    budget_left = budget
    if weights is None:
        # ROUND ROBIN
        weights = np.ones(n_ues)
    else:
        weights = np.array(weights, dtype=float)
    requests_left = deepcopy(requests)
    ues_left = np.ones(n_ues, dtype=bool)
    while (sum(ues_left) > 0) & (budget_left > tol):
        # time until the first UE has nothing more to transmit, if budget is unlimited
        time_to_request_depletion = requests_left[ues_left] / weights[ues_left]
        # time until budget depletes, if requests are unlimited
        time_to_budget_depletion = budget_left / sum(weights[ues_left])
        # time until a UE depletes requests or budget depletes
        time_first_event = np.min(np.r_[time_to_request_depletion, time_to_budget_depletion])
        requests_served = weights[ues_left] * time_first_event
        # update allocation, remaining requests, budget and remaining UEs
        alloc[ues_left] = alloc[ues_left] + requests_served
        requests_left[ues_left] = requests_left[ues_left] - requests_served
        budget_left -= sum(requests_served)
        ues_left = (requests_left > tol)
            
    return alloc


if __name__ == '__main__':
    requests = [3, 5, 1, 10]
    budget = 10
    weights = np.array([.1, 1, 1, 1])

    alloc_wrr = weighted_round_robin(requests, budget, weights=weights)
    print(alloc_wrr)
    print(sum(alloc_wrr))

    alloc_rr = weighted_round_robin(requests, budget)
    print(alloc_rr)
    print(sum(alloc_rr))

    alloc_rr1 = round_robin(requests, budget)
    print(alloc_rr1)
    print(sum(alloc_rr1))
