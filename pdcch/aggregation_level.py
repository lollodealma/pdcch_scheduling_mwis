import random
import numpy as np

def generate_al_gauss(al_min, al_max):
    # generate Aggregation Level within [1,2,4,8] from a distribution
    mu = 200
    sigma = 50
    v = random.gauss(mu, sigma)
    al = 16
    if v < (mu - 1.1 * sigma):
        al = 1
    elif v < (mu - 0.3 * sigma):
        al = 2
    elif v < (mu + 0.8 * sigma):
        al = 4
    elif v < (mu + 1.5 * sigma):
        al = 8

    if al < al_min:
        al = al_min
    if al > al_max:
        al = al_max
    return al

def generate_al(al_vec, al_prob, n=1):
    return np.random.choice(al_vec, n, p=al_prob)

