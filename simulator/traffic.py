
import numpy as np


class TrafficGenerator:
    # generates n. bits to be transmitted according to Poisson law
    def __init__(self, avg_n_bits_per_slot=None, is_full_buffer=True, seed=None, len_out=None):
        assert ((avg_n_bits_per_slot is None) & is_full_buffer) | ((avg_n_bits_per_slot is not None) & (not is_full_buffer)), \
            'cannot interpret the inputs'
        self.avg_n_bits_per_slot = avg_n_bits_per_slot
        self.full_buffer = is_full_buffer
        self.out_vec = None
        if (len_out is not None) & (not is_full_buffer):
            if seed is None:
                seed = np.random.randint(10000)
            self.out_vec = np.random.RandomState(seed=seed).poisson(lam=self.avg_n_bits_per_slot, size=len_out)
            self.ind = 0

    def incoming_n_bits(self):
        if self.full_buffer:
            return np.inf
        elif self.out_vec is not None:
            out = self.out_vec[self.ind]
            self.ind += 1
            return out
        else:
            return np.random.poisson(lam=self.avg_n_bits_per_slot)
