
import numpy as np
import os

class MovingAverage:

    def __init__(self, mean, lims, coef=None, sigma=0, record_hist=False):
        if coef is None:
            coef = [0]
        self.lims = lims  # [min_spectral_efficiency, max_spectral_efficiency]
        self.mean = mean
        self.coef = coef  # coefficient AR
        self.sigma = sigma  # noise std AR
        self.eps_vec = self.sigma * np.random.randn(len(self.coef))
        self.hist = []
        self.val = None
        self.record_hist = record_hist


    def evolve(self):
        # moving-average process for spectral efficiency (SE)
        self.val = self.mean + np.dot(self.coef, self.eps_vec)
        # stay within limits
        self.val = min(max(self.val, self.lims[0]), self.lims[1])
        # update eps_vec
        self.eps_vec = np.r_[self.eps_vec[1:], self.sigma * np.random.randn()]
        # record values
        if self.record_hist:
            self.hist.append(self.val)
        return self.val



class MetropolisHastings:
    # generate Markov Chain given its stationary distribution
    # https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
    def __init__(self, pi_stationary, states=None, transition=None):
        # pi: stationary distribution
        self.pi_stationary = pi_stationary
        self.n_states = len(pi_stationary)
        if states is None:
            self.states = np.arange(self.n_states)
        else:
            self.states = states
        self.state_idx = np.random.choice(np.arange(self.n_states), p=self.pi_stationary)
        self.state = self.states[self.state_idx]
        self.state_idx_hist = [self.state_idx]
        if transition is None:
            # tri-diagonal transition matrix
            eps = .3
            self.transition = np.diag(np.ones(self.n_states), 0) + np.diag(eps * np.ones(self.n_states-1), 1) + \
                              np.diag(eps * np.ones(self.n_states-1), -1)
            self.transition[0, 1] = 2 * eps
            self.transition[-1, -2] = 2 * eps
            self.transition = self.transition / self.transition.sum(axis=1).reshape(-1, 1)
        else:
            self.transition = transition

    def evolve(self):
        next_state_idx_tent = np.random.choice(np.arange(self.n_states), p=self.transition[self.state_idx, :])
        num = self.pi_stationary[next_state_idx_tent] * self.transition[next_state_idx_tent, self.state_idx]
        den = self.pi_stationary[self.state_idx] * self.transition[self.state_idx, next_state_idx_tent]
        acceptance_rate = num / den
        if np.random.rand() <= acceptance_rate:
            self.state_idx = next_state_idx_tent
            self.state = self.states[self.state_idx]
        self.state_idx_hist.append(self.state_idx)
        return self.state


def tridiagonal(size, off_diagonal_prob):
    transition_mat = np.diag(np.ones(size), 0) + np.diag(off_diagonal_prob * np.ones(size-1), 1) + \
                      np.diag(off_diagonal_prob * np.ones(size-1), -1)
    transition_mat[0, 1] = 2 * off_diagonal_prob
    transition_mat[-1, -2] = 2 * off_diagonal_prob
    transition_mat = transition_mat / transition_mat.sum(axis=1).reshape(-1, 1)
    return transition_mat


def geomean(vec):
    vec = np.array(vec)
    vec = vec[~np.isnan(vec)]
    return np.power(np.prod(vec), 1 / len(vec))


def cdf_fun(vec, clean=True):
    vec_isnotnan = ~np.isnan(vec)
    n_isnotnan = sum(vec_isnotnan)
    x = np.sort(np.array(vec)[vec_isnotnan])
    y = np.arange(1, n_isnotnan+1) / n_isnotnan
    if clean:
        ind = []
        x_unique = np.sort(np.unique(x))
        for x_ in x_unique:
            ind_tmp = np.where(x==x_)[0]
            if len(ind_tmp)==1:
                ind.append(ind_tmp[0])
            else:
                ind.extend([ind_tmp[0], ind_tmp[-1]])
        x, y = x[ind], y[ind]
    return x, y


def write_on_file(filename, pdcch_opt_type, seed, n_ues, n_cce, n_candidates_per_al, n_slots, is_full_buffer,
                  avg_n_bits_per_slot, n_re, td_scheduler, pdcch_correction_factor_per_al, off_diagonal_prob_al,
                  n_ue_per_al, n_alloc_re_per_al, avg_thpt_per_al, geomean_thpt, n_sched_ues_per_slot_avg,
                  interval_tx_per_al, load):

    if not bool(pdcch_correction_factor_per_al):  # is None or empty dictionary
        pdcch_correction_factor_per_al = {1: 1, 2: 1, 4: 1, 8: 1}
    if not (os.path.isfile(filename)):
        # create file and write colum names
        with open(filename, 'a+') as fid:
            str_ = ("pdcch_opt_type,"
                    "seed,"
                    "n_ues,"
                    "n_cce,"
                    "n_slots,"
                    "n_candidates_AL1,n_candidates_AL2,n_candidates_AL4,n_candidates_AL8,"
                    "is_full_buffer,"
                    "avg_n_bits_per_slot,"
                    "n_re,"
                    "off_diagonal_prob_al,"
                    "td_scheduler_type,td_scheduler_param,"
                    "n_ue_AL1,n_ue_AL2,n_ue_AL4,n_ue_AL8,"
                    "pdcch_correction_factor_AL1,pdcch_correction_factor_AL2,pdcch_correction_factor_AL4,"
                    "pdcch_correction_factor_AL8,"
                    "n_sched_ues_per_slot,"
                    "n_allocated_RE_AL1,n_allocated_RE_AL2,n_allocated_RE_AL4,n_allocated_RE_AL8,"
                    "load[%],"
                    "geomean_thpt,"
                    "thpt_AL1,thpt_AL2,thpt_AL4,thpt_AL8,"
                    "inter-tx_interval_AL1,inter-tx_interval_AL2,inter-tx_interval_AL4,inter-tx_interval_AL8")
            fid.write('%s\n' % str_)

    # write results
    with open(filename, 'a+') as fid:
        fid.write('%s,' % pdcch_opt_type)
        fid.write('%s,' % seed)
        fid.write('%s,' % n_ues)
        fid.write('%s,' % n_cce)
        fid.write('%s,' % n_slots)
        fid.write('%s,' % n_candidates_per_al.loc[1])
        fid.write('%s,' % n_candidates_per_al.loc[2])
        fid.write('%s,' % n_candidates_per_al.loc[4])
        fid.write('%s,' % n_candidates_per_al.loc[8])
        fid.write('%s,' % is_full_buffer)
        fid.write('%s,' % avg_n_bits_per_slot)
        fid.write('%s,' % n_re)
        fid.write('%s,' % off_diagonal_prob_al)
        if td_scheduler is None:
            fid.write('%s,' % 'None')
            fid.write('%s,' % 'None')
        else:
            fid.write('%s,' % td_scheduler['type'])
            if td_scheduler['type'] == 'sortPF_cutoff_cce':
                fid.write('%s,' % td_scheduler['portion_cce_cutoff'])
            elif td_scheduler['type'] == 'sortPF_cutoff_ues':
                fid.write('%s,' % td_scheduler['n_ues_cutoff'])
            elif td_scheduler['type'] == 'sortPF_cutoff_re':
                fid.write('%s,' % td_scheduler['portion_re_cutoff'])
        fid.write('%s,' % n_ue_per_al[1])
        fid.write('%s,' % n_ue_per_al[2])
        fid.write('%s,' % n_ue_per_al[4])
        fid.write('%s,' % n_ue_per_al[8])
        fid.write('%s,' % pdcch_correction_factor_per_al[1])
        fid.write('%s,' % pdcch_correction_factor_per_al[2])
        fid.write('%s,' % pdcch_correction_factor_per_al[4])
        fid.write('%s,' % pdcch_correction_factor_per_al[8])
        fid.write('%s,' % n_sched_ues_per_slot_avg)
        fid.write('%s,' % n_alloc_re_per_al[0])
        fid.write('%s,' % n_alloc_re_per_al[1])
        fid.write('%s,' % n_alloc_re_per_al[2])
        fid.write('%s,' % n_alloc_re_per_al[3])
        fid.write('%s,' % load)
        fid.write('%s,' % geomean_thpt)
        fid.write('%s,' % avg_thpt_per_al[0])
        fid.write('%s,' % avg_thpt_per_al[1])
        fid.write('%s,' % avg_thpt_per_al[2])
        fid.write('%s,' % avg_thpt_per_al[3])
        fid.write('%s,' % np.mean(interval_tx_per_al[1]))
        fid.write('%s,' % np.mean(interval_tx_per_al[2]))
        fid.write('%s,' % np.mean(interval_tx_per_al[4]))
        fid.write('%s\n' % np.mean(interval_tx_per_al[8]))

    return None


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    pi = [.3, .1, .4, .2]
    mh = MetropolisHastings(pi_stationary=pi)
    for ii in range(10000):
        mh.evolve()
    pi_empirical = [np.sum(np.array(mh.state_idx_hist)==x)/len(mh.state_idx_hist) for x in range(len(pi))]
    print(f'target stationary distribution: {pi}')
    print(f'empirical stationary distribution: {pi_empirical}')
    plt.plot(mh.state_idx_hist)
    plt.title('state index evolution')
    plt.xlabel('step')
    plt.ylabel('state index')
    plt.grid()
    plt.show()
