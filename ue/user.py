from pdcch.search_space import search_space_start_cce
import numpy as np
import warnings


class User:

    def __init__(self, coreset, n_candidates_per_al, rnti, se_gen, al_gen, traffic_gen,
                 slot_init=0, record_hist=True, discount_se=.8, pdcch_weight_type='PF', weight_qos=1,
                 buffer_drop_factor=0, buffer_max_length=1e10):
        self.coreset = coreset
        assert 'id' in coreset.keys()
        assert 'n_cce' in coreset.keys()
        assert 'cce_start_idx' in coreset.keys()
        assert 'symbol' in coreset.keys()
        assert (coreset['symbol'] == 0) | (coreset['symbol'] == 1) | (coreset['symbol'] == 2)
        self.coreset = coreset
        self.n_candidates_per_al = n_candidates_per_al
        self.traffic_gen = traffic_gen  # if None, then full buffer
        self.se_gen = se_gen
        self.al_gen = al_gen
        self.rnti = rnti
        self.slot = slot_init
        self.pdcch_weight_type = pdcch_weight_type
        self.record_hist = record_hist
        self.discount_se = discount_se
        self.weight_qos = weight_qos
        self.buffer_max_length = min(buffer_max_length, 1e10)
        self.buffer_drop_factor = min(max(buffer_drop_factor, 0), 1)
        self.n_bits_buffer = 0
        self.thpt_past_avg = 1
        # self.se_past_avg = 1
        self.n_bits_buffer_hist, self.se_hist, self.al_hist, self.is_scheduled_hist, \
        self.n_tx_bits_hist, self.pf_metric_hist, self.pddch_weight_hist, \
        self.n_re_alloc_hist = [], [], [], [], [], [], [], []
        self.pf_metric, self.se, self.al, self.n_candidates, self.search_space, self.pdcch_weight, self.re_requested = \
            None, None, None, None, None, None, None
        self.flag = True

    def new_slot(self):
        assert self.flag, 'after \'new_slot\' one must call \'update_buffer\''

        # drop bits from buffer
        self.n_bits_buffer = (1 - self.buffer_drop_factor) * self.n_bits_buffer

        # update buffer with new bits to transmit
        self.n_bits_buffer += self.traffic_gen.incoming_n_bits()
        self.n_bits_buffer = min(self.n_bits_buffer, self.buffer_max_length)

        # new spectral efficiency (SE) and aggregation level (AL)
        self.se = self.se_gen.evolve()
        self.al = self.al_gen.evolve()

        # requested resources
        self.re_requested = self.n_bits_buffer / self.se

        # n. PDCCH candidates
        self.n_candidates = self.n_candidates_per_al.loc[self.al]

        # update proportional fairness (PF) metric
        self.pf_metric = self.se / (1e-6 + (1 - self.discount_se) * self.thpt_past_avg)

        # Search Space
        self.search_space = search_space_start_cce(self.coreset['n_cce'], self.coreset['id'], self.slot,
                                                   self.rnti, self.al, self.n_candidates,
                                                   coreset_cce_start_idx=self.coreset['cce_start_idx'], n_ci=0)

        if self.record_hist:
            self.update_hist(al=self.al, se=self.se, n_bits_buffer=self.n_bits_buffer, pf_metric=self.pf_metric)
        self.flag = False
        self.slot += 1

        return self.n_bits_buffer, self.se, self.al, self.n_candidates, self.pf_metric, \
               self.pdcch_weight, self.search_space

    def transmit(self, n_tx_bits, n_re_alloc=None):
        # after the scheduler has granted transmission, it updates buffer and avg achieved SE
        assert not self.flag, 'after \'transmit\' one must call \'new_slot\''
        if n_tx_bits > self.n_bits_buffer:
            raise ValueError('# tx bits > # bits in the buffer')
        self.n_bits_buffer -= n_tx_bits

        # discounted average throughput (used for PF)
        self.thpt_past_avg = (n_tx_bits + self.discount_se * self.thpt_past_avg)

        if self.record_hist:
            self.update_hist(n_tx_bits=n_tx_bits, n_re_alloc=n_re_alloc)
        self.flag = True

    def update_hist(self, al=None, se=None, n_bits_buffer=None, n_tx_bits=None,
                    pf_metric=None, n_re_alloc=None):
        if al is not None:
            self.al_hist.append(al)
        if se is not None:
            self.se_hist.append(se)
        if n_bits_buffer is not None:
            self.n_bits_buffer_hist.append(n_bits_buffer)
        if n_tx_bits is not None:
            self.n_tx_bits_hist.append(n_tx_bits)
            self.is_scheduled_hist.append(n_tx_bits > 0)
        if pf_metric is not None:
            self.pf_metric_hist.append(pf_metric)
        if n_re_alloc is not None:
            self.n_re_alloc_hist.append(n_re_alloc)
        return None
