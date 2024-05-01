
from scheduler.wrr import weighted_round_robin
from mwis.greedy import greedy_mwis_static_sort, greedy_mwis_dynamic_sort
from mwis.optimal import optimal_recursive
from mwis.feige_reichmann import feige_reichmann_mwis
from pdcch.incompatibility_graph import get_incompatibility_graph
from pdcch.utils import get_exp_degree_per_al, check_cce_allocation
import numpy as np
from mwis.utils import nodes2ue_ind
import time



class SchedulerDL:

    def __init__(self, n_re, n_cce, pddch_opt_type='weight/deg static',
                 pdcch_correction_factor_per_al=None, pdcch_weight_type='PF',
                 td_params=None, opt_params=None):
        self.n_re = n_re  # n. available Resource Elements (RE)
        self.n_cce = n_cce  # n. CCEs for PDCCH
        self.pdcch_opt_type = pddch_opt_type
        self.td_params = td_params
        self.opt_params = opt_params
        if not bool(pdcch_correction_factor_per_al):  # is None or empty dictionary
            self.pdcch_correction_factor_per_al = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
        else:
            self.pdcch_correction_factor_per_al = pdcch_correction_factor_per_al
        assert (pdcch_weight_type=='PF') | (pdcch_weight_type=='PF_RE'), f'pdcch_weight_type ' \
                                                                         f'\'{pdcch_weight_type}\' is not supported'
        self.pdcch_weight_type = pdcch_weight_type
        self.re_alloc_per_ue = None
        self.ind_scheduled_ues = None
        self.bit_tx_per_ue = None
        self.ues = None  # list of User objects
        self.n_ues = None


    def _reinit(self):
        self.re_alloc_per_ue = None
        self.ind_scheduled_ues = None
        self.bit_tx_per_ue = None
        self.ues = None
        self.n_ues = None


    def _td_scheduler(self):
        if self.td_params is None:
            # no td scheduler -- all UEs are selected
            self.ind_scheduled_ues = np.arange(self.n_ues)

        elif self.td_params['type'] == 'sortPF_cutoff_re':
            # select UEs with highest PF metric up until n_re_cutoff REs are requested
            assert 'portion_re_cutoff' in self.td_params.keys(), 'if self.td_type==sortPF_cutoff_re, then ' \
                                                         'self.td_params must have key \'portion_re_cutoff\''
            re_requested_vec = np.array([ue.re_requested for ue in self.ues])
            n_re_cutoff = (self.n_re * self.td_params['portion_re_cutoff'])
            if np.sum(re_requested_vec) <= n_re_cutoff:
                self.ind_scheduled_ues = np.arange(self.n_ues)
            else:
                pf_vec = [ue.pf_metric for ue in self.ues]
                ind_ue_sort = np.argsort(pf_vec)[::-1]
                ue_cutoff = np.where(np.cumsum(re_requested_vec[ind_ue_sort]) > n_re_cutoff)[0][0]
                self.ind_scheduled_ues = ind_ue_sort[: ue_cutoff]

        elif self.td_params['type'] == 'sortPF_cutoff_cce':
            # select UEs with highest PF metric up until n_re_cutoff RE's are requested
            assert 'portion_cce_cutoff' in self.td_params.keys(), 'if self.td_type==sortPF_cutoff_cce, then ' \
                                                         'self.td_params must have key \'portion_cce_cutoff\''
            al_per_ue = np.array([ue.al for ue in self.ues])
            n_cce_cutoff = (self.n_cce * self.td_params['portion_cce_cutoff'])
            if np.sum(al_per_ue) <= n_cce_cutoff:
                self.ind_scheduled_ues = np.arange(self.n_ues)
            else:
                pf_vec = [ue.pf_metric for ue in self.ues]
                ind_ue_sort = np.argsort(pf_vec)[::-1]
                ue_cutoff = np.where(np.cumsum(al_per_ue[ind_ue_sort]) > n_cce_cutoff)[0][0]
                self.ind_scheduled_ues = ind_ue_sort[: ue_cutoff]

        elif self.td_params['type'] == 'sortPF_cutoff_ues':
            assert 'n_ues_cutoff' in self.td_params.keys(), 'if self.td_type==sortPF_cutoff_ues, then ' \
                                                         'self.td_params must have key \'n_ues_cutoff\''
            if self.td_params['n_ues_cutoff'] >= self.n_ues:
                self.ind_scheduled_ues = np.arange(self.n_ues)
            else:
                pf_vec = [ue.pf_metric for ue in self.ues]
                ind_ue_sort = np.argsort(pf_vec)[::-1]
                self.ind_scheduled_ues = ind_ue_sort[: int(self.td_params['n_ues_cutoff'])]

        return self.ind_scheduled_ues


    def _pdcch_scheduler(self):
        # build incompatibility graph
        ues_selected = [self.ues[ii] for ii in self.ind_scheduled_ues]

        for ue in ues_selected:
            if self.pdcch_weight_type == 'PF':
                ue.pdcch_weight = ue.pf_metric * self.pdcch_correction_factor_per_al[ue.al]
            elif self.pdcch_weight_type == 'PF_RE':
                n_re_req = min(ue.n_bits_buffer / ue.se, self.n_re)
                ue.pdcch_weight = ue.pf_metric * n_re_req * self.pdcch_correction_factor_per_al[ue.al]

        G = get_incompatibility_graph(ues_selected, mode='DL', do_plot=False)

        if self.pdcch_opt_type == 'random':
            nodes_mwis, val_mwis, node_sel_pos, exec_time_mwis = greedy_mwis_static_sort(G, sort_node_type='random')
        elif self.pdcch_opt_type == 'weight':
            nodes_mwis, val_mwis, node_sel_pos, exec_time_mwis = greedy_mwis_static_sort(G, sort_node_type='weight')
        elif self.pdcch_opt_type == 'weight/E[deg]':
            al_vec = [ue.al for ue in ues_selected]
            exp_degree_per_al = get_exp_degree_per_al(al_vec, ues_selected[0].n_candidates_per_al, self.n_cce)
            nodes_mwis, val_mwis, node_sel_pos, exec_time_mwis = \
                greedy_mwis_static_sort(G, sort_node_type='weight/E[deg]', exp_degree_per_al=exp_degree_per_al,
                                        clutter_candidates=True)
        elif self.pdcch_opt_type == 'weight/AL':
            nodes_mwis, val_mwis, node_sel_pos, exec_time_mwis = greedy_mwis_static_sort(G, sort_node_type='weight/AL')
        elif self.pdcch_opt_type == 'weight/deg static':
            nodes_mwis, val_mwis, node_sel_pos, exec_time_mwis = greedy_mwis_static_sort(G, sort_node_type='weight/deg')
        elif self.pdcch_opt_type == 'weight/deg dynamic':
            nodes_mwis, val_mwis, exec_time_mwis = greedy_mwis_dynamic_sort(G, sort_node_type='weight/deg')
        elif self.pdcch_opt_type == 'GGWMIN':
            nodes_mwis, val_mwis, exec_time_mwis = greedy_mwis_dynamic_sort(G, sort_node_type='GGWMIN')
        elif self.pdcch_opt_type == 'weight/sum_weights':
            nodes_mwis, val_mwis, exec_time_mwis = greedy_mwis_dynamic_sort(G, sort_node_type='weight/sum_weights')
        elif self.pdcch_opt_type == 'suresh':
            nodes_mwis, val_mwis, node_sel_pos, exec_time_mwis = \
                greedy_mwis_static_sort(G, sort_node_type='suresh',
                                        n_pdcch_candidates_per_al=ues_selected[0].n_candidates_per_al)
        elif self.pdcch_opt_type == 'Feige-Reichmann':
            ##################
            nodes_mwis_greedy, _, exec_time_greedy = greedy_mwis_dynamic_sort(G, sort_node_type='weight/deg')
            nodes_mwis, val_mwis, exec_time_fr = feige_reichmann_mwis(G=G, sort_node_type='weight/deg',
                                                                      initial_set=nodes_mwis_greedy,
                                                                      do_plot=False)
            exec_time_mwis = exec_time_greedy + exec_time_fr
        elif self.pdcch_opt_type == 'SoA':
            start_time = time.time()

            # select the M users with highest PDCCH metric
            pdcch_weight_vec = [self.ues[ii].pdcch_weight / self.ues[ii].al for ii in self.ind_scheduled_ues]
            ind_ue_sort = np.argsort(pdcch_weight_vec)[::-1]
            ues_firstM = [self.ues[ii] for ii in self.ind_scheduled_ues[ind_ue_sort[: self.opt_params['M']]]]

            # compute optimal solution on subset of users
            G_subgraph = get_incompatibility_graph(ues_firstM, mode='DL', do_plot=False)
            val_mwis_tmp, nodes_mwis_tmp = optimal_recursive(G_subgraph)

            # use weight/AL greedy on remaining users
            nodes_mwis, val_mwis, _, exec_time_mwis = greedy_mwis_static_sort(G, sort_node_type='weight/AL',
                                                                              initial_set=nodes_mwis_tmp)

            exec_time_mwis = time.time() - start_time
        elif self.pdcch_opt_type == 'optimal':
            start_time = time.time()
            val_mwis, nodes_mwis = optimal_recursive(G)
            exec_time_mwis = time.time() - start_time
        else:
            raise ValueError('\'type\' is not recognized')

        # check that there are no conflicts
        check_cce_allocation(G, nodes_mwis, self.n_cce)

        ind_sel = nodes2ue_ind(G, ues_selected, nodes_mwis)
        self.ind_scheduled_ues = [self.ind_scheduled_ues[ii] for ii in ind_sel]
        return self.ind_scheduled_ues, val_mwis, exec_time_mwis


    def _fd_scheduler(self):
        # FD scheduler
        # n. RE per UE
        re_requested_vec_td = [self.ues[ii].re_requested for ii in self.ind_scheduled_ues]
        weight_qos_vec = [self.ues[ii].weight_qos for ii in self.ind_scheduled_ues]
        re_alloc_ue_sel = weighted_round_robin(re_requested_vec_td, self.n_re, weights=weight_qos_vec)
        self.re_alloc_per_ue = np.zeros(self.n_ues)
        self.re_alloc_per_ue[self.ind_scheduled_ues] = re_alloc_ue_sel

        # assert sum(self.re_alloc_per_ue) <= self.n_re, f'# allocated RE\'s exceeds the total ' \
        #                                                f'# available RE\'s by {sum(self.re_alloc_per_ue)-self.n_re}'
        # n. tx bits per UE
        se_per_ue = np.array([ue.se for ue in self.ues], dtype=float)
        self.bit_tx_per_ue = self.re_alloc_per_ue * se_per_ue
        return self.bit_tx_per_ue, self.re_alloc_per_ue


    def schedule(self, ues):
        # ues: list of User objects, being UEs to be scheduled
        self._reinit()

        self.ues = ues
        self.n_ues = len(self.ues)

        # TD scheduler
        self._td_scheduler()

        # PDCCH scheduler
        self._pdcch_scheduler()

        # FD scheduler
        self._fd_scheduler()

        return self.bit_tx_per_ue, self.re_alloc_per_ue
