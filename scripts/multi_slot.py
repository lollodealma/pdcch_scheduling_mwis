import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import shutil
from ue.user import User
from simulator.spectral_efficiency import SE_Generator
from simulator.aggregation_level import AL_Generator
from simulator.traffic import TrafficGenerator
from scheduler.dl_scheduler import SchedulerDL
from utils import tridiagonal, cdf_fun, geomean, write_on_file
from copy import deepcopy

do_plots = False

cols = ['k', 'k', 'b', 'g', 'orange']
styl = ['-', '--', '-.', '-', '--']
mark = ['', 'o', 'x', 's', '']
# from cycler import cycler
# monochrome = (cycler('color', ['k', 'r']) * cycler('marker', ['', 'o']) *
#               cycler('linestyle', ['-', '--'])) # , '-.'
# plt.rc('axes', prop_cycle=monochrome)

plot_per_ue = False

config_file = "../config/multi_slot_config_paper.yaml"

with open(config_file, "r") as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

al_vec = params['aggregation_level']['possible_values']
p_al = params['aggregation_level']['distribution_across_ues']
n_cce = params['n_cce']  # 45 at most
n_candidates_per_al = pd.Series(params['n_pdcch_candidates_per_al'])
n_slots = params['n_slots']
n_ues = params['n_ues']
n_re = params['n_re']
td_scheduler = params['td_scheduler']
pdcch_opt_type_vec = params['pddch_opt_type']
pdcch_correction_factor_per_al_vec = params['pdcch_correction_factor_per_al']
opt_params_vec = params['opt_params']
# pdcch_weight_type_vec = params['pdcch_weight_type']
off_diagonal_prob_al = params['off_diagonal_prob_al']
out_file = f'res_{n_ues}_ues.csv'
n_config = len(pdcch_correction_factor_per_al_vec)
n_trials = params['n_trials']
seed_vec = np.random.randint(10000, size=n_trials)
avg_n_bits_per_slot = params['traffic_model']['avg_n_bits_per_slot']
is_full_buffer = params['traffic_model']['is_full_buffer']
fname_results = params['filename_results']

assert (len(pdcch_opt_type_vec) == len(
    pdcch_correction_factor_per_al_vec))  # , 'pdcch_weight_PF_coeff_vec, pdcch_opt_type_vec must have the same length'

for it, seed in enumerate(seed_vec):
    print(f'\niteration {it}')
    np.random.seed(seed)

    ll = 'pure' if pdcch_correction_factor_per_al_vec is None else 'coeff'

    mwis_algo_lab = []
    for nn in range(n_config):
        mwis_algo_lab.append(pdcch_opt_type_vec[nn])
        # if pdcch_opt_type_vec[nn] == 'weight':
        #     mwis_algo_lab.append('baseline')
        # elif pdcch_opt_type_vec[nn] == 'random':
        #     mwis_algo_lab.append('random')
        # else:
        #     mwis_algo_lab.append('our invention')  # pdcch_opt_type_vec[nn])

    ###########
    # CORESET #
    ###########
    coreset = {'id': 1,
               'n_cce': n_cce,
               'cce_start_idx': 0,
               'symbol': 0}

    transition_mat_al = tridiagonal(len(al_vec), off_diagonal_prob_al)

    ################
    # Generate UEs #
    ################

    rnti_per_ue = np.random.choice(range(100000), n_ues, replace=False)

    scheduler_dl = {}
    ues = {}
    pf_fair_thpt = []
    ue_list = []
    al_per_ue = []
    for uu in range(n_ues):
        traffic_gen = TrafficGenerator(is_full_buffer=is_full_buffer, avg_n_bits_per_slot=avg_n_bits_per_slot,
                                       seed=seed, len_out=n_slots + 1)
        al_gen = AL_Generator(p_al, states=al_vec, transition=transition_mat_al)
        se_ue = 8 / al_gen.state
        al_per_ue.append(al_gen.state)
        se_gen = SE_Generator(se_ue, [se_ue, se_ue])
        ue = User(coreset, n_candidates_per_al, rnti_per_ue[uu], se_gen, al_gen, traffic_gen,
                  buffer_drop_factor=1, slot_init=0)
        ue.new_slot()
        ue_list.append(ue)
        pf_fair_thpt.append(se_ue * n_re / n_ues)

    for nn in range(n_config):
        ues[nn] = deepcopy(ue_list)

    n_ue_per_al = {}
    for al in al_vec:
        if off_diagonal_prob_al == 0:
            n_ue_per_al[al] = al_per_ue.count(al)
        else:
            n_ue_per_al[al] = None

    geomean_thpt_max = geomean(pf_fair_thpt)

    #############
    # SIMULATOR #
    #############
    for nn in range(n_config):
        print(f'\nconfiguration: {pdcch_opt_type_vec[nn]}')
        # DL Scheduler
        scheduler_dl[nn] = SchedulerDL(n_re, n_cce, td_params=td_scheduler,
                                       pddch_opt_type=pdcch_opt_type_vec[nn],
                                       pdcch_correction_factor_per_al=pdcch_correction_factor_per_al_vec[nn],
                                       opt_params=opt_params_vec[nn])
        # Simulate across slots
        for slot in range(n_slots):
            if np.mod(slot, 50) == 0:
                print(f'{slot}/{n_slots - 1}')
            # BTS schedules UEs in downlink
            bit_tx_per_ue, re_alloc_per_ue = scheduler_dl[nn].schedule(ues[nn])

            for ue_idx, ue in enumerate(ues[nn]):
                # UE transmits
                ue.transmit(bit_tx_per_ue[ue_idx], n_re_alloc=re_alloc_per_ue[ue_idx])
                # new slot
                ue.new_slot()

    ################
    # PERF METRICS #
    ################
    al_mat, n_tx_bits_mat, pf_metric_mat, n_re_alloc_mat_per_ue, load, n_sched_ues_per_slot, \
        interval_tx, interval_tx_per_al = {}, {}, {}, {}, {}, {}, {}, {}
    for nn in range(n_config):
        al_mat[nn] = np.zeros((n_ues, n_slots))
        n_tx_bits_mat[nn] = np.zeros((n_ues, n_slots))
        pf_metric_mat[nn] = np.zeros((n_ues, n_slots))
        n_re_alloc_mat_per_ue[nn] = np.zeros((n_ues, n_slots))
        interval_tx[nn] = [[] for uu in range(n_ues)]
        interval_tx_per_al[nn] = {}
        for al in al_vec:
            interval_tx_per_al[nn][al] = []

        for ue_idx, ue in enumerate(ues[nn]):
            al_mat[nn][ue_idx, :] = ue.al_hist[:-1]
            n_tx_bits_mat[nn][ue_idx, :] = ue.n_tx_bits_hist
            slots_with_tx = np.r_[0, np.where(np.array(ue.n_tx_bits_hist) > 0)[0]]
            interval_tx_curr = [(slots_with_tx[ii + 1] - slots_with_tx[ii]) for ii in range(len(slots_with_tx) - 1)]
            interval_tx[nn][ue_idx].extend(interval_tx_curr)
            if off_diagonal_prob_al == 0:
                al_curr = int(al_mat[nn][ue_idx, 0])
                interval_tx_per_al[nn][al_curr].extend(interval_tx_curr)
            pf_metric_mat[nn][ue_idx, :] = ue.pf_metric_hist[:-1]
            n_re_alloc_mat_per_ue[nn][ue_idx, :] = ue.n_re_alloc_hist
        load[nn] = np.mean(n_re_alloc_mat_per_ue[nn].sum(axis=0)) / n_re * 100
        n_sched_ues_per_slot[nn] = (n_tx_bits_mat[nn] > 0).sum(axis=0)

    al_avg_per_ue, n_re_alloc_per_ue, thpt_per_ue = {}, {}, {}
    for nn in range(n_config):
        al_avg_per_ue[nn], thpt_per_ue[nn], n_re_alloc_per_ue[nn] = [], [], []
        for ue in ues[nn]:
            al_avg_per_ue[nn].append(np.mean(ue.al_hist))
            thpt_per_ue[nn].append(np.mean(ue.n_tx_bits_hist))
            n_re_alloc_per_ue[nn].append(np.mean(ue.n_re_alloc_hist) / n_re * 100)

    n_re_alloc_per_ue_avg = {}
    thpt_per_ue_avg = {}
    for nn in range(n_config):
        n_re_alloc_per_ue_avg[nn] = np.mean(n_re_alloc_per_ue[nn])
        thpt_per_ue_avg[nn] = np.mean(thpt_per_ue[nn])

    n_alloc_re_per_al_dict, std_re_per_al_dict, avg_thpt_per_al_dict, std_thpt_per_al_dict = {}, {}, {}, {}

    for nn in range(n_config):
        if off_diagonal_prob_al != 0:
            (n_alloc_re_per_al_dict[nn], std_re_per_al_dict[nn], avg_thpt_per_al_dict[nn],
             std_thpt_per_al_dict[nn]) = None, None, None, None
        else:
            n_alloc_re_per_al, std_re_per_al = [], []
            for al in al_vec:
                id1 = (np.array(al_avg_per_ue[nn]) == al)
                n_alloc_re_per_al_tmp = np.mean(np.array(n_re_alloc_per_ue[nn])[id1])
                std_re_per_al_tmp = np.std(np.array(n_re_alloc_per_ue[nn])[id1]) / 2
                n_alloc_re_per_al.append(n_alloc_re_per_al_tmp)
                std_re_per_al.append(std_re_per_al_tmp)
            n_alloc_re_per_al_dict[nn] = n_alloc_re_per_al
            std_re_per_al_dict[nn] = std_re_per_al

            avg_thpt_per_al, std_thpt_per_al = [], []
            for al in al_vec:
                id1 = (np.array(al_avg_per_ue[nn]) == al)
                avg_thpt_per_al_tmp = np.mean(np.array(thpt_per_ue[nn])[id1])
                std_thpt_per_al_tmp = np.std(np.array(thpt_per_ue[nn])[id1]) / 2
                avg_thpt_per_al.append(avg_thpt_per_al_tmp)
                std_thpt_per_al.append(std_thpt_per_al_tmp)
            avg_thpt_per_al_dict[nn] = avg_thpt_per_al
            std_thpt_per_al_dict[nn] = std_thpt_per_al

    geomean_thpt_dict = {}
    print('\nThroughput (geo-mean):')
    for nn in range(n_config):
        geomean_curr = geomean(thpt_per_ue[nn])
        geomean_thpt_dict[nn] = geomean_curr  # np.power(np.prod(thpt_per_ue[nn]), 1/len(thpt_per_ue[nn]))
        print(f'{mwis_algo_lab[nn]}: {geomean_thpt_dict[nn]:.2f}')


    print('\n# scheduled UEs:')
    n_sched_ues_per_slot_dict = {}
    for nn in range(n_config):
        n_sched_ues_per_slot_dict[nn] = np.mean(n_sched_ues_per_slot[nn])
        print(f'{mwis_algo_lab[nn]}: {n_sched_ues_per_slot_dict[nn]:.2f}')

    for nn in range(n_config):
        write_on_file(fname_results, pdcch_opt_type_vec[nn], seed, n_ues, n_cce, n_candidates_per_al, n_slots,
                      is_full_buffer, avg_n_bits_per_slot, n_re, td_scheduler,
                      pdcch_correction_factor_per_al_vec[nn], off_diagonal_prob_al, n_ue_per_al,
                      n_alloc_re_per_al_dict[nn], avg_thpt_per_al_dict[nn], geomean_thpt_dict[nn],
                      n_sched_ues_per_slot_dict[nn], interval_tx_per_al[nn], load[nn])

    print('results updated in ' + fname_results)

    #########
    # PLOTS #
    #########
    if do_plots:

        if td_scheduler['type'] == 'sortPF_cutoff_cce':
            res_folder = f'../results/{n_ues}_UEs_{n_slots}_slots_seed{seed}_{td_scheduler["portion_cce_cutoff"]}' \
                         f'_cce_cutoff_dyn_{off_diagonal_prob_al}/'
        else:
            res_folder = f'../results/{n_ues}_UEs_{n_slots}_slots_seed{seed}_{td_scheduler["n_ues_cutoff"]}' \
                         f'_ue_cutoff_dyn_{off_diagonal_prob_al}/'

        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        shutil.copy(config_file, res_folder)

        if off_diagonal_prob_al == 0:
            fig, axs = plt.subplots(2, 1, figsize=(7, 7))
            for nn in range(n_config):
                axs[0].errorbar(al_vec, n_alloc_re_per_al_dict[nn], yerr=std_re_per_al_dict[nn], color=cols[nn],
                                linestyle=styl[nn],
                                marker=mark[nn], label=mwis_algo_lab[nn])
                corrective_factor = max(n_alloc_re_per_al_dict[nn]) / np.array(n_alloc_re_per_al_dict[nn])
                print(f'{mwis_algo_lab[nn]}: \n corrective factor: {corrective_factor}')
                if bool(pdcch_correction_factor_per_al_vec[nn]):
                    new_coeff = corrective_factor * np.array(list(pdcch_correction_factor_per_al_vec[nn].values()))
                else:
                    new_coeff = corrective_factor
                new_coeff_dict = {}
                for ii, al in enumerate(al_vec):
                    new_coeff_dict[al] = new_coeff[ii]
                print(f'new coeff: {new_coeff_dict}\n')
            axs[0].grid()
            axs[0].legend(fontsize=11)
            axs[0].set_xlabel('Aggregation Level', fontsize=12)
            axs[0].set_ylabel('% allocated RE\'s', fontsize=12)

            for nn in range(n_config):
                axs[1].errorbar(al_vec, avg_thpt_per_al_dict[nn], yerr=std_thpt_per_al_dict[nn], color=cols[nn],
                                linestyle=styl[nn],
                                marker=mark[nn], label=mwis_algo_lab[nn])
            axs[1].grid()
            axs[1].legend(fontsize=11)
            axs[1].set_xlabel('Aggregation Level', fontsize=12)
            axs[1].set_ylabel('Throughput [b/slot]', fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(res_folder, f'THPT_RE_vs_AL_per_UE.png'))

        fig, axs = plt.subplots(2, 1, figsize=(7, 7))
        for nn in range(n_config):
            x, y = cdf_fun(n_re_alloc_per_ue[nn])
            axs[0].plot(x, y, color=cols[nn], linestyle=styl[nn], marker=mark[nn], label=mwis_algo_lab[nn], markevery=5)
            axs[0].plot([n_re_alloc_per_ue_avg[nn]] * 2, [0, 1], color=cols[nn], linestyle=styl[nn])
        axs[0].set_xlabel('% allocated RE\'s per UE', fontsize=12)
        axs[0].set_ylabel('Cumulative Density Function', fontsize=12)
        axs[0].legend(fontsize=11)
        axs[0].grid()

        for nn in range(n_config):
            x, y = cdf_fun(thpt_per_ue[nn])
            axs[1].plot(x, y, color=cols[nn], linestyle=styl[nn], marker=mark[nn], label=mwis_algo_lab[nn], markevery=5)
            axs[1].plot([thpt_per_ue_avg[nn]] * 2, [0, 1], color=cols[nn], linestyle=styl[nn])
        axs[1].set_xlabel('Throughput per UE [b/slot]', fontsize=12)
        axs[1].set_ylabel('Cumulative Density Function', fontsize=12)
        axs[1].legend(fontsize=11)
        axs[1].grid()

        fig.tight_layout()
        fig.savefig(os.path.join(res_folder, f'RE_per_UE_CDF.png'))

        fig, axs = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[3, 1])
        print('\n# scheduled UEs:')
        for nn in range(n_config):
            # print(f'{mwis_algo_lab[nn]}: {n_sched_ues_per_slot_dict[nn]:.2f}')
            x, y = cdf_fun(n_sched_ues_per_slot[nn])
            axs[0].plot(x, y, color=cols[nn], linestyle=styl[nn], marker=mark[nn], label=mwis_algo_lab[nn])
            axs[0].plot([n_sched_ues_per_slot_dict[nn]] * 2, [0, 1], color=cols[nn], linestyle=styl[nn])
        axs[0].set_xlabel('# scheduled UEs per slot', fontsize=12)
        axs[0].set_ylabel('Cumulative Density Function', fontsize=12)
        axs[0].set_xticks(np.arange(np.floor(axs[0].get_xlim()[0]), np.floor(axs[0].get_xlim()[1]) + 1))
        axs[0].grid()
        axs[0].legend(fontsize=11)

        axs[1].bar(mwis_algo_lab, geomean_thpt_dict.values(), color='k')
        axs[1].xaxis.set_tick_params(rotation=30)
        axs[1].grid()
        axs[1].set_ylabel('Geo-mean Throughout')
        fig.tight_layout()
        fig.savefig(os.path.join(res_folder, f'sched_UEs_per_slot_CDF.png'))

        n_diff_al = len(al_vec)

        fig, axs = plt.subplots(n_diff_al, 1, figsize=(7, 9), sharey='all')  # , sharex='all')
        for nn in range(n_config):
            al_flat = al_mat[nn].flatten()
            n_re_alloc_flat = n_re_alloc_mat_per_ue[nn].flatten()
            for al_idx, al in enumerate(al_vec):
                x, y = cdf_fun(n_re_alloc_flat[al_flat == al])
                axs[al_idx].plot(x, y, color=cols[nn], linestyle=styl[nn], label=mwis_algo_lab[nn])

        for al_idx, al in enumerate(al_vec):
            axs[al_idx].set_title(f'AL={al}', fontweight='bold')
            axs[al_idx].grid()
            axs[al_idx].set_ylabel('CDF', fontsize=12)
            axs[al_idx].set_xlabel('% allocated RE\'s', fontsize=12)
        axs[3].legend(fontsize=10)
        fig.suptitle('% allocated RE\'s (per slot)', fontweight='bold', y=.975)
        fig.tight_layout()
        fig.savefig(os.path.join(res_folder, f'RE_per_AL_per_slot.png'))

        fig, axs = plt.subplots(n_diff_al, 1, figsize=(7, 9),
                                sharey='all')  # , sharex='all')
        for nn in range(n_config):
            # al_flat = al_mat[nn].flatten()
            # n_re_alloc_flat = n_re_alloc_mat[nn].flatten()
            for al_idx, al in enumerate(al_vec):
                ues_al = np.where(np.array(al_avg_per_ue[nn]) == al)[0]
                interval_curr = []
                for ue_idx in ues_al:
                    interval_curr.extend(interval_tx[nn][ue_idx])
                x, y = cdf_fun(interval_curr)
                axs[al_idx].plot(x, y, color=cols[nn], linestyle=styl[nn], label=mwis_algo_lab[nn])

        for al_idx, al in enumerate(al_vec):
            axs[al_idx].set_title(f'AL={al}', fontweight='bold')
            axs[al_idx].grid()
            axs[al_idx].set_ylabel('CDF', fontsize=12)
            axs[al_idx].set_xlabel('inter-Tx #slots', fontsize=12)
        axs[3].legend(fontsize=10)

        fig.suptitle('#slots between consecutive transmissions', fontweight='bold', y=.975)
        fig.tight_layout()
        fig.savefig(os.path.join(res_folder, f'interval_tx_per_AL_per_slot.png'))

        fig, axs = plt.subplots(n_diff_al, 1, figsize=(7, 9))  # , sharey='all', sharex='all')
        for nn in range(n_config):
            al_flat = al_mat[nn].flatten()
            pf_metric_flat = pf_metric_mat[nn].flatten()
            for al_idx, al in enumerate(al_vec):
                x, y = cdf_fun(pf_metric_flat[al_flat == al])
                axs[al_idx].plot(x, y, color=cols[nn], linestyle=styl[nn], label=mwis_algo_lab[nn])

        for al_idx, al in enumerate(al_vec):
            axs[al_idx].set_title(f'AL={al}', fontweight='bold')
            axs[al_idx].grid()
            axs[al_idx].set_ylabel('CDF', fontsize=12)
            axs[al_idx].set_xlabel('PF metric', fontsize=12)
        axs[3].legend(fontsize=10)
        fig.suptitle('Proportional Fairness metric (per slot)', fontweight='bold', y=.975)
        fig.tight_layout()
        fig.savefig(os.path.join(res_folder, f'PF_per_AL_per_slot.png'))

        if off_diagonal_prob_al == 0:
            fig, axs = plt.subplots()
            axs.bar(n_ue_per_al.keys(), n_ue_per_al.values())
            axs.set_xlabel('AL')
            axs.set_ylabel('# UEs')
            axs.grid()
            fig.savefig(os.path.join(res_folder, f'AL_hist.png'))

        # fig, axs = plt.subplots()
        # axs.plot(pf_metric_flat, n_re_alloc_flat, 'o', alpha=.2)
        # axs.set_xlabel('Proportional Fairness metric')
        # axs.set_ylabel('% allocated RE\'s')
        # axs.set_title('1 point = 1 slot')
        # axs.grid()
        # fig.savefig(os.path.join(res_folder, f'PF_vs_RE_per_slot.png'))

        if plot_per_ue:
            res_folder_per_UE = os.path.join(res_folder, 'per_ue/')
            if not os.path.exists(res_folder_per_UE):
                os.makedirs(res_folder_per_UE)
            for ue_idx in range(n_ues):
                fig, axs = plt.subplots(4, n_config, figsize=(4 * n_config, 8), sharey='row',
                                        sharex='row')
                for jj in range(n_config):
                    ue = ues[jj][ue_idx]
                    axs[0, jj].plot(ue.al_hist, label='Aggregation Level (AL)')
                    axs[0, jj].set_ylim([0.5, 8.5])
                    axs[1, jj].plot(np.array(ue.n_re_alloc_hist) / n_re * 100, label='Allocated RE\'s [%]')
                    axs[1, jj].set_ylim([-1, max(n_re_alloc_mat_per_ue[jj].flatten()) / n_re * 100 + 1])
                    axs[2, jj].plot(np.cumsum(ue.n_re_alloc_hist), label='cumulative # RE\'s')
                    axs[2, jj].set_ylim([-1, max(n_re_alloc_mat_per_ue[jj].sum(axis=1)) + 1])
                    axs[3, jj].plot(ue.pf_metric_hist, label='Proportional Fairness (PF) metric')
                    axs[3, jj].set_ylim([0, max(pf_metric_mat[jj].flatten()) + 1])
                    axs[0, jj].set_title(f'opt: {mwis_algo_lab[jj]}')
                    for ii in range(4):
                        axs[ii, jj].grid()
                        axs[ii, jj].legend()
                        axs[ii, jj].set_xlabel('slot')
                fig.suptitle(f'RNTI={rnti_per_ue[ue_idx]}', y=.995)
                fig.tight_layout()
                fig.savefig(os.path.join(res_folder_per_UE, f'UE_{ue_idx}.png'))
                plt.close('all')

        print(f'results written in folder {res_folder}')
        # plt.show()
