import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA, SparsePCA
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
import scipy.stats as scs
import seaborn as sns
from dataclasses import dataclass
from typing import NamedTuple, Any
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import random
from sklearn import svm
from scipy.stats import entropy
import matplotlib.ticker as mtick

TRIALS = 'test_trials.npy'
TARGETS = 'test_targets.npy'
PREDS = 'test_preds.npy'
LIFZ = 'test_lif0_z.npy'
LIFV = 'test_lif0_v.npy'
LIFI = 'test_lif0_i.npy'

BATCH_SIZE = 64
NEURON_NUM = 500
EXC_NUM = 400
DT = 2
MS = 0.001

TRIAL_TIME = {
    'fixation': 200,
    'cueing': 100,
    'cueing_delay': 400,
    'stimuls': 100
}

DELAY_START = int((TRIAL_TIME['fixation'] + TRIAL_TIME['cueing']) / DT)
DELAY_END = DELAY_START + int(TRIAL_TIME['cueing_delay'] / DT)


@dataclass
class TrialCollection:
    desc: Any
    test_trials: np.array
    test_targets: np.array
    test_preds: np.array
    test_lif0_z: np.array
    test_lif0_v: np.array
    test_lif0_i: np.array
    test_readout_v: np.array
    test_readout_i: np.array


@dataclass
class AnalysisCollection:
    path: Any = None
    pca: Any = None
    acc_dict: Any = None
    c0_spikes: Any = None
    c1_spikes: Any = None
    c0_fr: Any = None
    c1_fr: Any = None
    pca_projs: Any = None
    pca_error: Any = None
    all_pca_projs: Any = None
    peak_diff: Any = None


@dataclass
class CellFr:
    fr: Any = None
    mean: Any = None
    std: Any = None
    peak: Any = None
    ci: Any = None


def read_trials(path):
    desc = None
    with open(path + 'test_desc.json', 'rb') as fp:
        desc = json.load(fp)
    trial_info = TrialCollection(
        desc=desc,
        test_trials=np.load(path + "test_trials.npy"),
        test_targets=np.load(path + "test_targets.npy"),
        test_preds=np.load(path + "test_preds.npy"),
        test_lif0_z=np.load(path + "test_lif0_z.npy"),
        test_lif0_v=np.load(path + "test_lif0_v.npy"),
        test_lif0_i=np.load(path + "test_lif0_i.npy"),
        test_readout_v=np.load(path + "test_readout_v.npy"),
        test_readout_i=np.load(path + "test_readout_i.npy")
    )
    seq_len, batch_size, neuron_num = np.shape(trial_info.test_lif0_z)
    return trial_info, seq_len, batch_size, neuron_num


def get_accuracy(trials,cc_num=1):
    trial_dict = {}
    for cc in range(cc_num):
        for c in range(2):
            for truth in range(2):
                trial_dict[(cc, c, truth)] = []

    for index, d in enumerate(trials.desc):
        trial_dict[(int(d['cc']), int(d['c']), int(d['truth']))].append(index)

    accuracy = np.equal(trials.test_targets, np.squeeze(trials.test_preds)).astype(int)
    print('All Accuracy: ', np.sum(accuracy) / len(trials.test_targets))
    acc_dict = {}
    for key in trial_dict.keys():
        acc_dict[key] = np.sum(accuracy[trial_dict[key]]) / len(trial_dict[key])

    correct_index = np.where(accuracy == 1)
    print('Visualize correct ', np.sum(accuracy), ' cases in ',len(trials.test_targets))
    return acc_dict, trial_dict, correct_index


def get_con_interval(ss, bin_size, delay_start, delay_end):
    ss = np.array([s[delay_start:delay_end] for s in ss])
    frs = np.array([np.mean(np.reshape(s, (-1, bin_size)), axis=1) / (MS * DT * bin_size) for s in ss])
    trial_num = np.shape(frs)[0]
    std = np.std(frs, axis=0)
    return std / np.sqrt(trial_num)


def get_cell_fr(spike, bin_size, sigma=2, std_th=2., delay_start=DELAY_START, delay_end=DELAY_END, peak_distance=10):
    input_spike = spike.T.cpu().numpy()
    smooth_spikes = gaussian_filter1d(input_spike, sigma)[:, delay_start:delay_end]
    
    frs = np.reshape(np.mean(smooth_spikes, axis=0), (-1, bin_size)) 
    fr = np.sum(frs, axis=1)/ (MS * DT * bin_size)
    
    fr_mean = np.mean(fr)
    fr_std = np.std(fr) #* std_th
    peak, _ = find_peaks(fr, height=fr_mean, distance=peak_distance)
    ci = get_con_interval(input_spike, bin_size=bin_size, delay_start=delay_start, delay_end=delay_end)

    return CellFr(fr, fr_mean, fr_std, peak, ci)


def plot_spikes(n_id, results: AnalysisCollection, bin_size=2, delay_start=DELAY_START, delay_end=DELAY_END,
                width=18, height=10, if_save=False, saved_name='raster.svg'):
    cell_fr0, cell_fr1 = results.c0_fr[n_id], results.c1_fr[n_id]
    print(np.mean(cell_fr0.fr),np.mean(cell_fr1.fr))
    base_len = int((DELAY_END-DELAY_START)/bin_size)
    
    if np.std(cell_fr0.fr) == 0.:
        fr0 = cell_fr0.fr
        ci0 = 0
    else:
        zscore_m0, zscore_std0 = np.mean(cell_fr0.fr[:base_len]), np.std(cell_fr0.fr[:base_len])
        fr0 = (cell_fr0.fr-zscore_m0)/zscore_std0
        ci0 = cell_fr0.ci/np.std(cell_fr0.fr)

    if np.std(cell_fr1.fr) == 0.:
        fr1 = cell_fr1.fr
        ci1 = 0
    else:
        zscore_m1, zscore_std1 = np.mean(cell_fr1.fr[:base_len]), np.std(cell_fr1.fr[:base_len])
        fr1 = (cell_fr1.fr-zscore_m1)/zscore_std1
        ci1 = cell_fr1.ci/np.std(cell_fr1.fr)
    
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (width, height),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    fig, ax = plt.subplots(2, 2)

    shift = int(bin_size / 2)
    x_labels = np.arange(np.shape(cell_fr0.fr)[0]) + shift
    
    y_min = -4.#min([min(fr0),min(fr1)])-0.2#-1.2#cell_fr0.mean - cell_fr0.std * 2
    y_max = 4 #max([max(fr0),max(fr1)])+0.2 #cell_fr0.mean + cell_fr0.std * 2.5

    
    if delay_end > DELAY_END:
        flag =(DELAY_END-DELAY_START)/(delay_end-DELAY_START)*len(x_labels)
#         print(DELAY_END-DELAY_START, delay_end-DELAY_START)
#         ax[1][0].axvline(x=flag, color='green', ls=':', lw=2)
#         ax[1][1].axvline(x=flag , color='green', ls=':', lw=2)
        # ax[2][0].axvline(x=flag , color='green', ls=':', lw=2)
        # ax[2][1].axvline(x=flag , color='green', ls=':', lw=2)
    
    ax[1][0].text(0.1, 0.9, np.around(np.mean(cell_fr0.fr), decimals=2), size=25)
    ax[1][0].fill_between(x_labels, (fr0 - ci0), (fr0 + ci0), color='b', alpha=.3)
    ax[1][1].text(0.1, 0.9, np.around(np.mean(cell_fr1.fr), decimals=2), size=25)
    ax[1][1].fill_between(x_labels, (fr1 - ci1), (fr1 + ci1), color='r', alpha=.3)

    ax[1][0].set_ylim(y_min, y_max)
    ax[1][1].set_ylim(y_min, y_max)
    
    x_sticks = np.arange(0, delay_end - delay_start+100, 100)
    
    inputs = [results.c0_spikes[n_id], results.c1_spikes[n_id]]
    for p in [0, 1]:
        curr_input = inputs[p][delay_start:delay_end].to_sparse().coalesce()

        t = curr_input.indices()[0]
        n = curr_input.indices()[1]

#         ax[0][p].scatter(x_sticks[0], 0, marker='^', color='blue') 
#         ax[0][p].scatter(x_sticks[-1], 0, marker='^', color='blue')

        ax[0][p].scatter(t, n, marker='|', color='black',s=40, linewidth=1)
        ax[0][p].set_xticks(x_sticks)
        ax[0][p].title.set_text('Context ' + str(p))
        
        if delay_end > DELAY_END: 
            flag = x_sticks[int((DELAY_END-DELAY_START)/(delay_end-DELAY_START)*len(x_sticks))]
#             ax[0][p].axvline(x=flag-1, color='green', ls=':', lw=2)
    
    ax[0][0].set_xticklabels([])
    ax[0][0].set_yticklabels([])
    ax[0][1].set_xticklabels([])
    ax[0][1].set_yticklabels([])
    ax[1][0].set_xticklabels([])
    ax[1][1].set_xticklabels([])
    ax[1][1].set_yticklabels([])
    
    ax[0][0].set_ylabel('Trials')
    ax[1][0].set_ylabel('Firing Rate')
    ax[1][0].set_xlabel('Time')
    ax[1][1].set_xlabel('Time')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.autoscale(False)

    print('Neuron ' + str(n_id) + ' spike samples')
    if if_save: plt.savefig(saved_name)
    plt.show()
    

# def check_peak_diff(fr0, fr1, tolerance=1.5):
#     diff_count = 0
#     for p in fr0.peak:
#         if fr0.fr[p] - fr0.ci[p]*tolerance >= fr1.fr[p] + fr1.ci[p]*tolerance:
#             diff_count += 1
#     for p in fr1.peak:
#         if fr1.fr[p] - fr1.ci[p]*tolerance >= fr0.fr[p] + fr0.ci[p]*tolerance:
#             diff_count += 1

#     return diff_count


# def check_all_peak_diff(analysis: AnalysisCollection):
#     peak_diff_list = []
#     for n_id in range(NEURON_NUM):
#         peak_diff_list.append(check_peak_diff(analysis.c0_fr[n_id], analysis.c1_fr[n_id]))
#     return peak_diff_list


def create_index(c_neurons):
    c_index = {}
    c_max = np.argmax(c_neurons, axis=1)
    for index, n in enumerate(c_neurons):
        c_index[index] = c_max[index]
    c_index = {k: v for k, v in sorted(c_index.items(), key=lambda item: item[1])}
    return list(c_index.keys())


def get_neuron_context_fr(spikes, bin_size=4):
    m_spike = np.mean(np.array(spikes), 2)
    fr = np.array([gaussian_filter1d(s, bin_size) for s in m_spike])
    fr = np.sum(np.reshape(fr, (EXC_NUM, -1, bin_size)), axis=2) / (MS * DT * bin_size)

    return fr


def get_trial_smooth_fr(trial_s, delay_time, bin_size, sigma=10):

    smooth_f = np.array([gaussian_filter1d(f, sigma) for f in trial_s])
    binned_f = np.sum(np.reshape(smooth_f, (EXC_NUM, int(delay_time // bin_size), bin_size)), axis=2) / (
           MS * DT * bin_size)
    return binned_f.T

def mesq(a, b, ax=0):
    return (np.square(a - b)).mean(axis=ax)


def plot_metrics(metrics, xs):
    _ = plt.figure(figsize=(10, 6))
    plt.plot(xs, metrics, 'bo-')
    for x, y in zip(xs, metrics):
        label = "{:.4f}".format(y)

        plt.annotate(label,
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')
    plt.show()

    
def plot_EI_fr(analy, delay_start=DELAY_START, delay_end=DELAY_END, th=0.5,if_plot=False,if_save=False,saved_nam=''):
    trailN = np.shape(analy.c0_spikes.cpu().numpy())[-1]
    time = (delay_end-delay_start)*DT*MS

    all_fs = np.zeros(NEURON_NUM)
    for n in range(NEURON_NUM):
        
        curr_frs = []
        for tr in range(trailN):
            curr_frs.append(np.sum((analy.c0_spikes.cpu().numpy()[n,delay_start:delay_end,tr]+ 
                    analy.c1_spikes.cpu().numpy()[n,delay_start:delay_end,tr])/2)/(time))
        
        all_fs[n] = np.mean(curr_frs)
    
    exc_f, ih_f = all_fs[:EXC_NUM],all_fs[EXC_NUM:]
    
    exc_f = exc_f[exc_f>th]
    ih_f = ih_f[ih_f>th]
    exc_f_m, ih_f_m = np.mean(exc_f), np.mean(ih_f)
    exc_f_s, ih_f_s = np.std(exc_f)/np.sqrt(EXC_NUM), np.std(ih_f)/np.sqrt(NEURON_NUM-EXC_NUM)
    
    if if_plot:
        params = {'legend.fontsize': 'xx-large',
                  'figure.figsize': (6, 8),
                 'axes.labelsize': 'xx-large',
                 'axes.titlesize':'xx-large',
                 'xtick.labelsize':'xx-large',
                 'ytick.labelsize':'xx-large',
                 'errorbar.capsize': 5}
        plt.rcParams.update(params)
        
        (_, caps, _) = plt.errorbar(['delay E','delay I'], 
                                    [exc_f_m,ih_f_m], 
                                    [exc_f_s,ih_f_s], 
                                    linestyle='None', fmt='-o',ms=3, lw=1., capsize=5, capthick=2)
        
        if if_save: plt.savefig(saved_name)
        plt.show()
    return exc_f_m, ih_f_m, exc_f_s, ih_f_s, exc_f, ih_f

    
def plot_analy_info(dir_analysis: [AnalysisCollection],info, ax):
    if info == 'acc':
        info_list = [np.mean(list(analy.acc_dict.values())) for analy in dir_analysis]
    elif info == 'pca_e':
        info_list = [analy.pca_error for analy in dir_analysis]
    else:
        info_list = [np.mean(analy.peak_diff) for analy in dir_analysis]
    plot_metrics(info_list,ax)
    return info_list



c0_patch = mpatches.Patch(color='blue', alpha=.3, label='Rule 1')
c1_patch = mpatches.Patch(color='red', alpha=.3, label='Rule 2')
t0_patch = Line2D([0], [0], color='black', linestyle='-', alpha=.3, label='Choice 1')
t1_patch = Line2D([0], [0], color='black', linestyle='--', alpha=.3, label='Choice 2')

def draw_comp_2Dpca(base_pca_projs, single_projss,cc, if_save=False, saved_name='sequence.svg'):
    params = {'legend.fontsize': 'xx-large',
      'figure.figsize': (18, 20),
     'axes.labelsize': 'xx-large',
     'axes.titlesize':'xx-large',
     'xtick.labelsize':'xx-large',
     'ytick.labelsize':'xx-large'}
    plt.rcParams.update(params)

    _ = plt.figure(figsize=(12, 8))

    c1_pj = np.mean(base_pca_projs[:2], axis=0)
    c_pcas = c1_pj - c1_pj[0]

    x = c_pcas[:, 0]
    y = c_pcas[:, 1]
    plt.plot(x, y, color='b', alpha=.3)
    plt.scatter(x, y, color='b', alpha=.3)
    plt.scatter(x[-1], y[-1], marker='o', color='b', alpha=.3)

    c2_pj = np.mean(base_pca_projs[2:], axis=0)
    c_pcas = c2_pj - c2_pj[0]
    x = c_pcas[:, 0]
    y = c_pcas[:, 1]
    plt.plot(x, y, color='r', alpha=.3)
    plt.scatter(x, y, color='r', alpha=.3)
    plt.scatter(x[-1], y[-1], marker='o', color='r', alpha=.3)
    
    for single_projs in single_projss:
        c_pcas = single_projs - single_projs[0]
        x = c_pcas[:, 0]
        y = c_pcas[:, 1]
        plt.plot(x, y, color='b',  linestyle='--',alpha=.3)
        plt.scatter(x, y, color='b', alpha=.3)
        plt.scatter(x[-1], y[-1], marker='o', color='b', alpha=.3)
    
    plt.scatter(0, 0, marker='x', color='m')
    
    corr_patch = Line2D([0], [0], color='black', alpha=.3, label='Correct')
    err_patch = Line2D([0], [0], color='black', linestyle='--', alpha=.3, label='Error')
    plt.legend(handles=[c0_patch, c1_patch,corr_patch,err_patch], loc='upper center', prop={'size': 30})
    
    if if_save: plt.savefig(saved_name)
    plt.show()
    

class ResultsVisualization:
    def __init__(self, path, bin_size=4, pca_bin_size=10, min_len=50, cc_num=1):
        self.base = AnalysisCollection(path=path + '/base_')

        self.seq_len, self.batch_size, self.neuron_num, self.paras = None, None, None, None
        self.bin_size = bin_size
        self.pca_bin_size = pca_bin_size
        self.base_pca = None
        self.min_len = min_len
        self.cc_num = cc_num
        
        self.tuned_ids = None
        self.tuned_c0_ids = None
        self.tuned_c1_ids = None
        

        with open(path + '/parameters.json', 'rb') as fp:
            self.paras = json.load(fp)


    def get_pca(self, spikes, trial_dict, correct_index, sample_size, delay_start=DELAY_START, delay_end=DELAY_END, if_base=False, if_all_pcas=False):
        all_pca_spikes = np.array(spikes).transpose([2, 0, 1])[:, :, delay_start:delay_end]
        if self.cc_num==2:
            c00 = np.intersect1d(trial_dict[(0, 0, 0)]+trial_dict[(0, 0, 1)], correct_index)[:sample_size]
            c01 = np.intersect1d(trial_dict[(1, 0, 0)]+trial_dict[(1, 0, 1)], correct_index)[:sample_size]
            
            c10 = np.intersect1d(trial_dict[(0, 1, 0)]+trial_dict[(0, 1, 1)], correct_index)[:sample_size]
            c11 = np.intersect1d(trial_dict[(1, 1, 0)]+trial_dict[(1, 1, 1)], correct_index)[:sample_size]
        else:
            c00 = np.intersect1d(trial_dict[(0, 0, 0)], correct_index)[:sample_size]
            c01 = np.intersect1d(trial_dict[(0, 0, 1)], correct_index)[:sample_size]
    
            c10 = np.intersect1d(trial_dict[(0, 1, 0)], correct_index)[:sample_size]
            c11 = np.intersect1d(trial_dict[(0, 1, 1)], correct_index)[:sample_size]

        context_index = [c00, c01, c10, c11]
        for i in range(len(context_index)):
            if len(context_index[i]) == 0:
                context_index[i] = [0] 

        min_len = min([len(c) for c in context_index])

        context_index = [c[np.random.choice(np.arange(len(c)),size=min_len,replace=False)] for c in context_index]

        spike_list = np.array([all_pca_spikes[c, :EXC_NUM, :] for c in context_index]) 

        all_pca_projs = np.array([self.get_pca_proj(ss, delay_end - delay_start, if_base=if_base) for ss in spike_list])
        pca_projs = np.mean(all_pca_projs, axis=1)

        error = np.mean([mesq(pca_projs[0][-1], pca_projs[1][-1]),
                         mesq(pca_projs[2][-1], pca_projs[3][-1])])
        if if_all_pcas:
            return pca_projs, error, all_pca_projs
        else:
            return pca_projs, error, None
    
    
    def get_pca_proj(self, spikes, delay_time, if_base=False ):
        fr_list = [get_trial_smooth_fr(s, delay_time, self.pca_bin_size) for s in spikes] 
        frs = np.vstack(fr_list)

        if if_base:
            self.base_pca_len = np.shape(fr_list[0])[0]
            self.base_pca = PCA(n_components=400)
            self.base_pca.fit(frs)
        projected_fr = np.array([self.base_pca.transform(f) for f in fr_list])

        return projected_fr

    
    def draw_2Dpca(self, pca_projs, if_save=False, saved_name='sequence.svg'):
        params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (18, 20),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
        plt.rcParams.update(params)

        _ = plt.figure(figsize=(12, 8))
    
        color = ['b', 'b', 'r', 'r']
        linestyles = ['-', '--','-', '--']
        for index, c_pcas in enumerate(pca_projs):
            c_pcas = c_pcas - c_pcas[0]
            x = c_pcas[:, 0]
            y = c_pcas[:, 1]
            plt.plot(x[:self.base_pca_len], y[:self.base_pca_len], color=color[index], linestyle=linestyles[index], alpha=.3)
            plt.plot(x[self.base_pca_len-1:], y[self.base_pca_len-1:], color=color[index], linestyle=':', alpha=.3)
            plt.scatter(x, y, color=color[index], alpha=.3)
            plt.scatter(x[-1], y[-1], marker='o', color=color[index], alpha=.3)
    
        plt.scatter(0, 0, marker='x', color='m')
        plt.legend(handles=[c0_patch, c1_patch,t0_patch,t1_patch], loc='upper center', prop={'size': 30})
        if if_save: plt.savefig(saved_name)
        plt.show()

        
    def draw_2Dmomen(self, pca_projs, if_save=False, saved_name='sequence.svg'):
        params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (18, 20),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
        plt.rcParams.update(params)

        _ = plt.figure(figsize=(12, 8))
    
        color = ['b', 'b', 'r', 'r']
        linestyles = ['-', '--','-', '--']
        for index, c_pcas in enumerate(pca_projs):
            c_pcas = c_pcas - c_pcas[0]
            x = c_pcas[:, 0]
            x_ = x[1:]
            x = x_ - x[:-1]
            
            y = c_pcas[:, 1]
            y_ = y[1:]
            y = y_- y[:-1]
            momen = np.array([a**2/2 for a in x]) #list(map(lambda a: a**2/2, x)))#np.array(list(map(lambda a: np.sqrt(a[0]**2+a[1]**2), zip(x,y))))
            plt.plot(np.arange(len(momen)), momen, linestyle=linestyles[index], color=color[index], alpha=.3)
            plt.scatter(np.arange(len(momen)), momen, color=color[index], alpha=.3)
#         print(np.arange(len(momen)).astype(int)*self.pca_bin_size*DT)
        plt.xticks(np.arange(len(momen),5), labels=np.arange(len(momen),5).astype(int)*self.pca_bin_size*DT)
#         plt.xticklabels = np.arange(len(momen),5).astype(int)*self.pca_bin_size*DT
        plt.xlabel('Time/ms')
        plt.legend(handles=[c0_patch, c1_patch,t0_patch,t1_patch], prop={'size': 30})
        if if_save: plt.savefig(saved_name)
        plt.show()

    def analysis_trial_dict(self, path, delay_time, min_len=50, if_base=False, if_error=False, if_all_trial=False, if_all_pcas=False):
        trial_info, seq_len, batch_size, neuron_num = read_trials(path)
        acc_dict, trial_dict, correct_index = get_accuracy(trial_info, self.cc_num)

        if if_error:
            c0 = np.setdiff1d(trial_dict[(0, 0, 0)] + trial_dict[(0, 0, 1)], correct_index)
            c1 = np.setdiff1d(trial_dict[(0, 1, 0)] + trial_dict[(0, 1, 1)], correct_index)
        if if_all_trial:
            c0 = np.array(trial_dict[(0, 0, 0)] + trial_dict[(0, 0, 1)])
            c1 = np.array(trial_dict[(0, 1, 0)] + trial_dict[(0, 1, 1)])
            correct_index = np.arange(len(trial_info.test_targets))
        elif self.cc_num==2:
            c0 = np.intersect1d(trial_dict[(0, 0, 0)] + trial_dict[(0, 0, 1)]+trial_dict[(1, 0, 0)] + trial_dict[(1, 0, 1)], correct_index)
            c1 = np.intersect1d(trial_dict[(0, 1, 0)] + trial_dict[(0, 1, 1)]+trial_dict[(1, 1, 0)] + trial_dict[(1, 1, 1)], correct_index)
        else:
            c0 = np.intersect1d(trial_dict[(0, 0, 0)] + trial_dict[(0, 0, 1)], correct_index)
            c1 = np.intersect1d(trial_dict[(0, 1, 0)] + trial_dict[(0, 1, 1)], correct_index)
        
        correct_len = min(np.shape(c0)[0], np.shape(c1)[0])#min([min(np.shape(c0)[0], np.shape(c1)[0]), self.min_len])
#         print('0shape',np.shape(c0))
#         print('1shape',np.shape(c1))
        if not if_all_trial:
            self.min_len=min([correct_len, self.min_len])
            
        c0 = c0[np.random.choice(np.arange(len(c0)),size=self.min_len,replace=False)]
        c1 = c1[np.random.choice(np.arange(len(c1)),size=self.min_len,replace=False)]

        all_spikes = torch.tensor(trial_info.test_lif0_z).transpose(0, 2).transpose(1, 2)
        
        c0_spikes = all_spikes[:, :, c0].squeeze()
        c1_spikes = all_spikes[:, :, c1].squeeze()
        
        
        pca_projs, pca_error, all_pca_projs = self.get_pca(all_spikes, trial_dict, correct_index, correct_len, delay_start=DELAY_START,
                                      delay_end=DELAY_START + delay_time, if_base=if_base,if_all_pcas=if_all_pcas) 
        return acc_dict, c0_spikes, c1_spikes,  pca_projs, pca_error, all_pca_projs


    def get_analysis(self, analysis, bin_size, delay_time=int(TRIAL_TIME['cueing_delay'] / DT), if_base=False, if_error=False,if_all_trial=False,if_all_pcas=False):
        print('Analysis: '+analysis.path)
        delay_end = DELAY_START + delay_time
        analysis.acc_dict, analysis.c0_spikes, analysis.c1_spikes, analysis.pca_projs, analysis.pca_error , analysis.all_pca_projs= self.analysis_trial_dict(analysis.path, delay_time, if_base=if_base, if_error=if_error,if_all_pcas=if_all_pcas, if_all_trial=if_all_trial)
        analysis.c0_fr = [get_cell_fr(spikes, bin_size, sigma=bin_size, delay_start=DELAY_START, delay_end=delay_end) for spikes in
                          analysis.c0_spikes]
        analysis.c1_fr = [get_cell_fr(spikes, bin_size, sigma=bin_size, delay_start=DELAY_START, delay_end=delay_end) for spikes in
                          analysis.c1_spikes]
        # analysis.peak_diff = check_all_peak_diff(analysis)
        return analysis

    
    def get_base_analysis(self,if_all_pcas=False, if_all_trial=False):
        self.get_analysis(self.base, self.bin_size, if_base=True, if_all_pcas=if_all_pcas, if_all_trial=if_all_trial)
        
#     def get_xxx_analysis(self,...):
#         write your own analysis function for your perturbation AnalysisCollection instance


    def draw_sequence(self, c0_spikes, c1_spikes, fr_th=2, peak_th=2.5,if_save=False, saved_name='sequence.svg', if_base=False, if_show=True):
        time_steps = np.shape(c0_spikes)[1]
        trial_fr0 = get_neuron_context_fr(c0_spikes)
        trial_fr1 = get_neuron_context_fr(c1_spikes)
 
        if if_base:
            peak_ids = []
            peak_c0_ids = []
            peak_c1_ids = []
            
            h_fr_ids = []
            for i in range(EXC_NUM):
                avg_fr= np.sum(c0_spikes[i]+c1_spikes[i]) / (2*MS * DT * time_steps*self.min_len)
                if avg_fr > fr_th:
                    h_fr_ids.append(i)
                
                zs_trial0, zs_trial1 = stats.zscore(trial_fr0[i]), stats.zscore(trial_fr1[i])
                peak0_ids, _ = find_peaks(zs_trial0, height=peak_th)
                peak1_ids, _ = find_peaks(zs_trial1, height=peak_th)
                if len(peak0_ids) > 0 or len(peak1_ids) > 0:
                    peak_ids.append(i)
                    if max(zs_trial0) > max(zs_trial1):
                        peak_c0_ids.append(i)
                    else:
                        peak_c1_ids.append(i)
                        
            peak_ids = np.array(peak_ids)
            print(len(peak_ids))
            peak_c0_ids = np.array(peak_c0_ids)
            peak_c1_ids = np.array(peak_c1_ids)
            
            h_fr_ids = np.array(h_fr_ids)
            print(len(h_fr_ids))
    
            self.tuned_ids = np.intersect1d(peak_ids,h_fr_ids)
            self.tuned_c0_ids = np.intersect1d(peak_c0_ids,h_fr_ids)
            self.tuned_c1_ids = np.intersect1d(peak_c1_ids,h_fr_ids)

        
        normed_fr0 = np.array([(f-np.min(f))/(np.max(f)-np.min(f)) for f in trial_fr0])
        normed_fr1 = np.array([(f-np.min(f))/(np.max(f)-np.min(f)) for f in trial_fr1])
    
        normed_fr0 = normed_fr0[self.tuned_ids]
        normed_fr1 = normed_fr1[self.tuned_ids]
        
        if if_base:
            self.c0_seq_index = create_index(normed_fr0)
            self.c1_seq_index = create_index(normed_fr1)
        
        params = {'legend.fontsize': 'x-large',
              'figure.figsize': (18, 20),
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
        plt.rcParams.update(params)
    
        fig, ax = plt.subplots(2, 2)
    
        for a in [0,1]:
            for b in [0,1]:
                xmin, xmax = ax[a][b].get_xlim()
                ymin, ymax = ax[a][b].get_ylim()
                norm_f = normed_fr0 if  a == 0 else normed_fr1 
                c_index = self.c0_seq_index if  b == 0 else self.c1_seq_index 
                im=ax[a][b].imshow(norm_f[c_index], cmap='rainbow', extent=(xmin,xmax,ymin,ymax))
                ax[a][b].set_xticklabels([])
                ax[a][b].set_yticklabels([])
                ax[a][b].set_xlabel('Time')
    
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax[1][1],
                       width="5%",  
                       height="50%",  
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax[1][1].transAxes,
                       borderpad=0,
                       )
    
    
        fig.colorbar(im, cax=axins, ticks=np.arange(0, 1.1, 0.2))
    
        ax[0][0].set_ylabel('Neurons')
        ax[1][0].set_ylabel('Neurons')
        
    
        plt.subplots_adjust(wspace=0, hspace=0)
        if if_save: plt.savefig(saved_name)
        if if_show: plt.show()
