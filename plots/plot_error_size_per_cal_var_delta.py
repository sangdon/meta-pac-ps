import os, sys
import numpy as np
import pickle
import glob
import argparse
import warnings
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot')
    ## parameters

    ## meta args
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--dataset', type=str, default='MiniImageNet')
    parser.add_argument('--model_name', type=str, default='protonet')

    parser.add_argument('--n_datasets', type=int, default=500)
    parser.add_argument('--n_ways', type=int, default=5)
    parser.add_argument('--n_shots', type=int, default=500)
    parser.add_argument('--n_test_cal_shots', type=int, default=20)    
    parser.add_argument('--eps', type=str, default='0.1')
    parser.add_argument('--alpha', type=str, default='0.1')
    parser.add_argument('--deltas', type=str, nargs='*', default=['1e-5', '0.1'])
    parser.add_argument('--fontsize', type=int, default=20)
    parser.add_argument('--figsize', type=float, nargs=2, default=[15.0, 5.0])
    args = parser.parse_args()
    
    fontsize = args.fontsize
    fig_root = f'{args.snapshot_root}/figs/{args.dataset}'

    method_list = [
        ('meta_pac_ps', 'Meta-PS'),
    ]
    
    # init
    os.makedirs(fig_root, exist_ok=True)

    # read results
    exp_name_list = []
    for delta in args.deltas:
        n = method_list[0]
        assert(n[0] == 'meta_pac_ps')
        exp_name_list.append(
            f'exp_{args.dataset}_{args.model_name}_{n[0]}_n_datasets_cal_{args.n_datasets}_n_ways_{args.n_ways}_n_shots_cal_{args.n_shots}_eps_{args.eps}_delta_{delta}_alpha_{args.alpha}'
        )

    name_list = []
    for delta in args.deltas:
        if delta == '0.1':
            name_list.append('$\delta = 0.1$')
        elif delta == '1e-3':
            name_list.append('$\delta = 10^{-3}$')
        elif delta == '1e-5':
            name_list.append('$\delta = 10^{-5}$')
        else:
            raise NotImplementedError
    

    # load results
    stats_list = []
    for exp_name in exp_name_list:
        res = []
        for fn in glob.glob(os.path.join(args.snapshot_root, exp_name+'_expid_*', 'stats_pred_set.pk')):
            r = pickle.load(open(fn, 'rb'))
            res_ds = []
            for s, e in zip(r['size_test'], r['error_test']):
                res_ds.append({'size_test': s.cpu().numpy(), 'error_test': e.cpu().numpy()})
            res.append(res_ds)
        stats_list.append(res)

    # summary results
    eps = float(args.eps)
    alpha = float(args.alpha)
    for n, stats in zip(method_list, stats_list):
        print(f'[method = {n[1]}, eps = {eps:e}], #calibration datasets = {len(stats)}, #test datasets = {len(stats[0])}')
    
    ## plot error
    error_list_per_cal = [[np.array([s['error_test'] for s in stat_cal]) for stat_cal in stats_method] for stats_method in stats_list]
    random.seed(10)
    for i in range(len(error_list_per_cal)):
        random.shuffle(error_list_per_cal[i])
        error_list_per_cal[i] = error_list_per_cal[i][:20] 

    fn = os.path.join(fig_root, f'plot_error_per_cal_n_datasets_{args.n_datasets}_n_ways_{args.n_ways}_n_shots_{args.n_shots}_eps_{args.eps}_alpha_{args.alpha}_delta_various')
    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1, figsize=args.figsize)
        plt.clf()
        hs = []
        leg_label = []
        min_len = min(len(error_list_per_cal[0]), len(error_list_per_cal[1]))
        
        # meta-ps
        error_list = error_list_per_cal[0][:min_len]
        h = plt.boxplot(
            error_list,
            positions=np.arange(len(error_list))-0.2,
            whis=(alpha*100, (1-alpha)*100),
            showmeans=True,
            widths=0.3,
            #patch_artist=True,
            boxprops=dict(linewidth=3, color='lightcoral'),
            whiskerprops=dict(color='lightcoral'),
            capprops=dict(color='lightcoral'),
            medianprops=dict(linewidth=3.0),
        )
        hs.append(h['boxes'][0])
        leg_label.append(name_list[0])

        # meta-ps
        error_list = error_list_per_cal[1][:min_len]
        h = plt.boxplot(
            error_list,
            positions=np.arange(len(error_list))+0.2,
            whis=(alpha*100, (1-alpha)*100),
            showmeans=True,
            widths=0.3,
            #patch_artist=True,
            boxprops=dict(linewidth=3, color='forestgreen'),
            whiskerprops=dict(color='forestgreen'),
            capprops=dict(color='forestgreen'),
            medianprops=dict(linewidth=3.0),
        )
        hs.append(h['boxes'][0])
        leg_label.append(name_list[1])

        # epsilon
        hs.append(
            plt.hlines(eps, -1.0, 1.0+len(error_list), colors='k', linestyles='dashed')
        )
        leg_label.append('$\epsilon = %.2f$'%(eps))

        # beautify
        plt.gca().set_xticks(np.arange(len(error_list)))
        plt.gca().set_xticklabels(np.arange(len(error_list)))
        
        plt.xlabel('calibration distribution index', fontsize=fontsize)
        plt.ylabel('prediction set error', fontsize=fontsize)
        plt.xlim((-1, 20))
        plt.ylim(bottom=0.0)
        plt.gca().tick_params('y', labelsize=fontsize*0.75)
        plt.grid('on')
        plt.legend(handles=hs, labels=leg_label, fontsize=fontsize)
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')
        print(f'[saved] {fn}')
    

