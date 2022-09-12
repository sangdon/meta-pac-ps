import os, sys
import argparse
import warnings
import numpy as np
import math
import pickle
import types

import torch as tc

import util
import data
import model
import learning
import uncertainty

    
def main(args):

    ## init datasets
    print("## init datasets: %s"%(args.data.src))    
    ds = getattr(data, args.data.src)(
        root=os.path.join('data', args.data.src.lower()),
        args=args.data,
    )
    print()

    ## init a model
    print("## init models: %s"%(args.model.base))    
    if 'FNN' in args.model.base or 'Linear' in args.model.base:
        mdl = getattr(model, args.model.base)(n_in=args.data.dim[0], n_out=args.data.n_labels, path_pretrained=args.model.path_pretrained)    
    elif 'ResNet' in args.model.base:
        mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels, path_pretrained=args.model.path_pretrained)
    elif 'ProtoNet' == args.model.base and args.data.src == 'MiniImageNet':
        mdl = getattr(model, args.model.base)(backbone='Convnet4', n_shots=args.data.n_shots_adapt, n_ways=args.data.n_ways,
                                              path_pretrained=args.model.path_pretrained)
    elif 'ProtoNetNLP' == args.model.base:
        mdl = getattr(model, args.model.base)(encoder=ds.encoder,
                                              n_shots_adapt=args.data.n_shots_adapt, n_shots_test=args.data.n_shots_test, n_ways=args.data.n_ways,
                                              path_pretrained=args.model.path_pretrained)
    elif 'ProtoNetGeneral' == args.model.base and args.data.src == 'Diabetes':
        mdl = getattr(model, args.model.base)(backbone='DiabetesFNN', n_samples_adapt=args.data.n_shots_adapt*args.data.n_ways, n_ways=args.data.n_ways,
                                              path_pretrained=args.model.path_pretrained)
    elif 'ProtoNetGeneral' == args.model.base and args.data.src == 'Heart':
        mdl = getattr(model, args.model.base)(backbone='HeartFNN', n_samples_adapt=args.data.n_shots_adapt*args.data.n_ways, n_ways=args.data.n_ways,
                                              path_pretrained=args.model.path_pretrained)
    elif args.data.src == 'Chembl':
        mdl = getattr(model, args.model.base)(args.model)
        mdl.load_state_dict(tc.load(args.model.path_pretrained))
    else:
        raise NotImplementedError

    if args.multi_gpus:
        assert(not args.cpu)
        mdl = tc.nn.DataParallel(mdl).cuda()
    print()

    ## compute lower bound
    if args.lowerbound:
        raise NotImplementedError
    
        # print('## compute performance lower bound')
        # l = learning.ClsLearner(mdl, args.train)
        # ld_train = data.ConcatDataLoader([l.train for l in ds.test])
        # ld_val = data.ConcatDataLoader([l.val for l in ds.test])
        # ld_test = data.ConcatDataLoader([l.test for l in ds.test])

        # l.train(ld_train, ld_val=ld_val, ld_test=ld_test)
        # l.test(ld_test, ld_name=f'test domain', verbose=True)
        
    else:
        if args.data.src == 'Heart':
            print('## a few-shot detection learning')
            l = learning.FewshotDetLearner(mdl, args.train)
            if args.model.path_pretrained is None:
                l.train(ds.train, ld_val=ds.val, ld_test=ds.test)
            print()
            print('## a few-shot detection test')
            l.test(ds.test, ld_name=f'test domain', verbose=True)
            
        elif args.data.src == 'Chembl':
            print('## a few-shot regression learning')
            l = learning.FewshotRegLearner(mdl, args.train)
            if args.model.path_pretrained is None:
                l.train(ds.train, ld_val=ds.val, ld_test=ds.test)
            print()
            print('## a few-shot regression test')
            l.test(ds.test, ld_name=f'test domain', verbose=True)

        else:
            print('## a few-shot classification learning')
            l = learning.FewshotClsLearner(mdl, args.train)
            if args.model.path_pretrained is None:
                l.train(ds.train, ld_val=ds.val, ld_test=ds.test)
            print()
            print('## a few-shot classification test')
            l.test(ds.test, ld_name=f'test domain', verbose=True)
        
    print()

    ## prediction set estimation
    if args.train_ps.method is None:
        return
    elif args.train_ps.method == 'pac_predset':
        raise NotImplementedError
    
        warnings.warn('the original pac_predset might be slow if m is large')
        if args.data.src == 'Chembl':
            mdl_predset = model.PredSetReg(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
        else:
            mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
        l = uncertainty.PredSetConstructor(mdl_predset, args.train_ps)        
        ld_cal = data.ConcatDataLoader([l.val for l in ds.cal])
        ld_test = data.ConcatDataLoader([l.test for l in ds.test])
        
    elif args.train_ps.method == 'meta_ps_naive':
        
        if args.data.src == 'Chembl':
            mdl_predset = model.PredSetReg(mdl)
        else:
            mdl_predset = model.PredSetCls(mdl)
        l = uncertainty.PredSetConstructor_meta_naive(mdl_predset, args.train_ps)
        ld_cal = ds.cal
        ld_test = ds.test
        params = types.SimpleNamespace(n=args.data.n_datasets_cal*args.data.n_shots_cal*args.data.n_ways,
                                       eps=args.train_ps.eps, delta=args.train_ps.delta, verbose=True, save=True)

    elif args.train_ps.method == 'meta_ps_ideal':
        if args.data.src == 'Chembl':
            mdl_predset = model.PredSetReg(mdl)
        else:
            mdl_predset = model.PredSetCls(mdl)

        l = uncertainty.PredSetConstructor_meta_ideal(mdl_predset, args.train_ps)
        ld_cal = None
        ld_test = ds.test
        params = types.SimpleNamespace(n=args.data.n_shots_cal*args.data.n_ways,
                                       n_adapt=args.data.n_shots_adapt*args.data.n_ways,
                                       n_ways=args.data.n_ways,
                                       n_shots_adapt=args.data.n_shots_adapt,
                                       n_shots_cal=args.data.n_shots_cal,
                                       n_shots_test=args.data.n_shots_test,
                                       eps=args.train_ps.eps, delta=args.train_ps.delta, verbose=True, save=False)
        
    elif args.train_ps.method == 'meta_pac_ps':
        if args.data.src == 'Chembl':
            mdl_predset = model.PredSetReg(mdl)
        else:
            mdl_predset = model.PredSetCls(mdl)

        l = uncertainty.PredSetConstructor_meta(mdl_predset, args.train_ps)        
        ld_cal = ds.cal
        ld_test = ds.test
        params = types.SimpleNamespace(n_datasets=args.data.n_datasets_cal, n_shots_cal=args.data.n_shots_cal, n_ways=args.data.n_ways,
                                       n_shots_test=args.data.n_shots_test, 
                                       eps=args.train_ps.eps, delta=args.train_ps.delta, alpha=args.train_ps.alpha, save=True)
        
    else:
        raise NotImplementedError
    
    l.train(ld_cal, params)
    l.test(ld_test, ld_name=f'test datasets', verbose=True)

    
def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='Meta Prediction Set')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--multi_gpus', action='store_true')
    parser.add_argument('--lowerbound', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=100)
    parser.add_argument('--data.n_workers', type=int, default=8)
    parser.add_argument('--data.src', type=str, required=True)
    parser.add_argument('--data.n_labels', type=int)
    #parser.add_argument('--data.n_examples', type=int, nargs=3)
    #parser.add_argument('--data.n_datasets', type=int, nargs=4)
    parser.add_argument('--data.seed', type=lambda v: None if v=='None' else int(v), default=0)
    #parser.add_argument('--data.n_max_aug', type=int, default=5)
    #parser.add_argument('--data.pert_type', type=lambda v: None if v=='None' else v, default=None)

    parser.add_argument('--data.n_datasets_train', type=int, default=800)
    parser.add_argument('--data.n_datasets_val', type=int, default=50)
    parser.add_argument('--data.n_datasets_cal', type=int, default=50)
    parser.add_argument('--data.n_datasets_test', type=int, default=50)
    parser.add_argument('--data.n_ways', type=int, default=5)
    
    parser.add_argument('--data.n_shots_adapt', type=int, default=5)
    parser.add_argument('--data.n_shots_train', type=int, default=5)
    parser.add_argument('--data.n_shots_val', type=int, default=5)
    parser.add_argument('--data.n_shots_cal', type=int, default=5)
    parser.add_argument('--data.n_shots_test', type=int, default=50)
    #parser.add_argument('--data.n_trials_test', type=int, default=10)

    #parser.add_argument('--data.n_queries', type=int, default=15)

    parser.add_argument('--data.encoder_name', type=str, default='cnn')
    parser.add_argument('--data.max_length', type=int, default=128)
    
    ## model args
    parser.add_argument('--model.base', type=str)
    parser.add_argument('--model.base_feat', type=str)
    parser.add_argument('--model.path_pretrained', type=str)
    parser.add_argument('--model.feat_dim', type=int)

    parser.add_argument("--model.features_generator", type=str, nargs="+", default=None)
    parser.add_argument("--model.use_mpn_features", type=bool, default=True)
    parser.add_argument("--model.use_mol_features", type=bool, default=True)
    parser.add_argument("--model.mpn_hidden_size", type=int, default=256)
    parser.add_argument("--model.mol_features_size", type=int, default=200)
    parser.add_argument("--model.ffnn_hidden_size", type=int, default=256)
    parser.add_argument("--model.num_ffnn_layers", type=int, default=3)
    parser.add_argument("--model.dropout", type=float, default=0.1)
    parser.add_argument("--model.mpn_depth", type=int, default=2)
    parser.add_argument("--model.undirected_mpn", type=bool, default=False)
    parser.add_argument("--model.enc_hidden_size", type=int, default=16)
        
    ## predset model args
    # parser.add_argument('--model_predset.eps', type=float, default=0.01)
    # parser.add_argument('--model_predset.delta', type=float, default=1e-5)
    # parser.add_argument('--model_predset.n', type=int)

    ## train args
    parser.add_argument('--train.rerun', action='store_true')
    parser.add_argument('--train.load_final', action='store_true')
    parser.add_argument('--train.resume', type=str)
    parser.add_argument('--train.method', type=str, default='src')
    parser.add_argument('--train.optimizer', type=str, default='SGD')
    parser.add_argument('--train.n_epochs', type=int)
    parser.add_argument('--train.lr', type=float, default=0.1) 
    parser.add_argument('--train.momentum', type=float, default=0.9)
    parser.add_argument('--train.weight_decay', type=float)
    parser.add_argument('--train.lr_decay_epoch', type=int)
    parser.add_argument('--train.lr_decay_rate', type=float)
    parser.add_argument('--train.val_period', type=int, default=1)

    # ## adaptation args
    # parser.add_argument('--adap.rerun', action='store_true')
    # parser.add_argument('--adap.load_final', action='store_true')
    # parser.add_argument('--adap.resume', type=str)
    # parser.add_argument('--adap.optimizer', type=str, default='SGD')
    # parser.add_argument('--adap.n_epochs', type=int, default=50)
    # parser.add_argument('--adap.lr', type=float, default=0.0001) 
    # parser.add_argument('--adap.momentum', type=float, default=0.9)
    # parser.add_argument('--adap.weight_decay', type=float, default=0.0)
    # parser.add_argument('--adap.lr_decay_epoch', type=int, default=10)
    # parser.add_argument('--adap.lr_decay_rate', type=float, default=0.5)
    # parser.add_argument('--adap.val_period', type=int, default=1)

    ## uncertainty estimation args
    parser.add_argument('--train_ps.method', type=str)
    parser.add_argument('--train_ps.rerun', action='store_true')
    parser.add_argument('--train_ps.load_final', action='store_true')
    parser.add_argument('--train_ps.binary_search', action='store_true')
    parser.add_argument('--train_ps.bnd_type', type=str, default='direct')

    parser.add_argument('--train_ps.T_step', type=float, default=1e-7) 
    parser.add_argument('--train_ps.T_end', type=float, default=np.inf)
    parser.add_argument('--train_ps.eps_tol', type=float, default=1.25)

    parser.add_argument('--train_ps.eps', type=float, default=0.01)
    parser.add_argument('--train_ps.delta', type=float, default=1e-5)
    parser.add_argument('--train_ps.alpha', type=float, default=1e-5)    

            
    args = parser.parse_args()
    args = util.to_tree_namespace(args)
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args = util.propagate_args(args, 'device')
    args = util.propagate_args(args, 'exp_name')
    args = util.propagate_args(args, 'snapshot_root')

    ## dataset specific parameters
    if 'CIFAR10' in args.data.src:
        if args.data.n_labels is None:
            args.data.n_labels = 10
        if args.model.base is None:
            args.model.base = 'ResNet50'
        if args.model.base_feat is None:
            args.model.base_feat = 'ResNetFeat'
        if args.model.feat_dim is None:
            args.model.feat_dim = 2048
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        else:
            args.model.pretrained = True

        if args.train.n_epochs is None:
            args.train.n_epochs = 400
        if args.train.weight_decay is None:
            args.train.weight_decay = 5e-4
        if args.train.lr_decay_epoch is None:
            args.train.lr_decay_epoch = 150
        if args.train.lr_decay_rate is None:
            args.train.lr_decay_rate = 0.1
            
            
        # if args.data.n_examples is None:
        #     ## 500-shot classification setup
        #     args.data.n_examples = [None, 5000, None] # [#train, #val, #test]
        # if args.data.n_datasets is None:
        #     args.data.n_datasets = [5, 5, 1] # [#train, #val, #test]

        if args.data.n_examples is None:
            ## 100-shot classification setup
            args.data.n_examples = [5000, 5000, 5000] # [#train, #val, #test]; note that currently I don't use val here
        if args.data.n_datasets is None:
            args.data.n_datasets = [100, 100, 100, 10] # [#train, #val, #cal, #test]
        if args.model_predset.n is None:
            args.model_predset.n = args.data.n_examples[0] * args.data.n_datasets[2]
        else:
            assert(args.model_predset.n <= args.data.n_val_src)
            
    elif 'MiniImageNet' in args.data.src:
        if args.train.n_epochs is None:
            args.train.n_epochs = 200
        if args.train.weight_decay is None:
            args.train.weight_decay = 0.0
        if args.train.lr_decay_epoch is None:
            args.train.lr_decay_epoch = 40
        if args.train.lr_decay_rate is None:
            args.train.lr_decay_rate = 0.5

    elif 'ImageNet' in args.data.src:
        if args.data.n_labels is None:
            args.data.n_labels = 1000
        if args.model.base is None:
            args.model.base = 'ResNet101'
        if args.model.base_feat is None:
            args.model.base_feat = 'ResNetFeat'
        if args.model.feat_dim is None:
            args.model.feat_dim = 2048
        if args.model.path_pretrained is None:
            args.model.path_pretrained = 'pytorch'
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        else:
            args.model.pretrained = True

        if args.train.n_epochs is None:
            warnings.warn('seeting the hyperparameters carefully')
            args.train.n_epochs = 60
        if args.train.weight_decay is None:
            args.train.weight_decay = 0.0
        if args.train.lr_decay_epoch is None:
            args.train.lr_decay_epoch = 20
        if args.train.lr_decay_rate is None:
            args.train.lr_decay_rate = 0.5


        raise NotImplementedError
        if args.data.n_examples is None:
            ## 5-shot classification setup
            args.data.n_examples = [5000, 5000, 5000] # [#train, #val, #test]
        if args.data.n_datasets is None:
            args.data.n_datasets = [100, 100, 1] # [#train, #val, #test]
            
        if args.model_predset.n is None:
            args.model_predset.n = args.data.n_examples[1] * args.data.n_datasets[1]
        else:
            assert(args.model_predset.n <= args.data.n_val_src)
            
    elif 'FewRel' in args.data.src:
        if args.train.n_epochs is None:
            args.train.n_epochs = 200
        if args.train.weight_decay is None:
            args.train.weight_decay = 0.0
        if args.train.lr_decay_epoch is None:
            args.train.lr_decay_epoch = 40
        if args.train.lr_decay_rate is None:
            args.train.lr_decay_rate = 0.5

    elif 'Diabetes' in args.data.src:
        if args.train.n_epochs is None:
            args.train.n_epochs = 200
        if args.train.weight_decay is None:
            args.train.weight_decay = 0.0
        if args.train.lr_decay_epoch is None:
            args.train.lr_decay_epoch = 40
        if args.train.lr_decay_rate is None:
            args.train.lr_decay_rate = 0.5
        
        args.data.dim = [385]
        args.data.n_labels = 2
        args.data.n_ways = 2 #TODO: redundant

        
    elif 'Heart' in args.data.src:
        if args.train.n_epochs is None:
            args.train.n_epochs = 200
        if args.train.weight_decay is None:
            args.train.weight_decay = 0.0
        if args.train.lr_decay_epoch is None:
            args.train.lr_decay_epoch = 40
        if args.train.lr_decay_rate is None:
            args.train.lr_decay_rate = 0.5
        
        args.data.dim = [54]
        args.data.n_labels = 2
        args.data.n_ways = 2 #TODO: redundant
        args.data.n_datasets_train = 200
        
    elif 'Chembl' in args.data.src:
        if args.train.n_epochs is None:
            args.train.n_epochs = 200
        if args.train.weight_decay is None:
            args.train.weight_decay = 0.0
        if args.train.lr_decay_epoch is None:
            args.train.lr_decay_epoch = 40
        if args.train.lr_decay_rate is None:
            args.train.lr_decay_rate = 0.5
            
        args.data.n_datasets_train = 800

    else:
        raise NotImplementedError    
        
    ## setup logger
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))    

    ## print args
    util.print_args(args)
    
    return args    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)


