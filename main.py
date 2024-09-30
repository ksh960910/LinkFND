import os
import random

import torch
import numpy as np
import gc

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter



n_users = 0
n_items = 0



def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n):
    def sampling(user_item, train_set, n=1):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                negitem = random.choice(range(n_items))
                while negitem in train_set[user]:
                    negitem = random.choice(range(n_items))
                neg_items.append(negitem)
        return neg_items
    
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set)).to(device)
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''empty cache'''
    torch.cuda.empty_cache()

    """read args"""
    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat, popular, normal, unpopular = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    from modules.LinkFND import LinkFND
    from modules.SimGCL import SimGCL
    from modules.LightGCN import LightGCN
    from modules.NGCF import NGCF
    if args.gnn == 'lightgcn':
        print('MixGCF-LightGCN model setup')
        model = LightGCN(n_params, args, norm_mat).to(device)
    elif args.gnn == 'simgcl':
        print('SimGCL model setup')
        model = SimGCL(n_params, args, norm_mat).to(device)
    elif args.gnn == 'linkfnd':
        print('SimGCL + LinkFND model setup')
        print('Support view 1개와 view1,view2 각각의 sim score의 max값')
        model = LinkFND(n_params, args, norm_mat).to(device)
    else:
        model = NGCF(n_params, args, norm_mat).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    anchor = 285 #69
    t = time()

    print("start training ...")

    writer = SummaryWriter('scalar/'+args.gnn+"_"+args.dataset)

    for epoch in range(args.epoch):
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        fn_idxs = []
        
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            batch_loss, user_emb, item_emb, fn_idx = model(anchor, batch)
            if len(fn_idx)>0 and len(fn_idxs)<=0:
                fn_idxs.append(fn_idx)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            s += args.batch_size

            writer.add_scalar("Loss/"+args.gnn+"_"+args.dataset+"_eps"+str(args.eps)+"_lamb"+str(args.lamb)+"_fnk"+str(args.fnk)+"/train", loss, epoch)

        train_e_t = time()

        if epoch % 5 == 0:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "popular", "normal","unpopular", "ndcg", "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, popular, normal, unpopular, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['popular'], test_ret['normal'], test_ret['unpopular'],
                 test_ret['ndcg'], test_ret['precision'], test_ret['hit_ratio']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, popular, normal, unpopular, mode='valid')
                # valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['popular'], valid_ret['normal'], valid_ret['unpopular'],
                     valid_ret['ndcg'], valid_ret['precision'], valid_ret['hit_ratio']])
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=2)
            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')

            """save embedding"""
            if os.path.exists('embedding/'):
                if valid_ret['recall'][0] == cur_best_pre_0 and args.emb_save:
                    user_emb = user_emb.detach().cpu().numpy()
                    item_emb = item_emb.detach().cpu().numpy()
                    np.save('embedding/'+args.gnn+'_'+args.dataset+'_eps'+str(args.eps)+'_lamb'+str(args.lamb)+'_user.npy', user_emb)
                    np.save('embedding/'+args.gnn+'_'+args.dataset+'_eps'+str(args.eps)+'_lamb'+str(args.lamb)+'_item.npy', item_emb)
            else:
                os.mkdir('embedding')

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    writer.close()
    print('TensorBoard close')
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))