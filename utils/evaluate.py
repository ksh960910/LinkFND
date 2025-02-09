from .metrics import *
from .parser import parse_args

import random
import torch
import math
import numpy as np
import multiprocessing
import heapq
from time import time

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag




def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    idx = []
    for i in K_max_item_score:
        # score가 높다고 판단한 item이 test set에 있다면 1, 없다면 0
        if i in user_pos_test:
            r.append(1)
            idx.append(i)
        else:
            r.append(0)
            idx.append(0)
    auc = 0.
    # idx는 총 K_max개의 원소를 가지고있고 relevant한 item의 idx가, relevant하지않다면 0으로 구성돼있음

    return r, auc, idx


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, idx, auc, Ks):
    precision, recall, p, n, u, ndcg, hit_ratio = [], [], [], [], [], [], []
    x, y, z = [], [], []
    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        # print('rrrrrrrrrrrrrrr : ', r)
        # print('iiiiiiiiiii : ', idx)
        # print('aaaaaaaaaaaaaa : ', a, len(a), b, len(b), c, len(c))
        # print('aaaaaaaaaa : ', sorted(a), len(a))
        # print(xxxxxxx)
        for i in idx:
            if i!=0:
                if i in a:
                    x.append(1)
                    y.append(0)
                    z.append(0)
                elif i in b:
                    y.append(1)
                    x.append(0)
                    z.append(0)
                # elif i in c:
                else:
                    z.append(1)
                    x.append(0)
                    y.append(0)
            else:
                x.append(0)
                y.append(0)
                z.append(0)
        p.append(recall_at_k(x, K, len(user_pos_test)))
        n.append(recall_at_k(y, K, len(user_pos_test)))
        u.append(recall_at_k(z, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'popular' : np.array(p), 'normal' : np.array(n), 'unpopular' : np.array(u), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []

    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        # r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
        r, auc, idx = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, idx, auc, Ks)


def test(model, user_dict, n_params, popular, normal, unpopular, mode='test'):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'popular' : np.zeros(len(Ks)),
              'normal' : np.zeros(len(Ks)),
              'unpopular' : np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}
    global a,b,c
    a= popular
    b= normal
    c= unpopular

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    else:
        # test_user_set = user_dict['test_user_set']
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    user_gcn_emb, item_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start:end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = item_gcn_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start:i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = item_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['popular'] += re['popular']/n_test_users
            result['normal'] += re['normal']/n_test_users
            result['unpopular'] += re['unpopular']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

    assert count == n_test_users
    pool.close()
    return result