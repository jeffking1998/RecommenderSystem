import numpy as np 


## predict rating: rmse & mAP 

def rmse(error_vector):
    error_vector = np.array(error_vector)
    return np.sqrt( np.mean( error_vector * error_vector ) )

#-------------------------------------------------------------------


## item recommendation: precsion, recall, auc; 
##                      HR, ARHR, NDCG
"""
R : Relative items == Ground Truth 
n : size of topN recommendations
r : gt's rank (in topN) for gt in GT 
All topN items are not include purchased items 

Those methods evaluate a pair<GT, topN> for one user each time.
- It's possibile that not all users are has GT in test set.
- So, maybe leave k out for each user is a nice choice. 
- We'll talk later about negative samplings. 
"""

# def evaluate_model_full(GT, AllTopN, N=[5,10,50]):
#     cnt = 0 
#     mean_precision, mean_avg_precision = 0, 0
#     mean_recall = [
#         0 for x in N
#     ] 
#     mean_ndcg, mean_auc = 0, 0
#     assert len(GT) == len(AllTopN)
#     for gt, topN in zip(GT, AllTopN):
#         if len(gt) == 0:
#             continue
#         cnt += 1 
#         mean_precision     += Precision(GT=gt, top_N=topN)
#         # mean_recall        += Recall(GT=gt, top_N=topN) 
#         for i, n in enumerate(N):
#             mean_recall[i] += Recall_at_N(GT=gt, top_N=topN, N=n)
#         mean_avg_precision += AveragePrecision(GT=gt, top_N=topN)
#         mean_ndcg          += NDCG(GT=gt, top_N=topN)
#         # mean_auc           += AUC(GT=gt, top_N=topN)
#     mean_precision     /= cnt 
#     for i in range(len(N)):
#         mean_recall[i] /= cnt
#     mean_avg_precision /= cnt 
#     mean_ndcg          /= cnt 
#     # mean_auc           /= cnt 
#     # print("mean_precision = {} \nmean_recall = {} \nmean_avg_precision = {} \nmean_ndcg = {} \nmean_auc = {}".format(mean_precision, mean_recall, mean_avg_precision, mean_ndcg, mean_auc)) 


#     print("mean_precision = {} \nmean_recall = {} \nmean_avg_precision = {} \nmean_ndcg = {} \n".format(mean_precision, mean_recall, mean_avg_precision, mean_ndcg)) 

def evaluate_model_full(GT, AllTopN):
    cnt = 0 
    mean_precision, mean_recall, mean_avg_precision = 0, 0, 0
    mean_ndcg, mean_auc = 0, 0
    assert len(GT) == len(AllTopN)
    for gt, topN in zip(GT, AllTopN):
        if len(gt) == 0:
            continue
        cnt += 1 
        mean_precision     += Precision(GT=gt, top_N=topN)
        mean_recall        += Recall(GT=gt, top_N=topN) 
        mean_avg_precision += AveragePrecision(GT=gt, top_N=topN)
        mean_ndcg          += NDCG(GT=gt, top_N=topN)
        # mean_auc           += AUC(GT=gt, top_N=topN)
    mean_precision     /= cnt 
    mean_recall        /= cnt 
    mean_avg_precision /= cnt 
    mean_ndcg          /= cnt 
    # mean_auc           /= cnt 
    # print("mean_precision = {} \nmean_recall = {} \nmean_avg_precision = {} \nmean_ndcg = {} \nmean_auc = {}".format(mean_precision, mean_recall, mean_avg_precision, mean_ndcg, mean_auc)) 
    print("mean_precision = {:.4} \nmean_recall = {:.4} \nmean_avg_precision = {:.4} \nmean_ndcg = {:.4} \n".format(mean_precision, mean_recall, mean_avg_precision, mean_ndcg)) 


def full_evaluate_At_N(GT, AllTopN, N=5):
    print("N = {}".format(N))
    evaluate_model_full(GT=GT, AllTopN=AllTopN[:,:N])


def Precision(GT, top_N):
    size_n = len(top_N)
    sz_tp = 0          ## size of true_positive
    for gt in GT:
        if gt in top_N:
            sz_tp += 1
    return sz_tp / size_n 

def Recall(GT, top_N):
    size_gt = len(GT)
    sz_tp = 0         ## size of true_positive
    for gt in GT:
        if gt in top_N:
            sz_tp += 1
    return sz_tp / size_gt 

def Recall_at_N(GT, top_N, N):
    return Recall(GT=GT[:N], top_N=top_N[:N])

def AveragePrecision(GT, top_N):
    size_n, size_gt = len(top_N), len(GT)
    sum_prec = 0
    for idx in range(size_n):
        if top_N[idx] in GT:
            sum_prec += Precision(GT, top_N[:idx+1])
    min_len = min(size_gt, size_n)
    return sum_prec / min_len 

def NDCG(GT, top_N):
    size_n, size_gt = len(top_N), len(GT)

    gain = 0
    for idx in range(size_n):
        if top_N[idx] in GT:
            gain += 1 / np.log2(idx + 1 + 1)

    denominator = 0 
    min_len = min(size_gt, size_n)
    for i in range(1, 1 + min_len):
        denominator += 1 / np.log2(i + 1)
    return gain / denominator 

def AUC(GT, top_N):
    numerator, denominator = 0, 0
    size_n, size_gt = len(top_N), len(GT)
    sum_rank = 0 
    for idx, item in enumerate(top_N):
        i_rank = idx + 1
        if item in GT:
            sum_rank += i_rank 
    avg_rank = sum_rank / size_gt 
    numerator = size_n - 0.5 * ( size_gt - 1 ) - avg_rank 
    denominator = size_n - size_gt 
    return numerator / denominator 





#-------------------------------------------------------------------

## evaluation metrics for leave-one-out methods 
"""
This two evaluate whole dataset at once.
"""

def hit_rate(all_GT, all_topN, K=10):
    size_user = len(all_GT)
    assert size_user == len(all_topN)
    hit = 0
    assert len(all_GT[0]) == 1
    for gt, topN in zip(all_GT, all_topN):
        if gt[0] in topN[:K]:
            hit += 1
    return hit / size_user 


def ARHR(all_GT, all_topN, K=10):
    size_user = len(all_GT)
    pos_numerator = 0
    size_n = len(all_topN[0])
    # assert size_n == size_user
    for gt, topN in zip(all_GT, all_topN):
        topN = topN[:K]
        if gt[0] in topN:
            for i in range(K):
                if topN[i] == gt[0]:
                    pos_numerator += 1 / (1 + i)
    return pos_numerator / size_user 

# def hit_rate(test_set, top_N):
#     """
#     given a test set, [ ... (u,i,r) ... ]

#     top_N = [
#         ...
#         [ci1, ci2, ..., ciN]
#         ...
#     ]
#     for each (u,i,_) in test_set, 
#         Do check whether i \in u's topN list. 
#     reference:  SLIM
#     """
#     hit, total = 0, 0
#     for u, i, _ in test_set:
#         topN_list = top_N[n] 
#         if i in topN_list:
#             hit += 1
#         total += 1
#     return hit / total 

# def average_reciprocal_hit_rate(test_set, top_N):
#     '''
#     top_N: topN list of all users. 
#     '''
#     total  = len(test_set)
#     N      = len(top_N[0])
#     pos_hr = 0
#     for u, i, _ in test_set:
#         if i in top_N[u]:
#             ## find pos
#             for p in range(N):
#                 if i == top_N[u,p]:
#                     pos_hr += 1 / (p + 1)
#                     break
#     return pos_hr / total 

# ![Recall](https://s2.loli.net/2022/05/24/IqfD3C5AFSNoZLy.png)

# def recall(Gt_u, top_N):
#     """
#     Gt_u :: u's held_out items,
#     top_N :: the topN Recom of u  
#     EASE^R (split in interaction) uses Recall, but didnot give a equation
#     vae_cf gives equation, but split in users.
#     -------------------------------------------------
#     here: we utilize the origional form of Recall := inter / Gt
#     """
#     hit = 0
#     total = len(Gt_u)
#     for item in top_N:
#         if item in Gt_u:
#             hit += 1
#     return hit / total 

# def avg_recall(all_Gt, all_topN):
#     total, cnt = 0, 0
#     for uid, gt in enumerate(all_Gt):
#         if len(gt):
#             rcl = recall(gt, top_N[uid])
#             total += rcl 
#             cnt += 1
#     return total / cnt 

# #https://github.com/CastellanZhang/NDCG
# def DCG(label_list):
#     dcgsum = 0
#     for i in range(len(label_list)):
#         dcg = (2**label_list[i] - 1)/math.log(i+2, 2)
#         dcgsum += dcg
#     return dcgsum

# def NDCG(label_list):
#     global topK
#     dcg = DCG(label_list[0:topK])
#     ideal_list = sorted(label_list, reverse=True)
#     ideal_dcg = DCG(ideal_list[0:topK])
#     if ideal_dcg == 0:
#         return 0
#     return dcg/ideal_dcg

if __name__ == "__main__":
    err = np.array([0.8, -0.9, 0.7,-0.7, 0.8])
    print('rmse = ', rmse(err))
    print('recall', recall([1,5], top_N=[1,2,3,4,5]))
    