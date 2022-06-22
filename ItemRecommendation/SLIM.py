"""
SLIM is a sparse item recommendation algorithms. 

It has a L1-reg & L2-reg is its target function. 

--------------

L1      - Lasso 
L2      - ridge
L1 & L2 - ElasticNet 

--------------

Given X, we wanna model SLIM as 
\hat{X} = X @ W 
where X is the UI-interaction matrix, 
and W is a sparse weight matrix (shows similarity of items maybe?)

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
- maybe look further about Multi-task Elastic-Net 
- At present, we calculate W[:,j] one by one. 

===========================

Now, need tune the parameters ( l1 & l2 reg )

results are not so good. 

"""


import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import datasets.DataLoader as DL
import utils.Evaluation as EVA
from sklearn.linear_model import ElasticNet
from tqdm import trange

import pandas as pd 

class SLIM:
    def __init__(self, num_users, num_items, size_topN, reg_alpha, l1_ratio, positive=True,fit_intercept=False, copy_X=False, max_iter=1000, tol=1e-4):
        self.UI_matrix = np.zeros(shape=(num_users, num_items))   ## matrix of U-I interaction
        self.W = np.zeros(shape=(num_items, num_items))           ## matrix of weight 
        self.topN_list = []
        self.szN = size_topN 
        self.reg_alpha = reg_alpha  ## equals l1 + l2
        self.l1_ratio  = l1_ratio   ## l1 / (l1 + l2)
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.model = ElasticNet(
            alpha=self.reg_alpha, 
            l1_ratio=self.l1_ratio,
            positive=self.positive,            ## for SLIM, this holds True
            fit_intercept=self.fit_intercept,  ## for SLIM, ~~no data is prepared to be centered~~ we have no bias
            copy_X=self.copy_X, 
            max_iter=self.max_iter,
            tol=self.tol,
            # precompute=True, 
            selection='random',
            )

    def fit(self, X):
        self.UI_matrix = X 
        num_items = self.UI_matrix.shape[1]
        print('training stage:\n')
        for item_idx in trange(num_items):        
            self.model.fit(self.UI_matrix, self.UI_matrix[:, item_idx].copy())
            self.W[:,item_idx] = self.model.coef_

    def _single_recom_topN(self, u, topK=10):
        size_item = self.UI_matrix.shape[1]
        items_purchased = np.where(self.UI_matrix[u] > 0)[0]
        hat_rs    = self.UI_matrix[u,:] @ self.W  
        hats_record = [(key_i, hat_rs[key_i]) for key_i in range(size_item) if key_i not in items_purchased]
        hats_record.sort(key = lambda x : x[1], reverse = True)
        hats_record = hats_record[:topK]
        topK_items = [x[0] for x in hats_record]
        return topK_items 

    def topN_Recom(self):
        size_u = self.UI_matrix.shape[0]
        self.topN_list = []
        print('predict stage:\n')
        for u in trange(size_u):
            self.topN_list.append(self._single_recom_topN(u, self.szN))
    
    def get_topN_recom(self):
        return self.topN_list



if __name__ == "__main__":

    ""

    # ml100k_folder = r'/Users/jeff/OneDrive/Code_bank/Learn/RS_dataset/MovieLens/ml-100k/'

    # data_record, trainingset, testset, num_users, num_items = DL.data_loader_ml100k(
    #     data_dir=ml100k_folder, 
    #     ratio=0.8, 
    #     value_form='implicit',
    #     )

    # X = DL.convert2matrix(trainingset, num_users, num_items) 
    # GT = DL.test_set2ground_truth(testset, num_users)

    # l1_reg, l2_reg = 0.3, 0.1 

    # slim = SLIM(
    #     num_users=num_users, 
    #     num_items=num_items, 
    #     size_topN=50, 
    #     reg_alpha=0.2,#l1_reg + l2_reg,
    #     l1_ratio=0.02,#l1_reg / (l1_reg + l2_reg),
    #     max_iter=1000,
    #     tol= 0.1, #1e-4,
    #     )

    # slim.fit(X)
    # slim.topN_Recom()
    # all_topN = slim.get_topN_recom()

    # N = [5, 10, 50]
    # for n in N:
    #     EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)

#----------------------------
    """
    read ml-10m 
    """


    ml10m_folder = r'/Users/jeff/OneDrive/Code_bank/Learn/RS_dataset/MovieLens/ml-10m/'
    ml10m = ml10m_folder + 'ratings.dat'
    data = pd.read_csv(
        ml10m, 
        sep='::', 
        engine='python', 
        header=None, 
        names=['user_id', 'item_id', 'rating', 'time_stamp'],
        )   

    num_users = data['user_id'].unique().shape[0]
    num_items = data['item_id'].unique().shape[0]


    ## unique item
    unique_item_idx = data['item_id'].sort_values().unique()
    ori2newidx = {}
    idx = 0
    for uidx in unique_item_idx:
        ori2newidx[uidx] = idx 
        idx += 1
    data['item_idx'] = data['item_id'].map(ori2newidx)

    # ## user_idx -= 1
    unique_user_idx = data['user_id'].sort_values().unique()
    ori2newidx = {}
    idx = 0
    for uidx in unique_user_idx:
        ori2newidx[uidx] = idx 
        idx += 1
    data['user_idx'] = data['user_id'].map(ori2newidx)


    grouped = data.groupby('user_id')
    last_time = grouped['time_stamp'].transform('max')

    training_flag = data['time_stamp'] != last_time
    training_flag = training_flag.to_numpy()

    training_set = data[training_flag]
    test_set     = data[~training_flag]

    training_set = training_set[['user_idx', 'item_idx', 'rating']].to_numpy()
    test_set     = test_set[['user_idx', 'item_idx', 'rating']].to_numpy()

    training_set = training_set.astype('int')
    test_set     = test_set.astype('int')


    """
    ml10m_inter = ml10m_folder + 'ml-10m.inter' 


    data = pd.read_csv(ml10m_inter, sep='\t')   

    num_users = data['user_id:token'].unique().shape[0]
    num_items = data['item_id:token'].unique().shape[0]

    grouped = data.groupby('user_id:token')
    last_time = grouped['timestamp:float'].transform('max')

    training_flag = data['timestamp:float'] != last_time
    training_flag = training_flag.to_numpy()

    training_set = data[training_flag]
    test_set     = data[~training_flag]

    training_set = training_set[['user_id:token', 'item_id:token', 'rating:float']].to_numpy()
    test_set     = test_set[['user_id:token', 'item_id:token', 'rating:float']].to_numpy()

    training_set = training_set.astype('int')
    test_set     = test_set.astype('int')
    """

    X = DL.convert2matrix(training_set, num_users, num_items) 
    X = np.sign(X) ## convert into implicit feedback 
    GT = DL.test_set2ground_truth(test_set, num_users)

    # ml10m_folder
    with open(ml10m_folder + 'implicit_UI_matrix.npy', 'wb') as f:
        np.save(f, X)
    

    slim = SLIM(
        num_users=num_users, 
        num_items=num_items, 
        size_topN=50, 
        reg_alpha=0.2,  #l1_reg + l2_reg,
        l1_ratio=0.02,  #l1_reg / (l1_reg + l2_reg),
        max_iter=1000,
        tol= 0.1,  #1e-4,
        )

    slim.fit(X)
    slim.topN_Recom()
    all_topN = slim.get_topN_recom()

    N = [5, 10, 50]
    for n in N:
        EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)





    