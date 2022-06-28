"""

reg_lambda is so big, about 400, works fine. WHY?

----------------

Those MF methods (based on SVD (or closed form equation))

are no need do negative sampling. 

"""


import numpy as np 

import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utils.DataLoader as DL 

import utils.Evaluation as EVA

from tqdm import trange

class RASE:
    def __init__(self, size_topN=10): 
        self.UI_matrix = 0
        self.Item_feat = 0
        self.topN_list = []
        self.topN = size_topN

    def fit(self, X, reg_lambda):
        self.UI_matrix = X 
        G = np.matmul(X.T, X)
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += reg_lambda 
        P = np.linalg.inv(G) 
        self.Item_feat = P / (- np.diag(P)) 
        self.Item_feat[diag_indices] = 0 


    def show_diag_rmse(self, diag_idx):
        print( 'rmse of diags: ',np.dot( self.Item_feat[diag_idx] , self.Item_feat[diag_idx] ) )


    def show_train_rmse(self):
        train_errs = self.UI_matrix - np.matmul(self.UI_matrix, self.Item_feat)
        rmse = np.sqrt(np.mean(train_errs * train_errs))
        print("training RMSE = {}".format(rmse))
        mae = np.sum(np.abs( train_errs )) / len(np.where(self.UI_matrix > 0)[0])
        print('training MAE = {}'.format(mae))

    def predict(self, a_test_item):
        u, i, r = a_test_item 
        if r:
            r = 1 
        hat_ui = np.dot( self.UI_matrix[u,:] , self.Item_feat[:,i] )
        return hat_ui, r - hat_ui

    def batch_predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.

        Args:
        pairs: A pair of lists (users, items) of the same length.
        batch_size: unused.
        verbose: unused.

        Returns:
        predictions: A list of the same length as users and items, such that
        predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        for i in range(num_examples):
            predictions[i], _ = self.predict([pairs[0][i], pairs[1][i], 0])
        return predictions


    def multi_predict(self, subset_test):
        hats, errors = [], [] 
        for t in subset_test:
            hat, err = self.predict(t)
            hats.append(hat)
            errors.append(err) 
        return hats, errors 
    
    def get_W(self):
        return self.Item_feat 

    # def _single_recom_topN(self, u, topK=10):
    #     size_item = len(self.UI_matrix[0])
    #     hat_dict = dict()
    #     hats = self.UI_matrix[u,:] @ self.Item_feat
    #     for can_i in range(size_item):
    #         if self.UI_matrix[u, can_i] == 0:
    #             hat_dict[can_i] = hats[i] 
    #     topK_hat_r = dict(sorted(hat_dict.items(), key=lambda item: item[1], reverse=True)[:topK])
    #     topK_items = list(topK_hat_r.keys())
    #     return topK_items 


    def _single_recom_topN(self, u, topK=10):
        size_item = self.UI_matrix.shape[1]
        items_purchased = np.where(self.UI_matrix[u] > 0)[0]
        hat_rs    = self.UI_matrix[u,:] @ self.Item_feat 
        hats_record = [(key_i, hat_rs[key_i]) for key_i in range(size_item) if key_i not in items_purchased]
        hats_record.sort(key = lambda x : x[1], reverse = True)
        hats_record = hats_record[:topK]
        topK_items = [x[0] for x in hats_record]
        return topK_items 

    def topN_Recom(self):
        size_u = self.UI_matrix.shape[0]
        self.topN_list = []
        for u in trange(size_u):
            self.topN_list.append(self._single_recom_topN(u, self.topN))

    def get_topN_recom(self):
        return self.topN_list


if __name__ == "__main__":

#------------------------- Data PrePare -----------------------#

    # data_path = r'/Users/jeff/OneDrive/Code_bank/Learn/RS/MovieLens/RS_XiangLiang/movielens_1m/ml-1m/'
    # data_record, traingset, testingset, user_set, item_set = DL.data_loader(
    #     file_path=data_path+'ratings.dat', 
    #     item_path=data_path+'movies.dat', 
    #     ratio=0.8, 
    #     sep='::',
    #     value_form='binary',
    # )

    ml100k_folder = r'/Users/jeff/OneDrive/Code_bank/Learn/RS_dataset/MovieLens/ml-100k/'

    data_record, trainingset, testset, num_users, num_items = DL.data_loader_ml100k(
        data_dir=ml100k_folder, 
        ratio=0.8, 
        value_form='implicit',
        )

    X = DL.convert2matrix(trainingset, num_users, num_items) 
    GT = DL.test_set2ground_truth(testset, num_users)


#------------------------- Fit the model -----------------------#

    rase = RASE(size_topN=50)
    rase.fit(X=X, reg_lambda=400)
    # rase.show_train_rmse()
    # weight = rase.get_W()
    rase.topN_Recom()
    all_topN = rase.get_topN_recom()
    # EVA.evaluate_model_full(GT=GT, AllTopN=all_topN)



    N = [5, 10, 50]
    for n in N:
        EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)
