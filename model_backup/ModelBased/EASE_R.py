"""

reg_lambda is so big, about 400, works fine. WHY?

"""


import numpy as np 

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import datasets.DataLoader as DL 

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

    ml100k_folder = r'/Users/jeff/Library/CloudStorage/OneDrive-个人/Code_bank/Learn/RS/datasets/ml-100k/'
    data_record, trainingset, testset, num_users, num_items = DL.data_loader_ml100k(
        data_dir=ml100k_folder, 
        ratio=0.8, 
        value_form='implicit',
        )

    X = DL.convert2matrix(trainingset, num_users, num_items) 
    GT = DL.test_set2ground_truth(testset, num_users)


#------------------------- Fit the model -----------------------#

    rase = RASE(size_topN=10)
    rase.fit(X=X, reg_lambda=400)
    # rase.show_train_rmse()
    # weight = rase.get_W()
    rase.topN_Recom()
    all_topN = rase.get_topN_recom()
    EVA.evaluate_model_full(GT=GT, AllTopN=all_topN)




# #------------------------- Predict the model -----------------------#

#     #------------------------- One Sample -----------------------#
    
#     a_test_item = testingset[0] ##wrong 0.5707396392968 # right:0.6843288646536259
#     # a_test_item = [1495, 3016, 4] ##wrong 0.8472066182771254 right 0.8241771718327384 ## this item's invert_file is empty
#     hat, err = rase.predict(a_test_item)
#     print(a_test_item, 'hat = ',hat, 'err = ', err,np.sqrt(err ** 2)) 


#     #------------------------- Batch -----------------------#
#     rand_idx = np.random.randint(low=0,high=199828,size=1000)
#     subset_test = np.array(testingset)[rand_idx]
#     # subset_test = np.array(traingset)[rand_idx]
#     hats, errors = rase.multi_predict(subset_test)
#     errors = np.array(errors)
#     print('mae = ', np.mean(np.abs(errors)))
#     # print('rmse: ', np.sqrt( np.dot(errors, errors) / len(errors) ))
#     ## all in: rmse:  0.9434870691054177

# # -----------------------

# # The result of ease^R is not good enough.

# # a_test = traingset[999]#testingset[3654]

# # test_nbs = np.where( X[:,a_test[1]]  > 0)[0]


# # this_w = weight[:, a_test[1]]


# # for nb in test_nbs:
# #     sim = VS.cosine(X[18], X[nb])
# #     if sim > max_sim:
# #         max_sim = sim 
# #         max_idx = nb 


# # print(X[max_idx]@this_w)
        

# # print(X[a_test[0]] @ this_w)


