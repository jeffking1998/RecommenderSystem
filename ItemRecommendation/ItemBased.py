import numpy as np 

import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utils.VectorSimilarity as VS 
import utils.DataLoader as DL 
from tqdm import trange

import utils.Evaluation as EVA

from sklearn.metrics import pairwise_distances


class ItemBasedRegression:
    def __init__(self, similarity_method='cosine', topK=10):
        self.similarity_method = similarity_method
        self.topK = topK
        self.X = 0 
        self.invert_file = 0 
        self.global_mean = 0 
        self.sim_matrix = 0

    def cal_vector_weight(self, va, vb):
        if self.similarity_method == 'cosine':
            return VS.cosine(va, vb) 
        elif self.similarity_method == 'pearson':
            return VS.pearson(va, vb)
        elif self.similarity_method == 'adjust_cosine':
            return VS.adjust_cosine(va, vb)
        else:
            raise ModuleNotFoundError(" Unknown Method for Similarity. ")

    def cal_global_mean(self):
        return np.sum(self.X) / (self.X > 0).sum() 

    def fit(self, X, invert_file):
        self.X = X 
        self.invert_file = invert_file 
        self.global_mean = self.cal_global_mean()
        self.sim_matrix = 1 - pairwise_distances(X.T, metric=self.similarity_method)

    def predict(self, a_test_item):
        u, i, r = a_test_item 

        u_Is = self.invert_file[u] ## U has brought de items && similar with i
        N_i   = dict() ## KNN of i
        for j in u_Is:
            sim_ij = self.sim_matrix[i, j]  #self.cal_vector_weight(self.X[:,i], self.X[:,j]) 
            N_i[j] = sim_ij 
        KNN_i = dict(sorted(N_i.items(), key=lambda item: item[1], reverse=True)[:self.topK])



        upper, lower = 0, 0

        for j, sim_ij in KNN_i.items():
            upper += sim_ij * self.X[u, j]
            lower += abs(sim_ij) 
        hat_ui = upper / (lower + 1e-9)

        if len(KNN_i) == 0:
            hat_ui = self.global_mean
        
        return hat_ui, hat_ui - r 

    def multi_predict(self, subset_of_test):
        hats, errors = [], [] 
        for t in subset_of_test:
            hat, err = self.predict(t) 
            hats.append(hat) 
            errors.append(err)
        return hats, errors 

    def _single_recom_topN(self, u):
        hat_r = dict()
        size_item = len(self.X[0])
        for can_i in range(size_item):
            if self.X[u, can_i] == 0:
                hat_rui, _ = self.predict(a_test_item=[u, can_i, 0])
                hat_r[can_i] = hat_rui 
        # print(hat_r)
        topK_hat_r = dict(sorted(hat_r.items(), key=lambda item: item[1], reverse=True)[:self.topK])
        # print(topK_hat_r)
        topK_items = list(topK_hat_r.keys())
        # print(topK_items)
        return topK_items 
    
    def topN_Recom(self):
        size_u = self.X.shape[0]
        self.topN_list = []
        for u in trange(size_u):
            self.topN_list.append(self._single_recom_topN(u))
 
    def get_topN_recom(self):
        return self.topN_list



class ItemBasedClassification(ItemBasedRegression):
    def __init__(self, similarity_method='cosine', topK=10, N_class=5):
        self.similarity_method = similarity_method
        self.topK = topK
        self.X = 0 
        self.invert_file = 0 
        self.global_mean = 0 
        self.N_class = N_class 

    def predict(self, a_test_item):
        u, i, r = a_test_item 
        class_dict = dict()
        for _idx in range(1, 1+self.N_class):
            class_dict[_idx] = 0 

        u_Is = self.invert_file[u] ## U has brought de items && similar with i
        N_i   = dict() ## KNN of i
        for j in u_Is:
            sim_ij = self.cal_vector_weight(self.X[:,i], self.X[:,j]) 
            N_i[j] = sim_ij 
        KNN_i = dict(sorted(N_i.items(), key=lambda item: item[1], reverse=True)[:self.topK])

        for j, sim_ij in KNN_i.items():
            r_uj = self.X[u,j]
            if r_uj in class_dict.keys():
                class_dict[r_uj] += sim_ij 
            else:
                raise ValueError("r_uj not in N_class.")
        hat_rui = max(class_dict, key=class_dict.get) 

        return hat_rui, hat_rui - r 
    







if __name__ == "__main__":

    #------------------------- Data PrePare -----------------------#
    ## data prepare
    ml100k_folder = r'/Users/jeff/OneDrive/Code_bank/Learn/RS_dataset/MovieLens/ml-100k/'

    data_record, trainingset, testset, num_users, num_items = DL.data_loader_ml100k(
        data_dir=ml100k_folder, 
        # ratio=0.8, 
        value_form='implicit',
        split_mode='random-one-out',
        )

    X = DL.convert2matrix(trainingset, num_users, num_items) 
    GT = DL.test_set2ground_truth(testset, num_users)

    invert_file = DL.convert2invert_file(trainingset, num_users, num_items, main_key=1) 

    #------------------------- ItemBased Regression -----------------------#
    
    item_reg_cf = ItemBasedRegression(topK=50)
    item_reg_cf.fit(X, invert_file)


    item_reg_cf.topN_Recom()
    all_topN = item_reg_cf.get_topN_recom()
    # EVA.evaluate_model_full(GT=GT, AllTopN=all_topN)


    hr = EVA.hit_rate(all_GT=GT, all_topN=all_topN, K=10)

    arhr = EVA.ARHR(all_GT=GT, all_topN=all_topN, K=10)

    print('for leave-rand-one-out, hr = {}, arhr = {}'.format(hr, arhr))

###########

    data_record, trainingset, testset, num_users, num_items = DL.data_loader_ml100k(
        data_dir=ml100k_folder, 
        ratio=0.8, 
        value_form='implicit',
        split_mode='random',
        )

    X = DL.convert2matrix(trainingset, num_users, num_items) 
    GT = DL.test_set2ground_truth(testset, num_users)

    invert_file = DL.convert2invert_file(trainingset, num_users, num_items, main_key=1) 

    #------------------------- ItemBased Regression -----------------------#
    
    item_reg_cf = ItemBasedRegression(topK=50)
    item_reg_cf.fit(X, invert_file)


    item_reg_cf.topN_Recom()
    all_topN = item_reg_cf.get_topN_recom()


    N = [5, 10, 50]
    for n in N:
        EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)




    #     #------------------------- A Sample -----------------------#
    # a_test = testset[0] 
    # hat, err = item_reg_cf.predict(a_test) 
    # print(a_test, hat, err) 

    #     #------------------------- Batch -----------------------#
    
    # subset_of_test = testingset[:100]
    # hats, errors = item_reg_cf.multi_predict(subset_of_test)
    # errors = np.array(errors)
    # print('rmse: ', np.sqrt( np.dot(errors, errors)  /  len(errors) ))



    # #------------------------- ItemBased Classification -----------------------#

    # item_cla_cf = ItemBasedClassification(N_class=5)
    # item_cla_cf.fit(X, invert_file)


    #     #------------------------- A Sample -----------------------#
    # a_test = testingset[0] 
    # hat, err = item_cla_cf.predict(a_test) 
    # print(a_test, hat, err) 

    #     #------------------------- Batch -----------------------#
    
    # subset_of_test = testingset[:100]
    # hats, errors = item_cla_cf.multi_predict(subset_of_test)
    # errors = np.array(errors)
    # print('rmse: ', np.sqrt( np.dot(errors, errors)  /  len(errors) ))


