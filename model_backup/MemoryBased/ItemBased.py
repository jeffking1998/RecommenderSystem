import numpy as np 

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utils.VectorSimilarity as VS 
import datasets.DataLoader as DL 



class ItemBasedRegression:
    def __init__(self, similarity_method='cosine', topK=10):
        self.similarity_method = similarity_method
        self.topK = topK
        self.X = 0 
        self.invert_file = 0 
        self.global_mean = 0 

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

    def predict(self, a_test_item):
        u, i, r = a_test_item 

        u_Is = self.invert_file[u] ## U has brought de items && similar with i
        N_i   = dict() ## KNN of i
        for j in u_Is:
            sim_ij = self.cal_vector_weight(self.X[:,i], self.X[:,j]) 
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
                hat_rui, _ = self.predict(item_of_test=[u, can_i, 0])
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
        for u in range(size_u):
            self.topN_list.append(self._single_recom_topN(u))
 

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
    data_path = '/Users/jeff/OneDrive/Code_bank/Learn/RS/MovieLens/RS_XiangLiang/movielens_1m/ml-1m/'
    data_record, traingset, testingset, user_set, item_set = DL.data_loader(file_path=data_path+'ratings.dat', item_path=data_path+'movies.dat', ratio=0.8, sep='::') 
    X = DL.convert2matrix(traingset, user_set, item_set)
    invert_file = DL.convert2invert_file(traingset, user_set, item_set, main_key=1) 

    #------------------------- ItemBased Regression -----------------------#
    
    item_reg_cf = ItemBasedRegression()
    item_reg_cf.fit(X, invert_file)

        #------------------------- A Sample -----------------------#
    a_test = testingset[0] 
    hat, err = item_reg_cf.predict(a_test) 
    print(a_test, hat, err) 

        #------------------------- Batch -----------------------#
    
    subset_of_test = testingset[:100]
    hats, errors = item_reg_cf.multi_predict(subset_of_test)
    errors = np.array(errors)
    print('rmse: ', np.sqrt( np.dot(errors, errors)  /  len(errors) ))



    #------------------------- ItemBased Classification -----------------------#

    item_cla_cf = ItemBasedClassification(N_class=5)
    item_cla_cf.fit(X, invert_file)


        #------------------------- A Sample -----------------------#
    a_test = testingset[0] 
    hat, err = item_cla_cf.predict(a_test) 
    print(a_test, hat, err) 

        #------------------------- Batch -----------------------#
    
    subset_of_test = testingset[:100]
    hats, errors = item_cla_cf.multi_predict(subset_of_test)
    errors = np.array(errors)
    print('rmse: ', np.sqrt( np.dot(errors, errors)  /  len(errors) ))


