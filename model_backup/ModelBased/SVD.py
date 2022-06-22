"""
When two params update in each other. 
There need old_  new_ prefix. 

Nevertheless, the meaning from equation is more explicit

-------------Old & Wrong update way de speed
@ epoch:  10  rmse =  0.8806504731948984
@ epoch:  20  rmse =  0.8359395978489732
@ epoch:  30  rmse =  0.8090917931906175
@ epoch:  40  rmse =  0.793442287903865
@ epoch:  50  rmse =  0.783562435828534
@ epoch:  60  rmse =  0.7768669466603115
@ epoch:  70  rmse =  0.7720737872110323
@ epoch:  80  rmse =  0.7684923259027764
@ epoch:  90  rmse =  0.7657243692903951
@ epoch:  100  rmse =  0.7635268928300173


--------------New & Right way to update de speed 
@ epoch:  10  rmse =  0.8823605585538176
@ epoch:  20  rmse =  0.8369897836303213
@ epoch:  30  rmse =  0.8095561519125347
@ epoch:  40  rmse =  0.7934371791316734
@ epoch:  50  rmse =  0.7833397860604281
@ epoch:  60  rmse =  0.7765765641002884
@ epoch:  70  rmse =  0.771788290137149
@ epoch:  80  rmse =  0.7682480940419474
@ epoch:  90  rmse =  0.7655395231593294
@ epoch:  100  rmse =  0.7634093350080167


It seems that correct way is slower. Reason is simply. let say p and q need to be updated.
If q updates after p. Then q has a better param in q's updating equation, and has a step nearer to optimal result.
But from math, we need to follow the right way.
BUT in the end, right way works better.

"""


import numpy as np 

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import datasets.DataLoader as DL 


class SVDCF:
    def __init__(self, n_users, n_items, n_features = 10, alpha=0.005, lambda_reg=0.02, max_iter=1e3):
        self.training_data = 0  # TriTable 
        self.mu =            0  #np.mean(TriTable[:,-1]) ## avg for entire matrix
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_feat = np.random.rand(n_users,  n_features)
        self.item_feat = np.random.rand(n_items, n_features)
        self.old_user_feat = np.random.rand(n_users,  n_features)
        self.old_item_feat = np.random.rand(n_items, n_features)
        self.alpha = alpha 
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter

    def sgd(self):
        for epoch_i in range(1, 1+int(self.max_iter)):
            rmse = 0
            for u, i, r in self.training_data:
                hat_rate_ui = self.mu + self.item_bias[i] + self.user_bias[u] + np.dot(self.old_user_feat[u], self.old_item_feat[i])
                err = r - hat_rate_ui # Determined by Loss Function
                rmse += err ** 2
                ## update paras by SGD 
                self.user_bias[u] += self.alpha * (err - self.lambda_reg * self.user_bias[u])
                self.item_bias[i] += self.alpha * (err - self.lambda_reg * self.item_bias[i])
                self.user_feat[u] += self.alpha * (err * self.old_item_feat[i] - self.lambda_reg * self.old_user_feat[u])
                self.item_feat[i] += self.alpha * (err * self.old_user_feat[u] - self.lambda_reg * self.old_item_feat[i])

                self.old_user_feat[u], self.old_item_feat[i] = self.user_feat[u], self.item_feat[i]

            rmse /= len(self.training_data)
            rmse = np.sqrt(rmse)
            if epoch_i % int(self.max_iter / 10) == 0:
                print('@ epoch: ', epoch_i, ' rmse = ', rmse)

    def fit(self, TrainTriTable):
        self.training_data = TrainTriTable
        self.mu = np.mean(TrainTriTable[:,-1])
        self.sgd() 

    def predict(self, a_test_item):
        u, i, r = a_test_item 
        hat_ui = self.mu + self.item_bias[i] + self.user_bias[u] + np.dot( self.user_feat[u], self.item_feat[i] )
        return hat_ui, r - hat_ui
    
    def multi_predict(self, subset_test):
        hats, errors = [], [] 
        for t in subset_test:
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
        topK_hat_r = dict(sorted(hat_r.items(), key=lambda item: item[1], reverse=True)[:self.topK])
        topK_items = list(topK_hat_r.keys())
        return topK_items 
    
    def topN_Recom(self):
        size_u = self.X.shape[0]
        self.topN_list = []
        for u in range(size_u):
            self.topN_list.append(self._single_recom_topN(u))



class SVDPlusPlus(SVDCF):
    def __init__(self, n_users, n_items, n_features=10, alpha=0.007, lambda_for_bias=0.005, lambda_for_feat=0.015, max_iter=1e3):
        self.training_data = 0  # TriTable 
        self.mu =            0  #np.mean(TriTable[:,-1]) ## avg for entire matrix
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_feat = np.random.rand(n_users,  n_features)
        self.item_feat = np.random.rand(n_items, n_features)
        self.old_user_feat = np.random.rand(n_users,  n_features)
        self.old_item_feat = np.random.rand(n_items, n_features)
        self.alpha = alpha 
        self.lambda_for_bias = lambda_for_bias
        self.lambda_for_feat = lambda_for_feat
        self.item_implicit_feat = np.random.rand(n_items, n_features) 
        self.old_item_implicit_feat = np.random.rand(n_items, n_features) 
        self.invert_file = 0 ## here is u:items, so main_key=1
        self.max_iter = max_iter
    

    def sgd(self):
        for epoch_i in range(1, 1+int(self.max_iter)):
            rmse = 0
            for u, i, r in self.training_data:

                u_Is = self.invert_file[u]
                feat_u = self.old_user_feat[u] + (np.sum(self.old_item_implicit_feat[ u_Is ], axis=0) * (len(u_Is) ** -0.5))
                ## BUG here: not divide but multi. 
                hat_rate_ui = self.mu + self.item_bias[i] + self.user_bias[u] + np.dot(self.old_item_feat[i], feat_u )
                err = r - hat_rate_ui # Determined by Loss Function
                rmse += err ** 2 
                ## update paras by SGD 
                self.user_bias[u] += self.alpha * (err - self.lambda_for_bias * self.user_bias[u])
                self.item_bias[i] += self.alpha * (err - self.lambda_for_bias * self.item_bias[i])
                self.user_feat[u] += self.alpha * (err * self.old_item_feat[i] - self.lambda_for_feat * self.old_user_feat[u])
                self.item_feat[i] += self.alpha * (err * feat_u - self.lambda_for_feat * self.old_item_feat[i])
                for j_idx in u_Is:
                    self.item_implicit_feat[j_idx] += self.alpha * ( err * (len(u_Is) ** -0.5) * self.old_item_feat[i] - self.lambda_for_feat * self.old_item_implicit_feat[j_idx] )

                ## transfer new to old. 
                self.old_user_feat[u], self.old_item_feat[i] = self.user_feat[u], self.item_feat[i]
                for j_idx in u_Is:
                    self.old_item_implicit_feat[j_idx] = self.item_implicit_feat[j_idx]

            rmse /= len(self.training_data)
            rmse = np.sqrt(rmse)
            if epoch_i % int(self.max_iter / 10) == 0:
                print('@ epoch: ', epoch_i, ' rmse = ', rmse)

    def fit(self, TrainTriTable, invert_file): 
        self.training_data = TrainTriTable
        self.mu = np.mean(TrainTriTable[:,-1])
        self.invert_file = invert_file 
        self.sgd()








if __name__ == "__main__":

#------------------------- Data PrePare -----------------------#
    


    data_path = r'/Users/jeff/OneDrive/Code_bank/Learn/RS/MovieLens/RS_XiangLiang/movielens_1m/ml-1m/'
    data_record, traingset, testingset, user_set, item_set = DL.data_loader(
        file_path=data_path+'ratings.dat', 
        item_path=data_path+'movies.dat', 
        ratio=0.8, 
        sep='::',
        value_form='binary',
        ) 

    invert_file = DL.convert2invert_file(traingset, user_set, item_set, main_key=1) 


    X = DL.convert2matrix(traingset, len(user_set), len(item_set)) 

##--------------------------------- SVD -------------------------------##

    #------------------------- Fit SVD Model -----------------------#
    
    svd = SVDCF(n_users=len(user_set), n_items=len(item_set), max_iter=100)
    svd.fit(np.array(traingset))

    #------------------------- One Sample -----------------------#
    
    a_test_item = testingset[0] ##wrong 0.5707396392968 # right:0.6843288646536259
    # a_test_item = [1495, 3016, 4] ##wrong 0.8472066182771254 right 0.8241771718327384 ## this item's invert_file is empty
    hat, err = svd.predict(a_test_item)
    print(a_test_item, hat, err)

    #------------------------- Batch -----------------------#
    subset_test = testingset[:100]
    hats, errors = svd.multi_predict(subset_test)
    errors = np.array(errors)
    print('rmse: ', np.sqrt( np.dot(errors, errors) / len(errors) ))
    ## wrong : 0.9550143536046075 right:0.8920455670599706
    

##--------------------------------- SVD++ -------------------------------##


    #------------------------- Fit SVD++  -----------------------#

    svd_pp = SVDPlusPlus(n_users=len(user_set), n_items=len(item_set), max_iter=10)
    svd_pp.fit(TrainTriTable=np.array(traingset), invert_file=invert_file)

# ep: 1 u & i: 4 - 3509  err:  7.351777831518468e+307  rmse  inf

    #------------------------- One Sample -----------------------#
    
    a_test_item = testingset[0] 
    # a_test_item = [1495, 3016, 4] ## this item's invert_file is empty
    hat, err = svd_pp.predict(a_test_item)
    print(a_test_item, hat, err)

    #------------------------- Batch -----------------------#
    
    subset_test = testingset[:100]
    hats, errors = svd_pp.multi_predict(subset_test)
    errors = np.array(errors)
    print('rmse: ', np.sqrt( np.dot(errors, errors) / len(errors) ))
    

