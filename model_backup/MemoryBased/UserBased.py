import numpy as np 

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utils.VectorSimilarity as VS 
import datasets.DataLoader as DL 
import utils.Evaluation as EVA 



class UserBasedRegression:
    def __init__(self, similarity_method='cosine', topK=10):
        """
        - Similarity Methods: `cosine`, `pearson`, `adjust_cosine`, `msd`(Mean Squared Difference), `src`(Spearman Rank Correlation)

        """
        self.similarity_method = similarity_method 
        self.topK = topK 
        self.X = 0 
        self.invert_file = 0
        self.global_mean = 0
        self.topN_list  = []

    def cal_global_mean(self):
        return np.sum(X) / (X>0).sum() 
    
    def cal_sim(self, va, vb):
        weight = 0 
        if self.similarity_method == 'cosine':
            weight = VS.cosine(va, vb)
        elif self.similarity_method == 'pearson':
            weight = VS.pearson(va, vb)
        elif self.similarity_method == 'adjust_cosine':
            weight = VS.adjust_cosine(va, vb)
        else:
            pass 

        # if weight == 0:
        #     raise ValueError("Similarity equals zero, it should not happen!")
        return weight 

    def fit(self, X, invert_file):
        self.global_mean = self.cal_global_mean()
        self.X, self.invert_file = X, invert_file 

    def predict(self, item_of_test):
        u, i, r = item_of_test 
        N_u = dict() # knn of u, with k,v = u, similarity, from invert_file 
        i_Us = self.invert_file[i]
        if u in i_Us: ## if build by training-set, u \notin i_Us
            raise ValueError("u should not in i_Us.")
        if self.X[u, i] != 0:
            raise ValueError("X[u,i] should be zero.")
        ## upper two error if happens, it is maual bug. 
                ## lower is dataset's problem. 
        # if len(i_Us) == 0:
        #     raise ValueError("invert file is empty.")
        for v in i_Us:
            sim_uv = self.cal_sim(self.X[u], self.X[v]) 
            N_u[v] = sim_uv
        N_i_of_u = dict(sorted(N_u.items(), key=lambda item: item[1], reverse=True)[:self.topK])


        ## predict 
        upper_part, lower_part = 0, 0 
        for v, w in N_i_of_u.items(): ## the worse thing is if invert-file is empty, the dicts are both empty
            # and upper & lower remains zeros. 
            upper_part += self.X[v, i] * w 
            lower_part += abs(w) 
        hat_rui = upper_part / (lower_part + 1e-9) ## if lower = 0, then upper must = 0
        ## Question is what hat_rui we should assign in this situation. Golbal Mean or zero. 
        if len(i_Us) == 0:
            hat_rui = self.global_mean

        return hat_rui, hat_rui - r

    def multi_predict(self, subset_of_test):
        rating_predictions, errors = [], [] 
        for d in subset_of_test:
            try:
                hat, err = self.predict(d)
                rating_predictions.append(hat) 
                errors.append(err)
            except:
                print("error at {}".format(d))
        return rating_predictions, errors 

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

    def get_topN_recom(self):
        return self.topN_list

class UserBasedClassification(UserBasedRegression):
    def __init__(self, similarity_method='cosine', topK=10, N_class=5):
        """
        - Similarity Methods: `cosine`, `pearson`, `adjust_cosine`, `msd`(Mean Squared Difference), `src`(Spearman Rank Correlation)

        """
        self.similarity_method = similarity_method 
        self.topK = topK 
        self.X = 0 
        self.invert_file = 0
        self.N_class = N_class 

    def predict(self, item_of_test):
        u, i, r = item_of_test  ## i is dangerous
        N_u = dict() # knn of u, with k,v = u, similarity, from invert_file 
        i_Us = self.invert_file[i]
        class_dict = {}
        for _idx in range(1, self.N_class+1):
            class_dict[_idx] = 0 


        if u in i_Us: ## if build by training-set, u \notin i_Us
            raise ValueError("u should not in i_Us.")
        if self.X[u, i] != 0:
            raise ValueError("X[u,i] should be zero.")
        for v in i_Us:
            sim_uv = self.cal_sim(self.X[u], self.X[v]) 
            N_u[v] = sim_uv
        N_i_of_u = dict(sorted(N_u.items(), key=lambda item: item[1], reverse=True)[:self.topK])


        for v, sim_uv in N_i_of_u.items():
            r_vi = self.X[v,i]
            if r_vi in class_dict.keys():
                class_dict[r_vi] += sim_uv 
            else:
                raise ValueError(" r_vi not in N_class. ") 
        hat_rui = max(class_dict, key=class_dict.get) # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        
        if len(i_Us) == 0:
            hat_rui = round( self.global_mean )

        return hat_rui, hat_rui - r 
    





if __name__ == "__main__":

    #------------------------- Data PrePare -----------------------#

    data_path = '/Users/jeff/OneDrive/Code_bank/Learn/RS/MovieLens/RS_XiangLiang/movielens_1m/ml-1m/'
    data_record, traingset, testingset, user_set, item_set = DL.data_loader(
        file_path=data_path+'ratings.dat', 
        item_path=data_path+'movies.dat', 
        ratio=0.8, 
        sep='::',
        value_form='binary',
    )

    X = DL.convert2matrix(traingset, len(user_set), len(item_set))
    invert_file = DL.convert2invert_file(traingset, user_set, item_set) 

    #--------------- Evaluate Dateset ---------------#
    eva = DL.Evaluator()
    eva.evaluate(X=X, test_set=testingset)
    eva.show_result()

    #--------------- UserBased Predict ---------------#

    ## fit the model 
    user_cf = UserBasedRegression() 
    user_cf.fit(X, invert_file)

    ## randomly assign an arbitary testing item 
    a_test = testingset[0]
    a_test = [1495, 3016, 1] ## empty invert file 
                             ## in this situation, similar users are meaningless. 
    hat, err = user_cf.predict(a_test)
    print(a_test, hat, err)

    ## full testing 
    hats, errors = user_cf.multi_predict(testingset[:100])
    errors = np.array(errors)
    print('rmse = ', np.sqrt(np.dot(errors, errors) / len(errors)))

    #--------------- TopN UserBased Predict ---------------#

    # user_cf._single_recom_topN(u=0)

    user_cf.topN_Recom()
    topN = user_cf.get_topN_recom()



    #--------------------------------- UserBased Classification ------------------------------#

    ## fit the model 
    user_class_cf = UserBasedClassification(N_class=5)
    user_class_cf.fit(X, invert_file)

    ## a random case 
    a_test = testingset[0]    # 
    # a_test = [1495, 3016, 4] 
    hat, err = user_class_cf.predict(a_test)
    print(a_test, hat, err)

    ## full testing
    subset_of_test = testingset[:100]
    hats, errors = user_class_cf.multi_predict(subset_of_test)
    errors = np.array(errors) 
    print('rmse : ', np.sqrt(  np.dot( errors , errors ) / len(errors) )  )
