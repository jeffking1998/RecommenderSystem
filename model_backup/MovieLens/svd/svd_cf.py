'''
* Author: Jeff 
* Date: March 24, 2022 
* Description:
        SVD-CF do not make a SVD decomposition actually, 
        but uses l2-least square + SGD to optimize the linear model.
        MODEL: \hat{r_{ui}} = \mu + b_i + b_u + q_i.T @ p_u,
             , where q_is and p_us simulate to U and V in SVD's result.
'''



import numpy as np 
import random 
# import sys
# sys.path.append('/Users/jeff/OneDrive/Code_bank/Learn/RS/MovieLens/RS_XiangLiang/')
# import user_CF

class SVDCF:
    def __init__(self, TriTable, n_users, n_items, n_features = 10, alpha=0.005, lambda_reg = 0.02):
        self.training_data = TriTable 
        self.mu = np.mean(TriTable[:,-1]) ## avg for entire matrix
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_feat = np.random.rand(n_users,  n_features)
        self.item_feat = np.random.rand(n_items, n_features)
        self.alpha = alpha 
        self.lambda_reg = lambda_reg

    def fit(self, max_iter=1e3):
        self.sgd(max_iter=max_iter) 


    def sgd(self, max_iter=1e3):
        for epoch_i in range(int(max_iter)):
            rmse = 0
            for u, i, r in self.training_data:
                
                hat_rate_ui = self.mu + self.item_bias[i] + self.user_bias[u] + np.dot(self.user_feat[u], self.item_feat[i])
 
                err = r - hat_rate_ui 
                rmse += err * err 
                ## update paras by SGD 
                self.user_bias[u] += self.alpha * (err - self.lambda_reg * self.user_bias[u])
                self.item_bias[i] += self.alpha * (err - self.lambda_reg * self.item_bias[i])
                self.user_feat[u] += self.alpha * (err * self.item_feat[i] - self.lambda_reg * self.user_feat[u])
                self.item_feat[i] += self.alpha * (err * self.user_feat[u] - self.lambda_reg * self.item_feat[i])
            rmse /= len(self.training_data)
            # if (epoch_i + 1) % 10 == 0:
            print('@ epoch: ', epoch_i+1, ' rmse = ', rmse)




def movielens_reader(data_path,item_path, training_ratio=1.0, filter_low=False):
    '''
    TODO: 3-24 22:00 night 
    return :: tri-table for training and testing 

    1. movie-idx and user-idx all minus 1 to make index start with zero
    2. n-user and n-item need to read another two files.
    '''
    def clean_item(item_path):
        with open(item_path, 'r', encoding='utf-8') as f:
            data = f.read().strip('\n').split('\n')
        # print(len(data))

        convert_dict = dict()

        for i, d in enumerate(data):
            if(len(d)):
                try:
                    tmp_list = d.split('::')
                    item_key = int(tmp_list[0])
                    convert_dict[item_key] = i 
                except:
                    print(d.split('::'))
        return convert_dict 

    convert_dict = clean_item(item_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read().strip('\n').split('\n')
    data_record = []
    user_set, item_set = set(), set()  
    for d in data:
        if(len(d)):
            userId, movId, score, _ = [int(x) for x in d.split('::')]
            userId -= 1
            movId  = convert_dict.get(movId)
            user_set.add(userId) 
            item_set.add(movId)
            if filter_low:
                if score >= 3:
                    data_record.append([userId, movId, score])
            else:
                data_record.append([userId, movId, score])
    ## spliting the dataset into training and testing:
    random.seed(0)
    traingset, testingset = [], []
    for d in data_record:
        if random.random() <= training_ratio:
            traingset.append(d)
        else:
            testingset.append(d)
    print(len(convert_dict))
    # print(len(user_set), len(item_set))
    return traingset, testingset, len(user_set), print(len(convert_dict))#len(item_set)


if __name__ == "__main__":
    np.random.seed(0) 
    ## load data in tri-table and matrix form
    data_path = '/Users/jeff/OneDrive/Code_bank/Learn/RS/MovieLens/RS_XiangLiang/movielens_1m/ml-1m/'
    
    # clean_item(data_path + 'movies.dat')

    trainingset, testingset, n_user, n_item = movielens_reader(data_path + 'ratings.dat', 
    data_path + 'movies.dat',
    0.8, False)

    ## train
    svd_cf = SVDCF(np.array(trainingset), n_users=6040, n_items=3883)
    svd_cf.fit()
