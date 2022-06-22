import numpy as np 
import pandas as pd 
from sklearn.neighbors import NearestNeighbors


"""
0. two hyder-parameters 
   * `K`-NN
   * top-N recommendation items

1. suppose that data is clean
   * the user and the item both need to be clean ans starting with 0
2. start the model
   * one question: ranking scores ? enheng

3. since i don't understand the `training_set` and `testing_sets`'s meanings in
   this algorithm. 
   -- I cancelled this partion.

-------------------------------
TODO:
    code is ok: cause the similarity de result are zeros

"""

class CF():
    def __init__(self, input_data, n_neighbor, n_topK_items):
        ## input variables
        self.data = input_data
        self.n_neighbor = n_neighbor 
        self.n_topK = n_topK_items
        ## working variables
        self.nearest_neighbor = None
        self.topK_items = [ [] for _ in range(len(self.data)) ] 
        self.recommendation_list = [ [] for _ in range(len(self.data)) ]
        # self.training_data = None 
        # self.testing_data = None

    def cal_topK(self):
        ## step0: cal knn
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbor,
                                metric='cosine',
                                algorithm='auto',
                                n_jobs=-1, # full processers work on it
                                ).fit(self.data)
        self.nearest_neighbor = nbrs.kneighbors(self.data, return_distance=False)
        
        ## step1: merge knns' items and rank by freqence
        for i in range(len(self.data)):
            item_freq = np.mean(self.data[self.nearest_neighbor[i],:], axis=0 )
            # print(item_freq.shape)
            # print(max(item_freq))
            ind = np.argsort(item_freq, axis=0)[::-1]
            self.topK_items[i] = ind[:self.n_topK] #TODO:what if there common items is less than `n_topK`
            # print(item_freq[ind[:5]])
        ## using built-in data variables , there should be no need to return 


    def make_recommendation(self):
        for i in range(len(self.data)): # user 
            for itm_idx in self.topK_items[i]: # item-candidate
                # print(i, itm_idx, self.data[i, itm_idx])
                if abs(self.data[i, itm_idx]) == 0:
                    self.recommendation_list[i].append(itm_idx)
                    print('user ', i, 'add a recom item: ', itm_idx)

    def save_recommendation_list(self):
        tmp_list = []
        for rcd in self.recommendation_list:
            tmp = [str(x) for x in rcd]
            tmp_list.append(','.join(tmp))

        with open('recommendation_list.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(tmp_list))

    def fit(self):
        self.cal_topK()
        self.make_recommendation()
        self.save_recommendation_list()

    def predict(self):
        pass 





def preprocess_dataset(data_url):
    # this convert datafile to user-item matrix
    cleaned_dir = data_url + '/cleaned_dataset/'
    ## step0: build a mapping map
    item_map = {}
    user_map = {}

    video_info = pd.read_csv(data_url + '/movies.csv')
    movie_items = video_info.movieId.to_numpy()
    for i, item in enumerate(movie_items):
        item_map[item] = i 

    rate_record = pd.read_csv(data_url + '/ratings.csv')
    user_idx = set(rate_record.userId.to_numpy())
    for i, user in enumerate(user_idx):
        user_map[user] = i
    
    ## step0.5: use mapping to clean data
    for row in range(len(rate_record)):
        rate_record.loc[row, 'userId'] = user_map.get(int(rate_record.loc[row, 'userId']))
        rate_record.loc[row, 'movieId'] = item_map.get(int(rate_record.loc[row, 'movieId']))
    rate_record.to_csv(cleaned_dir + 'cleaned_ratings.csv', index=None)

    ## step1: clean the `rating.csv`
    ratings = pd.read_csv(cleaned_dir + 'cleaned_ratings.csv')
    pivoted_rating = ratings.pivot(index="userId", columns="movieId", values="rating")
    pivoted_rating = pivoted_rating.fillna(0)
    # save
    pivoted_rating.to_csv(cleaned_dir + 'pivoted_cleaned_ratings.csv',header=None, index=None)


if __name__ == "__main__":
    data_url = './dataset/ml-latest-small/'
    # preprocess_dataset(data_url=data_url) ## done
    cleaned_dir = data_url + '/cleaned_dataset/'
    pivoted_rating = pd.read_csv(cleaned_dir + 'pivoted_cleaned_ratings.csv', sep=',',header=None)
    data = pivoted_rating.values
    cf = CF(data, 20, 10)
    cf.fit()

