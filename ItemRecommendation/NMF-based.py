
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

##---
import scipy.sparse as sp
import multiprocessing
import heapq
import math 
from time import time
# import utils.Evaluation as EVA
from tqdm import trange
import utils.DataLoader as DL
#------------------------------
from PureSVD import PureSVD 
from SLIM import SLIM
from EASE_R import RASE
from ItemBased import ItemBasedRegression
from MF import SVDCF 

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.batch_predict([users, np.array(items)], 
                                batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)


if __name__ == "__main__":

    ml1m_folder = r'/Users/jeff/Library/CloudStorage/OneDrive-个人/Code_bank/Learn/RS_dataset/MovieLens/ml-1m/xiangnanhe/'


    print('data folder = ' + ml1m_folder)
    dataset = Dataset(ml1m_folder + 'ml-1m')
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    X = train.toarray()
    GT = DL.test_set2ground_truth2(testRatings, num_users)


#--------------------------- PureSVD
    svd = PureSVD(size_user=num_users, size_item=num_items, latent_factors=50, size_topN=50,random_seed=0)
    svd.fit(X)
    # svd.topN_Recom()
    # all_topN = svd.get_topN_recom()


    K = 1
    num_thread = 1
    hr, ndcg = evaluate_model(svd, testRatings, testNegatives, K, num_thread)
    print(np.array(hr).mean(), np.array(ndcg).mean())

    # N = [5, 10, 15, 20, 50]
    # for n in N:
    #     EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)


#--------------------------- SLIM about 30 minutes

    slim = SLIM(
        num_users=num_users, 
        num_items=num_items, 
        model='sgd',
        size_topN=50, 
        reg_alpha=0.2,
        l1_ratio=0.02,
        max_iter=1000,
        tol= 0.1, #1e-4,
        )

    slim.fit(X)

    # with open('slim_for_NeuCF_1m.npy', 'wb') as f:
    #     np.save(f, slim.W)

    # with open('test.npy', 'rb') as f:
    #     a = np.load(f)

    K = 10
    num_thread = 1
    hr, ndcg = evaluate_model(slim, testRatings, testNegatives, K, num_thread)
    print(np.array(hr).mean(), np.array(ndcg).mean())


    # slim.topN_Recom()
    # all_topN = slim.get_topN_recom()
    # N = [5, 10, 50]
    # for n in N:
    #     EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)


#-----------------EASE R

    rase = RASE(size_topN=50)
    rase.fit(X=X, reg_lambda=1700)
    K = 1
    num_thread = 1
    hr, ndcg = evaluate_model(rase, testRatings, testNegatives, K, num_thread)
    print(np.array(hr).mean(), np.array(ndcg).mean())


##-------------IKNN

    item_reg_cf = ItemBasedRegression(topK=50)
    item_reg_cf.fit(X, invert_file)



    K = 10
    num_thread = 1
    hr, ndcg = evaluate_model(item_reg_cf, testRatings, testNegatives, K, num_thread)
    print(np.array(hr).mean(), np.array(ndcg).mean())


#-----------------------MF  ## takes about23 minutes
    traingset = []
    for uidx in range(num_users):
        for iidx in range(num_items):
            if (uidx, iidx) in train:
                traingset.append([uidx, iidx, 1])
    invert_file = DL.convert2invert_file(traingset, num_users, num_items, main_key=1) 

    svd = SVDCF(n_users=num_users, n_items=num_items, max_iter=100)
    svd.fit(np.array(traingset))

    # with open('svd_u_bias_for_NeuCF_1m.npy', 'wb') as f:
    #     np.save(f, svd.user_bias)
    # with open('svd_i_bias_for_NeuCF_1m.npy', 'wb') as f:
    #     np.save(f, svd.item_bias)
    # with open('svd_u_feat_for_NeuCF_1m.npy', 'wb') as f:
    #     np.save(f, svd.user_feat)
    # with open('svd_i_feat_for_NeuCF_1m.npy', 'wb') as f:
    #     np.save(f, svd.item_feat)


    K = 10
    num_thread = 1
    hr, ndcg = evaluate_model(svd, testRatings, testNegatives, K, num_thread)
    print(np.array(hr).mean(), np.array(ndcg).mean())

