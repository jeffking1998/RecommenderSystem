import numpy as np 

import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utils.DataLoader as DL 

from sklearn.utils.extmath import randomized_svd

from tqdm import trange

import utils.Evaluation as EVA

class PureSVD:
    def __init__(self, size_user = 0, size_item = 0, latent_factors=20, size_topN=10, random_seed=0):
        self.UI_matrix = np.zeros(shape=(size_user, size_item))
        self.topN = size_topN
        self.latent_factors = latent_factors
        self.random_seed = random_seed 
        self.U_feat = np.zeros(shape=(size_user, latent_factors))
        self.I_feat = np.zeros(shape=(size_item, latent_factors))
        self.topN_list = []

    def fit(self, X):
        self.UI_matrix = X 
        U, Sigma, QT = randomized_svd(self.UI_matrix,
                                      n_components=self.latent_factors,
                                      random_state = self.random_seed)
        self.U_feat = U @ np.diag(Sigma) 
        self.I_feat = QT  ## f \times |I|


    def predict(self, a_test_item):
        u, i, r = a_test_item
        hat_r = self.U_feat[u,:] @ self.I_feat[:,i]
        return hat_r, hat_r - r 

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


    def _single_recom_topN(self, u, topK=10):
        size_item = self.UI_matrix.shape[1]
        items_purchased = np.where(self.UI_matrix[u] > 0)[0]
        hat_rs    = self.U_feat[u,:] @ self.I_feat 
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


class SparsePureSVD(PureSVD):
    def __init__(self):
        pass 

    def fit(self):
        pass 

    def predict(self):
        pass





if __name__ == "__main__":
    """
    Because this methods uses PureSVD without any types of LOSS function.
    So we dont warry about which loss function, and negative samplings.
    --------
    All trainingset & testset are splited from non-zero samples.  
    ========
    TWO tune-able parameters:
    - number of latent factors 
      - it looks larger no imporment but lower for four evaluation. 
    - number of items to recommendate 
      - larger for lower precision
      - larger for higher recall 
    """

    """ ml100k 

    ml100k_folder = r'/Users/jeff/OneDrive/Code_bank/Learn/RS_dataset/MovieLens/ml-100k/'

    data_record, trainingset, testset, num_users, num_items = DL.data_loader_ml100k(
        data_dir=ml100k_folder, 
        ratio=0.8, 
        value_form='implicit',
        )

    X = DL.convert2matrix(trainingset, num_users, num_items) 
    GT = DL.test_set2ground_truth(testset, num_users)


    svd = PureSVD(size_user=num_users, size_item=num_items, latent_factors=20, size_topN=50,random_seed=0)
    svd.fit(X)
    svd.topN_Recom()
    all_topN = svd.get_topN_recom()


    N = [5, 10, 50]
    for n in N:
        EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)
    """
        
#------------------------------------------------------------------
    # TODO: this is for rebuild original paper's results

    ml1m_folder = r'/Users/jeff/OneDrive/Code_bank/Learn/RS_dataset/MovieLens/ml-1m/'

    data_record, trainingset, testset, num_users, num_items = DL.data_loader_ml1m(
        file_path = ml1m_folder + 'ratings.dat', 
        item_path = ml1m_folder + 'movies.dat',
        ratio     = 1 - 0.14,   ## training / total  _OR_  1 - test / total 
        sep       = '::',
        value_form= 'remain',
        )    

    X = DL.convert2matrix(trainingset, num_users, num_items, matrix_form='numpy', value_form='remain') 
    # in original paper, using ratings to train 



    def filter(data, key_idx=0,  min_val=0, max_val=5):
        res = [
            x for x in data if (x[key_idx] >= min_val and x[key_idx] <= max_val)
        ]
        return res 
    testset = filter(data=testset, key_idx=-1, min_val=5)
    # 139966 --> 31698

    GT = DL.test_set2ground_truth(testset, num_users)


    svd = PureSVD(size_user=num_users, size_item=num_items, latent_factors=10, size_topN=50,random_seed=0)
    svd.fit(X)
    svd.topN_Recom()
    all_topN = svd.get_topN_recom()


    N = [5, 10, 15, 20, 50]
    for n in N:
        EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)

        
