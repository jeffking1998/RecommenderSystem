import numpy as np 

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import datasets.DataLoader as DL 
import utils.VectorSimilarity as VS 


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

    X = DL.convert2matrix(traingset, len(user_set), len(item_set)) 

########### Test 

    one_row = np.ones(shape=(1, X.shape[1]))
    X_norms = np.reshape(np.linalg.norm(X, axis=1), newshape=(X.shape[0], 1)) @ one_row
    X = np.true_divide(X, X_norms)

    a_j = 338

    for j in range(1000):
        my_fun(X, j)

"""
Output exceeds the size limit. Open the full output data in a text editor
0.927718442668249
0.898120897049902
0.8920211911880553
0.8578855095303515
0.8573128609150368
0.9117727179840329
0.8755395746835257
0.8460860541722335
0.8278683089922291
0.8917060579917263
0.902053406681854
0.8467440294956964
0.8445281849675381
0.8983615243302214
0.8561386845787609
0.9177599883214276
0.902414441047353
0.8933431031304314
0.8597063639391073
0.8316370969334538
0.9265890939382053
0.8886914925006822
0.8668544679597043
0.9130377239344254
0.923130562070113
"""

def my_fun(X, a_j):
    X_j = X[:,a_j] 
    X_j_g_1 = np.where(X_j > 0)[0] 
    X_j_e_0 = np.where(X_j == 0)[0]

    pt_1 = X[X_j_g_1[4]]
    pt_1 = pt_1 / np.linalg.norm(pt_1)
    pt_2 = X[X_j_g_1[44]]
    pt_2 = pt_2 / np.linalg.norm(pt_2)
    VS.cosine(va=pt_1, vb=pt_2)

    ng_1 = X[X_j_e_0[14]]
    VS.cosine(va=pt_1, vb=ng_1)

    pt_X_mean = np.mean(X[X_j_g_1], axis=0)
    ng_X_mean = np.mean(X[X_j_e_0], axis=0)

    dis_pos_ng = VS.cosine(va=pt_X_mean, vb=ng_X_mean)
    print(dis_pos_ng)
