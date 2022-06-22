"""
::In the deep model, we firstly map the user_id & item_id to their embeddings.
There are three types of NeuCF models. 


::The origin version from NeuCF's author was implemented in TF 1.x.
And I modified it to use TensorFlow 2.x. 


::About negative sampling. In NeuCF:
    - training & testset are the same as the previous setting 
    - negative samples are selected 99 for one gt in testset
        - so every time, we feed into the model with a user + 100 items(1postive + 99 negaitve)
        - check whether the topK can hit the one real positive  

::Jeff
    - I really dont need follow his experiment's full settings. 
    - I only need run the model & get the prediction(topN recommendations) & evaluate it.    

"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Helper libraries
import matplotlib.pyplot as plt
# TensorFlow and tf.keras
import tensorflow as tf
import datasets.DataLoader as DL
import utils.Evaluation as EVA
from sklearn.linear_model import ElasticNet
from tensorflow import keras
from tqdm import trange

print(tf.__version__)
from keras import regularizers
from keras.layers import (Concatenate, Dense, Embedding, Flatten, Input,
                          Multiply)
from keras.models import Model



"""
class of Python:
    - in __init__ can call function that implemented later.
    - instances in __init__ can be called out of class. 
"""
# class Dataset(object):
#     '''
#     classdocs
#     '''

#     def __init__(self, path):
#         '''
#         Constructor
#         '''
#         self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
#         self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
#         self.testNegatives = self.load_negative_file(path + ".test.negative")
#         assert len(self.testRatings) == len(self.testNegatives)
        
#         self.num_users, self.num_items = self.trainMatrix.shape
        
#     def load_rating_file_as_list(self, filename):
#         ratingList = []
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line != None and line != "":
#                 arr = line.split("\t")
#                 user, item = int(arr[0]), int(arr[1])
#                 ratingList.append([user, item])
#                 line = f.readline()
#         return ratingList
    
#     def load_negative_file(self, filename):
#         negativeList = []
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line != None and line != "":
#                 arr = line.split("\t")
#                 negatives = []
#                 for x in arr[1: ]:
#                     negatives.append(int(x))
#                 negativeList.append(negatives)
#                 line = f.readline()
#         return negativeList
    
#     def load_rating_file_as_matrix(self, filename):
#         '''
#         Read .rating file and Return dok matrix.
#         The first line of .rating file is: num_users\t num_items
#         '''
#         # Get number of users and items
#         num_users, num_items = 0, 0
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line != None and line != "":
#                 arr = line.split("\t")
#                 u, i = int(arr[0]), int(arr[1])
#                 num_users = max(num_users, u)
#                 num_items = max(num_items, i)
#                 line = f.readline()
#         # Construct matrix
#         mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line != None and line != "":
#                 arr = line.split("\t")
#                 user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
#                 if (rating > 0):
#                     mat[user, item] = 1.0
#                 line = f.readline()    
#         return mat



def get_model(num_users, num_items, latent_dim=10, reg_mf=0, layers=[20,10], reg_ls=[0,0] ):
    assert len(layers) == len(reg_ls)
    num_layers = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input') 
    item_input = Input(shape=(1,), dtype='int32', name='item_input')


    mf_user_Embedding = Embedding(input_dim=num_users, output_dim=latent_dim, embeddings_initializer='uniform',
                               embeddings_regularizer=regularizers.l2(reg_mf),
                               input_length=1,
                               )

    mf_item_Embedding = Embedding(input_dim=num_items, output_dim=latent_dim,
                               embeddings_initializer='uniform',
                               embeddings_regularizer=regularizers.l2(reg_mf),
                               input_length=1,
                               )

    mf_user_latent = Flatten()(mf_user_Embedding(user_input))
    mf_item_latent = Flatten()(mf_item_Embedding(item_input)) 
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    mlp_user_Embedding = Embedding(input_dim=num_users, 
                               output_dim=int(layers[0]/2), 
                               embeddings_initializer='uniform',
                               embeddings_regularizer=regularizers.l2(reg_ls[0]),
                               input_length=1,
                               )
    mlp_item_Embedding = Embedding(input_dim=num_items, 
                               output_dim=int(layers[0]/2),
                               embeddings_initializer='uniform',
                               embeddings_regularizer=regularizers.l2(reg_ls[0]),
                               input_length=1,
                               )

    mlp_user_latent = Flatten()(mlp_user_Embedding(user_input))
    mlp_item_latent = Flatten()(mlp_item_Embedding(item_input)) ##why (class's params)(args)

    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

    print(type(mlp_vector))
    print(mlp_vector.shape)
    for idx in range(1, num_layers):
        mlp_vector = Dense(units=layers[idx],
                       activation='relu',
                       kernel_regularizer=regularizers.l2(reg_ls[idx]),
                       name='layer{}'.format(idx),
                       )(mlp_vector)

    merged_vector = Concatenate()([mf_vector, mlp_vector])

    prediction = Dense(units=1,
                       activation='sigmoid',
                       kernel_initializer='lecun_uniform',
                       name = 'prediction', 
                       )(merged_vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model 


# def evaluate_model(model, UI_matrix, GT, szN=10):

#     def _single_recom_topN(model, UI_matrix, u, topK=10):
#         size_item = UI_matrix.shape[1]
#         items_not_purchased = np.where(UI_matrix[u] == 0)[0]
#         user_test_input = np.ones_like(items_not_purchased) * u 
#         pred = model.predict(x=[user_test_input, items_not_purchased], batch_size=1)
#         pred = pred[:,0]
#         hats_record = [(item, i_pred) for item,i_pred in zip(items_not_purchased, pred)]
#         hats_record.sort(key = lambda x : x[1], reverse = True)
#         hats_record = hats_record[:topK]
#         topK_items = [x[0] for x in hats_record]
#         return topK_items 

#     def topN_Recom(UI_matrix, szN):
#         size_u = UI_matrix.shape[0]
#         topN_list = []
#         print('predict stage:\n')
#         for u in trange(size_u):
#             topN_list.append(_single_recom_topN(model, UI_matrix, u, szN))
#         return topN_list
    
#     all_topN = topN_Recom(UI_matrix, szN)
#     EVA.evaluate_model_full(GT=GT, AllTopN=all_topN)


def evaluate_model(model, UI_matrix, GT, szN=50):

    def _single_recom_topN(model, UI_matrix, u, topK):
        size_item = UI_matrix.shape[1]
        items_not_purchased = np.where(UI_matrix[u] == 0)[0]
        user_test_input = np.ones_like(items_not_purchased) * u 
        pred = model.predict(x=[user_test_input, items_not_purchased], batch_size=1)
        pred = pred[:,0]
        hats_record = [(item, i_pred) for item,i_pred in zip(items_not_purchased, pred)]
        hats_record.sort(key = lambda x : x[1], reverse = True)
        hats_record = hats_record[:topK]
        topK_items = [x[0] for x in hats_record]
        return topK_items 

    def topN_Recom(UI_matrix, szN):
        size_u = UI_matrix.shape[0]
        topN_list = []
        print('predict stage:\n')
        for u in trange(size_u):
            topN_list.append(_single_recom_topN(model, UI_matrix, u, szN))
        return topN_list
    
    all_topN = topN_Recom(UI_matrix, szN)
    # EVA.evaluate_model_full(GT=GT, AllTopN=all_topN)
    N = [5, 10, 50]
    for n in N:
        EVA.full_evaluate_At_N(GT=GT, AllTopN=np.array(all_topN), N = n)


def get_train_instances(train, num_negatives):
    # 1 VS num_neg :: positive VS negative
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    nonzero_idx = train.nonzero()
    nonzero_idx_loop = [
        (u,i) for u, i in zip(nonzero_idx[0], nonzero_idx[1])
    ]
    for i in trange(len(nonzero_idx_loop)):
        u, i = nonzero_idx_loop[i]
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in nonzero_idx_loop:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels





if __name__ == "__main__":

    ml100k_folder = r'/Users/jeff/OneDrive/Code_bank/Learn/RS_dataset/MovieLens/ml-100k/'

    data_record, trainingset, testset, num_users, num_items = DL.data_loader_ml100k(
        data_dir=ml100k_folder, 
        ratio=0.8, 
        value_form='implicit',
        )

    X = DL.convert2matrix(trainingset, num_users, num_items) 
    GT = DL.test_set2ground_truth(testset, num_users)


    # dataset = Dataset(mvlens_dir + 'ml-1m')
    # train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    train = X 

    num_users, num_items = train.shape

    alpha = 0.001

    model = get_model(num_users=num_users, num_items=num_items, latent_dim=8, reg_mf=0, layers=[64,32,16,8], reg_ls=[0,0,0,0])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha), loss="binary_crossentropy")




    user_input, item_input, labels = get_train_instances(train, num_negatives=1)        

#--------------------------------------------------------
    mvlens_dir = r'/Users/jeff/OneDrive/Code_bank/Learn/RS/model_ckpt/NeucF/'
    checkpoint_path = mvlens_dir + "NeuCF_on_cpu_macmini2.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    batch_size = 2048
    num_epochs = 40

    topK = 10
    evaluation_threads = 1 
    
    for ep in range(num_epochs):
        # Train the model with the new callback
        history = model.fit(x=[np.array(user_input), np.array(item_input)], 
                        y=np.array(labels),  
                        epochs=1,
                        batch_size=batch_size, verbose=0, shuffle=True,
                        #   validation_data=(test_images, test_labels),
                        callbacks=[cp_callback]
                        )  # Pass callback to training


        # hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        loss = history.history['loss'][0]
        print('iter = {}, loss = {}'.format(ep+1, loss))

    evaluate_model(model, X, GT, szN=50)

# 100%|██████████| 943/943 [1:19:05<00:00,  5.03s/it]mean_precision = 0.19777306468716788 
# mean_recall = 0.12464972025383221 
# mean_avg_precision = 0.11968747169540624   
# mean_ndcg = 0.22888223298882227 
