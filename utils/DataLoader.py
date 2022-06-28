import random 
import numpy as np 
import pandas as pd
import os 
import scipy.sparse as sp

class Evaluator:
    def __init__(self):
        self.X = 0
        self.test_set = []
        # self.n_users = 0
        # self.n_items = 0 
    def evaluate(self, X, test_set, n_users=0, n_item=0):
        self.X = X 
        self.test_set = test_set 
        # self.n_users = n_users
        # self.n_items = n_items 
    
    def _basic_info(self):
        n_train_set = np.sum(self.X) 
        n_test_set  = len(self.test_set) 
        test_ratio  = n_test_set / (n_test_set + n_train_set)
        info = 'train_set_size = {}, test_set_size = {}, test_ratio = {:.2f}'.format(n_train_set, n_test_set, test_ratio)
        return info 

    def _data_distribution(self):
        """
        For a record -- (u, i, r) in test_set:
        we want to find:
        1. # other people who brought this same item.
        2. # other items that this person has brought.
        from TRAIN_SET. 
        once, those two index <= 1, this record is a poor guy.
        """
        n_col_nb = []
        n_row_nb = []
        for u, i, r in self.test_set:
            n_col_nb.append(self.X[:,i].sum())
            n_row_nb.append(self.X[u,:].sum())
        n_col_less1 = len( [n for n in n_col_nb if n <= 1] )
        n_row_less1 = len( [n for n in n_row_nb if n <= 1] )
        info = '{} columns that no neighbor, {} rows that no neighbor'.format(n_col_less1, n_row_less1)
        return info 

    def show_result(self):
        print(self._basic_info())
        print(self._data_distribution())

def data_loader_ml100k(data_dir, ratio=0.8, value_form='remain', split_mode='random'):
    def read_data_ml100k(data_dir, value_form='remain'):
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                        engine='python')
        ## set userid & itemid start from 0
        data['user_id'] = data['user_id'] - 1
        data['item_id'] = data['item_id'] - 1
        ## check the feedbacks are explict/implicit 
        if value_form != 'remain':
            data['rating'] = np.sign(data['rating'])
        num_users = data.user_id.unique().shape[0]
        num_items = data.item_id.unique().shape[0]
        return data, num_users, num_items
    def split_data_ml100k(data, num_users, num_items,
                        split_mode='random', test_ratio=0.1):
        """Split the dataset in random mode or seq-aware mode."""
        if split_mode == 'seq-aware':
            train_items, test_items, train_list = {}, {}, []
            for line in data.itertuples():
                u, i, rating, time = line[1], line[2], line[3], line[4]
                train_items.setdefault(u, []).append((u, i, rating, time))
                if u not in test_items or test_items[u][-1] < time:
                    test_items[u] = (i, rating, time)
            for u in range(1, num_users + 1):
                train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
            test_data = [(key, *value) for key, value in test_items.items()]
            train_data = [item for item in train_list if item not in test_data]
            train_data = pd.DataFrame(train_data)
            test_data = pd.DataFrame(test_data)
        elif split_mode == 'random-one-out':
            flag = [True for _ in range(data.shape[0])]
            grouped = data.groupby('user_id')
            for u, idxs in grouped.groups.items():
                test_idx = np.random.choice(idxs) 
                flag[test_idx] = False
            neg_flag = [not x for x in flag]
            train_data, test_data = data[flag], data[neg_flag]
        else:
            mask = [True if x == 1 else False for x in np.random.uniform(
                0, 1, (len(data))) < 1 - test_ratio]
            neg_mask = [not x for x in mask]
            train_data, test_data = data[mask], data[neg_mask]
        return train_data, test_data
    data, num_users, num_items = read_data_ml100k(data_dir=data_dir, value_form=value_form)
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode=split_mode, test_ratio=1-ratio)
    sparsity = 1 - len(data) / (num_users * num_items)
    print(f'number of users: {num_users}, number of items: {num_items}')
    print(f'matrix sparsity: {sparsity:f}')

    selected_columns = ['user_id', 'item_id', 'rating']

    data_record = data[selected_columns].to_numpy()
    trainingset = train_data[selected_columns].to_numpy()
    testset     = test_data[selected_columns].to_numpy()
    return data_record, trainingset, testset, num_users, num_items



def data_loader_ml1m(file_path, item_path, ratio=0.8, sep='::', value_form='remain'):
    """
    * value_form = 'remain' or 'binary'
    """

    def clean_item(item_path):
        with open(item_path, 'r', encoding='utf-8') as f:
            data = f.read().strip('\n').split('\n')
        convert_dict = dict()
        # print(len(data))
        for i, d in enumerate(data):
            tmp_list = d.split( '::' )
            item_key = int(tmp_list[0])
            convert_dict[item_key] = i
        return convert_dict 

    convert_dict = clean_item(item_path)
    # print(len(convert_dict.items()))

    user_set, item_set = [], []
    data_record = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip('\n').split('\n')
    for d in data:
        if(len(d)):
            userId, movId, score, _ = [int(x) for x in d.split('::')]
            if value_form == 'binary' and score > 0:
                score = 1
            userId -= 1
            movId  = convert_dict.get(movId) ## clean data
            data_record.append([userId, movId, score])
            user_set.append(userId)
            item_set.append(movId)

    random.seed(0)
    traingset, testingset = [], []
    for d in data_record:
        if random.random() <= ratio:
            traingset.append(d)
        else:
            testingset.append(d)

    user_set, item_set = sorted(set(user_set)), sorted(set(item_set))
    return data_record, traingset, testingset, len(user_set), len(sorted(list(range(len(convert_dict)))))  #item_set 
    ##     data_record:: full of tri_table:: to make invert_file 
    ##     trainingset:: training dataset :: to make matrix , to calculate similarity 
    ##     testingset :: remaining tri_table:: to test in the end 
    ##     len(item) != len(dict), because not all movies are being rated in the movie.dat 


# def data_loader_ml10m_sift(file_path, item_path, sep='::', value_form='remain'):

#     def clean_item(item_path):
#         with open(item_path, 'r', encoding='utf-8') as f:
#             data = f.read().strip('\n').split('\n')
#         convert_dict = dict()
#         # print(len(data))
#         for i, d in enumerate(data):
#             tmp_list = d.split( '::' )
#             item_key = int(tmp_list[0])
#             convert_dict[item_key] = i
#         return convert_dict 

#     convert_dict = clean_item(item_path)

#     user_set, item_set = [], []
#     data_record = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = f.read().strip('\n').split('\n')
#     for d in data:
#         if(len(d)):
#             userId, movId, score, time_stamp = [int(x) for x in d.split('::')]
#             if value_form == 'binary' and score > 0:
#                 score = 1
#             userId -= 1
#             movId  = convert_dict.get(movId) ## clean data
#             data_record.append([userId, movId, score, time_stamp])
#             user_set.append(userId)
#             item_set.append(movId)

#     # random.seed(0)
#     # traingset, testingset = [], []
#     # for d in data_record:
#     #     if random.random() <= ratio:
#     #         traingset.append(d)
#     #     else:
#     #         testingset.append(d)

#     training_flag = [
#         True for _ in range(len(data_record))
#     ]


#     user_set, item_set = sorted(set(user_set)), sorted(set(item_set))
#     return data_record, traingset, testingset, len(user_set), len(sorted(list(range(len(convert_dict)))))  #item_set 





#-----------------------------------------------------------------------

def test_set2ground_truth(testset, num_users):
    ground_truth = [
        [] for _ in range(num_users)
    ]
    for test in testset:
        u, i, _ = test
        ground_truth[u].append(i)
    return ground_truth 

def test_set2ground_truth2(testset, num_users):
    ground_truth = [
        [] for _ in range(num_users)
    ]
    for test in testset:
        u, i = test
        ground_truth[u].append(i)
    return ground_truth 


def convert2matrix(tri_table, n_users, n_items, matrix_form='numpy', value_form='remain'):
    if matrix_form == 'sparse':
        matrix = sp.dok_matrix((n_user, n_items), dtype=np.float32)
    else:
        matrix = np.zeros((n_users, n_items), dtype=np.float32) 
    for t in tri_table:
        u, i, r = t 
        if value_form == 'remain':
            matrix[u,i] = r 
        else:
            matrix[u,i] = 1.0 #r ## not 1, now has rating scores
    return matrix 

def convert2invert_file(tri_table, n_user, n_item, main_key=0):
    """
    main_key in the List.
    sub_key  is the index.
    return:
        main_key = 0: [
            i:[u_1, u_2, ...]
        ]
        main_key = 1: [
            u:[i_1, i_2, ...]
        ]
    """
    if main_key == 0:
        length = n_item
        sub_key = 1
    else:
        length = n_user
        sub_key = 0 

    invert_file = [
        [] for _ in range(length) 
    ]

    for t in tri_table:
        # u, i, _ = t 
        main_v, sub_v = t[main_key], t[sub_key]
        invert_file[sub_v].append(main_v) # sub.append(main) is the invert file 
    return invert_file 

# def load_negative(trainingset) #TODO:


if __name__ == "__main__":
    pass 