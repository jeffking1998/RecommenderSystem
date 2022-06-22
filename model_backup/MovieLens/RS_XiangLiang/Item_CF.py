import numpy as np 
import random  


def read_raw_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read().split('\n')
    raw = [x.split('::')  for x in raw if len(x)] 
    raw = [ [int(x[0]), int(x[1]), int(x[2])] for x in raw]
    return np.array(raw)



def split_dataset(data, split_ratio, filt_low):
    training_set, testing_set = [], []
    random.seed(0)
    for d in data:
        _, _, score = d 
        if filt_low:
            if score >= 3.0:
                if random.random() >= split_ratio:
                    testing_set.append(d) 
                else:
                    training_set.append(d)
        else:
            if random.random() >= split_ratio:
                testing_set.append(d)
            else:
                training_set.append(d)
    return training_set, testing_set 

def movie_loader(file_path, split_ratio=0.8, filt_low=False):
    return split_dataset(read_raw_data(file_path), split_ratio, filt_low)

#--------------------------------------------------#

def build_item_users_table(train_set):
    item_users = dict()
    for d in train_set:
        user, item, score = d
        item_users[item] = item_users.get(item, [])
        item_users[item].append(user)
    return item_users 


def invert_file(item_users):
    user_items = dict()
    for k_item in item_users.keys():
        for u in item_users.get(k_item, []):
            user_items[u] = user_items.get(u, [])
            user_items[u].append(k_item)
    return user_items 



def cal_common_size(invertfile):
    common_matrix = dict() ## a matrix works as a dict 
    
    for _, items in invertfile.items():
        for it_a in items:
            common_matrix[it_a] = common_matrix.get(it_a, dict())
            for it_b in items:
                if it_a == it_b:
                    continue
                common_matrix[it_a][it_b] = common_matrix[it_a].get(it_b, 0)
                common_matrix[it_a][it_b] += 1
    return common_matrix



def cal_items_weight(training_set):
    item_users = build_item_users_table(training_set)
    invertfile = invert_file(item_users) ## cannot has a same name with a function 
    common_matrix = cal_common_size(invertfile)

    item_holding_size = dict()
    for k_it, u_list in item_users.items():
        item_holding_size[k_it] = len(u_list)

    weight_matrix = dict() 

    for it_a in common_matrix.keys():
        it_bs = common_matrix[it_a]
        weight_matrix[it_a] = weight_matrix.get(it_a, dict())
        for it_b in it_bs:
            weight_matrix[it_a][it_b] = weight_matrix[it_a].get(it_b, 0)
            weight_matrix[it_a][it_b] = common_matrix[it_a][it_b] / np.sqrt( item_holding_size[it_a] * item_holding_size[it_b] )
    return weight_matrix

def recommend(user_items, weight_matrix, a_user, NtopN=10, NtopK=10):
    # weight_matrix = cal_items_weight(training_set)
    items_this_user_brought = user_items[a_user]

    rank = dict()

    for item in items_this_user_brought:
        tmp = weight_matrix.get(item)
        topK_items_like_this_item = dict(sorted(tmp.items(), key=lambda item: item[1], reverse=True)[: NtopK])
        for cd_item, sim in topK_items_like_this_item.items():
            rank[cd_item] = rank.get(cd_item, 0)
            rank[cd_item] += sim ## another for loop?

    return dict(sorted(rank.items(), key=lambda item: item[1], reverse=True)[: NtopN])

def recommend_multi_users(user_list, user_items, weight_matrix, NtopN=10, NtopK=10):
    recommends = []
    for u in user_list:
        recm = recommend(user_items, weight_matrix, u, NtopN=NtopN, NtopK=NtopK)
        recommends.append(recm.keys())
    return recommends

if __name__ == "__main__":
    training_set, testing_set = movie_loader('movielens_1m/ml-1m/ratings.dat', 0.8, False)
    weight_matrix = cal_items_weight(training_set)

    item_users = build_item_users_table(training_set)
    user_items = invert_file(item_users)
    print(len(user_items[1]), len(item_users[1]))
    print(recommend(user_items, weight_matrix, 1))




