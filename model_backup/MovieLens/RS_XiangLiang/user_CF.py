import random 


"""
TODO:
Q0: main problem is sloved.
Q1: low scores should be fitered or not?
"""

def movielens_reader(data_path, training_ratio=1.0, filter_low=False):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read().strip('\n').split('\n')
    data_record = []
    for d in data:
        if(len(d)):
            userId, movId, score, _ = [int(x) for x in d.split('::')]
            if filter_low:
                if score >= 3:
                    data_record.append([userId, movId, score])
            else:
                data_record.append([userId, movId, score])
    ## spliting the dataset into training and testing:
    random.seed(0)
    traingset, testingset = dict(), dict()
    for d in data_record:
        userId, movId, _ = d 
        if random.random() <= training_ratio:
            if traingset.get(userId):
                traingset[userId].append(movId)
            else:
                traingset[userId] = [movId]
        else:
            if testingset.get(userId):
                testingset[userId].append(movId)
            else:
                testingset[userId] = [movId]
    # print(len(data) == len(data_record), len(data_record) == len(traingset) + len(testingset))
    return traingset, testingset

def CalUserSim(trainingset):
    inversed_item_table = dict() 
    N = dict()
    for u, items in trainingset.items():
        N[u] = len(items)
        for itm in items:
            if inversed_item_table.get(itm):
                inversed_item_table[itm].append(u)
            else:
                inversed_item_table[itm] = [u] 
    intersection = dict()
    for _, users in inversed_item_table.items():
        for i in users:
            for j in users:
                if i != j:
                    if i not in intersection:
                        intersection[i] = dict()
                    if j not in intersection[i]:
                        intersection[i][j] = 0
                    intersection[i][j] += 1
                    # if j not in intersection:
                    #     intersection[j] = dict()
                    # if i not in intersection[j]:
                    #     intersection[j][i] = 0
                    # intersection[j][i] += 1
    Sim = dict()
    for u, related_users in intersection.items():
        for v, val in related_users.items():
            if u not in Sim:
                Sim[u] = dict()
            Sim[u][v] = val / (N[u] * N[v]) ** 0.5
    return Sim  

def Recommend(a_user, trainingset, Sim):
    item_candidate_rank = dict()

    has_brought_items = trainingset[a_user]
    similar_user = Sim[a_user] 
    similar_user = dict(sorted(similar_user.items(), key=lambda item: item[1], reverse=True)[:10])
    for v, similarity in similar_user.items():
        for itm in trainingset[v]: # items of v
            if itm in has_brought_items:
                continue
            if itm not in item_candidate_rank:
                item_candidate_rank[itm] = 0
            item_candidate_rank[itm] += similarity
    return dict(sorted(item_candidate_rank.items(), key=lambda item: item[1], reverse=True)[:10]) # [: 10] recom ALL

def recommend_multi_uses(user_list, trainingset, Sim):
    all_recommend = dict()
    for user in user_list:
        this_recom = list( Recommend(user, trainingset, Sim).keys() )
        all_recommend[user] = this_recom
    return all_recommend

def recall(recommend_results, testingset):
    hitting_num, recommend_num = 0, 0

    for u, recom in recommend_results.items():
        ground_truth = set(testingset[u]) 
        unique_recom = set(recom)
        hitting_num += len( ground_truth & unique_recom )
        recommend_num += len(unique_recom)

    return hitting_num / recommend_num 

def precision(recommend_results, testingset):
    hitting_num, gt_num = 0, 0
    for u, recom in recommend_results.items():
        ground_truth = set(testingset[u])
        unique_recom = set(recom) 
        hitting_num += len( ground_truth & unique_recom )
        gt_num += len(ground_truth)
    
    return hitting_num / gt_num 

if __name__ == "__main__":
    trainingset, testingset = movielens_reader('movielens_1m/ml-1m/ratings.dat', training_ratio=0.8, filter_low=True)
    print( len(trainingset.keys()) , len(testingset.keys())  ) 
    # what about if those two lens are not equals. 
    Sim = CalUserSim(trainingset)
    # a_user = 1
    # candidate_items = Recommend(a_user, trainingset, Sim)
    # print(candidate_items)
    # print('----------')
    # print(testingset[a_user])

    testing_recom = recommend_multi_uses(testingset.keys(), trainingset, Sim)
    print(recall(testing_recom, testingset))
    print(precision(testing_recom, testingset))


