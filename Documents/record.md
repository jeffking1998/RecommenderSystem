    """
    ml10m_inter = ml10m_folder + 'ml-10m.inter' 


    data = pd.read_csv(ml10m_inter, sep='\t')   

    num_users = data['user_id:token'].unique().shape[0]
    num_items = data['item_id:token'].unique().shape[0]

    grouped = data.groupby('user_id:token')
    last_time = grouped['timestamp:float'].transform('max')

    training_flag = data['timestamp:float'] != last_time
    training_flag = training_flag.to_numpy()

    training_set = data[training_flag]
    test_set     = data[~training_flag]

    training_set = training_set[['user_id:token', 'item_id:token', 'rating:float']].to_numpy()
    test_set     = test_set[['user_id:token', 'item_id:token', 'rating:float']].to_numpy()

    training_set = training_set.astype('int')
    test_set     = test_set.astype('int')
    """



## PureSVD 

#### PureSVD50
N = 5
mean_precision = 0.1785 
mean_recall = 0.1877 
mean_avg_precision = 0.1627 
mean_ndcg = 0.2383 

N = 10
mean_precision = 0.1388 
mean_recall = 0.278 
mean_avg_precision = 0.1559 
mean_ndcg = 0.2545 

N = 15
mean_precision = 0.1171 
mean_recall = 0.3441 
mean_avg_precision = 0.1597 
mean_ndcg = 0.2736 

N = 20
mean_precision = 0.1016 
mean_recall = 0.3906 
mean_avg_precision = 0.1636 
mean_ndcg = 0.2882 

N = 50
mean_precision = 0.06108 
mean_recall = 0.5562 
mean_avg_precision = 0.1775 
mean_ndcg = 0.3401 


#### PureSVD150
N = 5
mean_precision = 0.1212 
mean_recall = 0.124 
mean_avg_precision = 0.1007 
mean_ndcg = 0.157 

N = 10
mean_precision = 0.09472 
mean_recall = 0.1882 
mean_avg_precision = 0.09566 
mean_ndcg = 0.1692 

N = 15
mean_precision = 0.08045 
mean_recall = 0.2362 
mean_avg_precision = 0.09783 
mean_ndcg = 0.1831 

N = 20
mean_precision = 0.07095 
mean_recall = 0.2762 
mean_avg_precision = 0.1005 
mean_ndcg = 0.1956 

N = 50
mean_precision = 0.04485 
mean_recall = 0.417 
mean_avg_precision = 0.11 
mean_ndcg = 0.2391 



## SLIM 

## XXnet
N = 5
mean_precision = 0.4245 
mean_recall = 0.142 
mean_avg_precision = 0.3506 
mean_ndcg = 0.4539 

N = 10
mean_precision = 0.356 
mean_recall = 0.2255 
mean_avg_precision = 0.2876 
mean_ndcg = 0.4247 

N = 50
mean_precision = 0.193 
mean_recall = 0.548 
mean_avg_precision = 0.2417 
mean_ndcg = 0.4595 

### SGD
N = 5
mean_precision = 0.4244 
mean_recall = 0.1489 
mean_avg_precision = 0.3469 
mean_ndcg = 0.4558 

N = 10
mean_precision = 0.3577 
mean_recall = 0.2365 
mean_avg_precision = 0.2864 
mean_ndcg = 0.4291 

N = 50
mean_precision = 0.1939 
mean_recall = 0.5563 
mean_avg_precision = 0.2481 
mean_ndcg = 0.467 