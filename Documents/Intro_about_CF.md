# Summary of Recommendation Systems 

-----------------------

[toc]

---------

## 0 $\star$ Current States

*   ~~User-Based~~
*   ~~Item-Based~~
*   ~~SVD~~

------------



## 1. User-Based & Item-Based TopK Methods

### User-Based

#### Basic idea

用这个用户的topN个最相似的用户们购买的商品中减去他自己已经买的，剩余的和这个用户最相似的TopK个商品，推荐给这个用户。

#### Key idea

* 首先需要两个相似度计算公式

  * User-User Similarity 
    $$
    sim_{u,v} = \frac{|u.Items \cap  v.Items|}{\sqrt{ u.Items.size() \times v.Items.size() }}
    $$

    * 分母比较好解决

    * 分子的话，用Invert File（倒排表）
      即本来说U-> U.Items，反过来建立I->I.Users，然后：

      ```shell
      for i in Items:
          user_set = i.Users
          for u in user_set:
              for v in user_set:
                  if u == v:
                      continue
                  N_user_interaction(u,v) += 1
      ```

      

  * User-Item Similarity
    $$
    sim_{u,i} = \sum_{v \in (kNN(u) \cap i.Users)} sim_{u,v}
    $$
    这里面u是固定的，因为我们已经知道推荐对象是谁；==可是i呢==？i的选择必须是u的KNN和i的Users有交集，这个是挺难的，绝大多数可能都不符合。换个思路，从u的KNN中的人的商品，这些商品至少有一个交集，就是这个KNN。

    来个$v \in KNN$就行，检查`v.Items`，只要没买过就是候选的，它的相似度就累加$<u,v>$的相似度就行。

    i有很多个，对吧？建立一个${i: sim_{u,i}}$的字典，逐次更新。更新规则按上面的

    ```shell 
    for v in KNN(u):
    	uv_sim = Sim(u,v) 
    	for i in v.Items:
    		if i in has_brought(u):
    			continue
    		sim(i) += uv_sim 
    
    sorted( [i] by i.similarity)
    ```

    

* 数据结构

  * 训练集-验证集的划分
    用的是`(User, Item, Rating)`三元组用随机数随机划分的
  * 划分后的训练集与验证集需要组织成User-Item矩阵的形式
    有个很棘手的问题是，无论是user的index，还是item的index可能不是从0开始的按顺序排列好的index
    用`dict`来放这些数据，竟然还和用矩阵的很像。~~（当然最好还会清洗数据，用矩阵）~~

#### Fit & Recommendate

* Fit 

  Fit的过程就是计算User-User相似度的过程

* Recommendation

  推荐的过程就是计算User-Item相似度并排序的过程

### Item-Based

#### Basic idea

每个Item有它的KNN的~Item~，按照这些所有的candidate~Item~和User的相似度最高的topN个推荐给User

#### Key idea

* 两个相似度公式

  * Item-Item相似度
    $$
    Sim_{i,j} = \frac{|i.User \cap j.User|}{\sqrt{|i.User| \times |j.User|}}
    $$
    两个商品的感兴趣人群越接近，这两个商品就越相似

    ```shell 
    // utilize the `invert file` again
    for u in Users:
    	item_set = u.items
    	for i in item_set:
    		for j in item_set:
    			if i == j:
    				continue
    			item_interaction[i,j] += 1
    ```

    

  * User-Item相似度

    最终商品是要推荐给人的，所以需要用户-商品相似度
    $$
    Sim_{u, i} = \sum_{j \in (u.Items \cap i.KNN)} sim(i,j)
    $$
    我买过的商品且和目标商品接近的商品们与目标商品相似度的和

    问题还是，目标不好找，不可能全部loop

    ```shell
    for i in u.Items:
    	for j in i.KNN:
    		sim(j) += sim(i,j)
    
    sorted(j, by j.similarity)
    ```

    

    #### Fit & Recommendation

    * Fit 

      就是计算Item-Item间相似度的过程

    * Recommendation

      就是计算User-Item相似度并排序的过程



## 2. SVD-Related

### SVD

#### Basic idea

* SVD

  把User-Item矩阵（训练集的）看做一个矩阵A，假如使用SVD对A进行分解：A=U$\Sigma V^T$。如果提取前$f$大的奇异值，可以认为矩阵U蕴含着User的特征，V蕴含着Item的特征。

* Least Square

  但是实际上并没有对A做SVD分解（==我也不知道为啥==）。而是对Rating建模，这次要把打分预测出来。
  $$
  \hat{r}_{u,i} = \mu + bias_u + bias_i + q_i ^T p_u
  $$
  而我们要优化的函数（Target Function / Loss Function）是：
  $$
  min_{b*, q*,p*} \sum_{u,i} (r - \hat{r})^2 + \lambda (bias_u^2 + bias_i^2 + ||q_i||^2 + ||p_u||^2)
  $$
  **优化**：

  *   定义： $e_{u,i} = r_{u,i} - \hat{r}_{u,i}$

  $$
  b_u \leftarrow b_u + \gamma \cdot (e_{u,i} - \lambda \cdot b_u)\\
  b_i \leftarrow b_i + \gamma \cdot (e_{u,i} - \lambda \cdot b_i)\\
  p_u \leftarrow p_u + \gamma \cdot (e_{u,i} \cdot q_i - \lambda \cdot p_u) \\
  q_i \leftarrow q_i + \gamma \cdot (e_{u,i} \cdot p_u - \lambda \cdot q_i)
  $$

  

* Evaluation
  $RMSE = \sqrt{ \sum_{u,i \in TestSet} (r_{u,i} - \hat{r}_{u,i})^2 / |TestSet| }$$

