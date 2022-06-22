# Implement of RS Algorithms

## Intro of this resp 

At first, I categorized it by methods, like NN-based and model-based. Now, I categorized it by tasks, i.e. predict ratings or item recommendation (TopN recommendation).

### Item Recommendation

- PureSVD / EASE^R

    Those two methods are based on MF. They have no opti steps. So no need care about losses / negative samples.
- SLIM 

   SLIM was for leave-one-out Item recommendation. In my rasp, i need to use a modified version of SLIM.
    Slim uses ML-10M, my m1 MBA cannot run it smoothly.

## Document

*   Intro of RS
  
    [Intro](./Documents/Intro_about_CF.md)

*   EASE^R 

    [EASE^R](./Documents/EASE_R.md)


## TODO
Logistic以及特征交叉的算法：Poly2, FM, FFM 等都需要引入side information。暂时手头没有这种数据。 
哦，其实有，movielens就有。我还没看。

## Method for Pushing update to Github
>
>     1.  git add .
>     2.  git commit -m "info" 
>     3.  git push -u origin main 


I have moved my focus on top-N recommendation task right now. NO ~~predict ratings task~~


## Algorithms that has been implemented

1.   Memory Based
     *   User Based
     *   Item Based 
2.   Model Based
     *   SVD in Predicting Ratings 
     *   EASE^R 


## Saying 
- MF矩阵分解中，任何一个u,i都对应一个f维的向量。这其实不就是Embedding吗？
- 《深度学习推荐系统》讲的是具体算法，以及算法的演化，对大脑形成算法进化的框架很有帮助。 
- ~~目前我需要的是探索下引入side info的算法&数据，不能被implicit的数据浸没住了。~~
- 还是先搞implicit的，全部的方法都试一遍。
- U_test \cap U_train = \Phi 


#### Note:

The Pytorch version can be access from my  [RS_D2L repository](https://github.com/jeffking1998/pytorch_RS_D2L).