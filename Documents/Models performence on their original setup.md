# Report

## PureSVD

> Performance of Recommender Algorithms on Top-N Recommendation Tasks

PureSVD竟然最初不是给implicit feedback用的。 

* Setup

  * 数据集
    * Netflix dataset (自带训练集 & 验证集) 100M + 384,573
    * MovieLens 在observation上分1.4%作为验证集，验证集里只选5分的作为test set

  * 分割方法：

    Observed ratings splits into training set & test set. Test set only contains 5 stars point.

    对奈飞：训练集就是奈飞的训练集，验证集只挑选里面5分的。

    对ML ： 整体分1.4%出来，剩余是训练，其余是验证，验证里只挑5分作为test

  * Evaluation

    * RECALL & Precision 



==不好==和原文对不上。



## NeuCF

> Neural Collaborative Filtering WWW2017

<img src="https://s2.loli.net/2022/06/20/JgouhGXPlZYazDs.png" alt="image-20220620151755778" style="zoom: 33%;" />



* Setup

  * 数据集
    * MovieLens 1M 
      * 训练集和测试集是在Observation上Leave-LastOne-Out。 
      * 每个正样本随机采样4个负样本

  * Model

  

* Result

  | MovieLens | HR     | NDCG   |
  | --------- | ------ | ------ |
  | **DIY**   | 0.6631 | 0.3861 |
  | Paper     | 0.688  | 0.410  |
  |           |        |        |

## SLIM 

* Setup
  * 数据集
    ML10M 
  * 分割方法
    Leave-one-out  last-one
  * 

