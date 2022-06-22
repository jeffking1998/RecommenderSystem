# $EASE^{R}$

> Note：
>
> * 它和AE的联系：试图让input跟output尽可能相似。
>
> * topN推荐： X中的数据是二值的。



## Model 

#### Notations

* User-Item Matrix $X$
    $$
    X \in R^{|\mathcal{U}|\times |\mathcal{I}|}
    $$

* Item-Item Weight Matrix $B$

$$
B \in R^{ |\mathcal{I}| \times |\mathcal{I}| }
$$

* Constraint of $B$
    $$
    diag(B) = 0
    $$

* Predicted Scores  $S_{u,j}$

    > $S_{u,j}$ is u-th row of X multiplied by j-th column of B 
    >
    > $S$ can be seen as $\hat{X}$. This is why  

    $$
    S_{u,j} = X_{u,\cdot} \cdot B_{\cdot, j}
    $$

* Object Function
    $$
    min_{B} ~~ ||X - XB||_F^2 + \lambda \cdot ||B||_F^2 
    \\
    s.t. ~~ diag(B) = 0 
    $$

* Loss Funtion based on Lagrange Multiplier
    $$
    L = ||X - XB||_F^2 + \lambda \cdot ||B||_F^2 + 2 \cdot \gamma^T \cdot diag(B)
    $$

#### Closed-Form Solution of Loss Function 

> derivative of Loss function and set it 0.

$$
\frac{\partial L}{\partial B} = 2X^T(XB-X) + 2\lambda B + 2 \cdot diag(\gamma) = 0
\\
(X^T X + \lambda) B = X^T X - diag(\gamma) 
\\
B = (X^T X + \lambda)^{-1} (X^T X - diag(\gamma))
$$

> Define: $\hat{p} = (X^T X + \lambda)^{-1}$ 

$$
B = \hat{p} (\hat{p}^{-1} - \lambda I - diag(\gamma)) 
\\
B = I - \hat{p}( \lambda I + diag(\gamma) ) 
$$

> Define: $\hat{\gamma} = \lambda I + \gamma$

$$
B = I - \hat{p} \cdot diag(\hat{\gamma})
$$



> Constraint: $diag(B) = 0$

$$
diag(I - \hat{p} \cdot diag(\hat{\gamma})) = 0 
\\
\vec{1} - diag(\hat{p}) \odot \hat{\gamma} = 0
\\
\hat{\gamma} = 1 \oslash diag(\hat{p})
$$

> Final form of $B$

$$
B = I - \hat{p} \cdot diagM( 1 \oslash diag(\hat{p}) )
$$

> OR

$$
B_{i,j} = \begin{cases}
0  ~~~~~~~~~~~~~\text{if}~~~~ i=j\\
- \frac{\hat{p}_{i,j}}{\hat{p}_{j,j}} ~~~~~~~otherwise
\end{cases} 
$$

#### Algorithm of $EASE^R$

*   Python 

```python
'''
X: U-I matrix
_lambda : (vector) reg parameter
'''
G = np.dot(X, X) 
diag_index = np.diag_indices(G.shape[0]) 
G[diag_index] += _lambda 
P = np.linalg.inv(G) 
B = P / (- np.diag(P))
B[diag_index] = 0
```



*   Matlab 

```matlab
clc;clear; close all;

X = sprandn(500,1000,0.1); %rmse: 0.0303
X = logical(X);

[w, h] = size(X);

sparse = sum(X, 'all') / (w*h);
sparse
 
lambda = 0.01;
 
 
G = X' * X; 
G = G + lambda * eye(size(G));

P = pinv(G);
B = - P ./ diag(P);

logi_matrix = ~eye(size(B));

B = B .* logi_matrix;

%-------------------------------

% Try X ~~ XB 

err = X - X * B;
rmse = sqrt(mean(err .* err, 'all'));
disp(rmse);
```

 

#### References

> * Paper
> 
>     Harald Steck (2019). Embarrassingly Shallow Autoencoders for Sparse Data arXiv: Information Retrieval.
>
> * Frobenius norm: definition
>
>     https://www.zhihu.com/question/22475774
>
> * derivative of Frobenius norm
>
>     https://math.stackexchange.com/questions/2128462/derivative-of-squared-frobenius-norm-of-a-matrix
>
>     https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf [P14]
>
> * Lagrange_multiplier
>     https://en.wikipedia.org/wiki/Lagrange_multiplier

