
clc;clear; close all;

% X = [ 1 0 1 0
%      0 0 0 1
%      1 1 1 0
%      1 0 1 1 ];
%  
% range = [0,1];
% X = randi(range, 20, 300);
% 


% X = sprandn(500,1000,0.1); %rmse: 0.0303

% X = sprandn(1000,500,0.1);  %rmse: 0.2119


% X = sprandn(200,1000,0.1); %rmse: 0.0141

% X = sprandn(1000,200,0.1);  %rmse: 0.2642

% X = sprandn(1000,1000,0.1);  %rmse: 0.6574

% X = sprandn(500,500,0.1);  %rmse: 0.6508

% X = sprandn(1000,100,0.1);  %rmse: 0.2800

% X = sprandn(100,1000,0.1);  %rmse: 0.0091

% X = sprandn(1000,2000,0.1);  %rmse: 0.0219

X = sprandn(1000,1000,0.1);  %rmse: 0.6233

% 1. 
% Why when X_w quals X_h, 
% the performence goes bad signicantly?

% 2. 
% When user is less than items, 
% the result is better than multi-user condition.

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