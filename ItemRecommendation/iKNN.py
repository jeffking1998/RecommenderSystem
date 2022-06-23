# import numpy as np 


# class BaseRecommender(object):
#     """
#     Minimal interface to be implemented by recommenders, along with
#     some helper methods. A concrete recommender must implement the
#     recommend_items() method and should provide its own implementation
#     of __str__() so that it can be identified when printing results.

#     Notes
#     =====
#     In most cases you should inherit from either
#     `mrec.mf.recommender.MatrixFactorizationRecommender` or
#     `mrec.item_similarity.recommender.ItemSimilarityRecommender`
#     and *not* directly from this class.

#     These provide more efficient implementations of save(), load()
#     and the batch methods to recommend items.
#     """

#     def recommend_items(self,dataset,u,max_items=10,return_scores=True,item_features=None):
#         """
#         Recommend new items for a user.

#         Parameters
#         ==========
#         dataset : scipy.sparse.csr_matrix
#             User-item matrix containing known items.
#         u : int
#             Index of user for which to make recommendations.
#         max_items : int
#             Maximum number of recommended items to return.
#         return_scores : bool
#             If true return a score along with each recommended item.
#         item_features : array_like, shape = [num_items, num_features]
#             Optionally supply features for each item in the dataset.

#         Returns
#         =======
#         recs : list
#             List of (idx,score) pairs if return_scores is True, else
#             just a list of idxs.
#         """
#         raise NotImplementedError('you must implement recommend_items()')

#     def fit(self,train,item_features=None):
#         """
#         Train on supplied data. In general you will want to
#         implement this rather than computing recommendations on
#         the fly.

#         Parameters
#         ==========
#         train : scipy.sparse.csr_matrix or mrec.sparse.fast_sparse_matrix, shape = [num_users, num_items]
#             User-item matrix.
#         item_features : array_like, shape = [num_items, num_features]
#             Features for items in training set, required by some recommenders.
#         """
#         raise NotImplementedError('you should implement fit()')

#     def save(self,filepath):
#         """
#         Serialize model to file.

#         Parameters
#         ==========
#         filepath : str
#             Filepath to write to, which must have the '.npz' suffix.

#         Notes
#         =====
#         Internally numpy.savez may be used to serialize the model and
#         this would add the '.npz' suffix to the supplied filepath if
#         it were not already present, which would most likely cause errors
#         in client code.
#         """
#         if not filepath.endswith('.npz'):
#             raise ValueError('invalid filepath {0}, must have ".npz" suffix'.format(filepath))

#         archive = self._create_archive()
#         if archive:
#             np.savez(filepath,**archive)
#         else:
#             pickle.dump(self,open(filepath,'w'))

#     def _create_archive(self):
#         """
#         Optionally return a dict of fields to be serialized
#         in a numpy archive: this lets you store arrays efficiently
#         by separating them from the model itself.

#         Returns
#         =======
#         archive : dict
#             Fields to serialize, must include the model itself
#             under the key 'model'.
#         """
#         pass

#     @staticmethod
#     def load(filepath):
#         """
#         Load a recommender model from file after it has been serialized with
#         save().

#         Parameters
#         ==========
#         filepath : str
#             The filepath to read from.
#         """
#         r = np.load(filepath)
#         if isinstance(r,BaseRecommender):
#             model = r
#         else:
#             model = np.loads(str(r['model']))
#             model._load_archive(r)  # restore any fields serialized separately
#         return model

#     def _load_archive(archive):
#         """
#         Load fields from a numpy archive.

#         Notes
#         =====
#         This is called by the static load() method and should be used
#         to restore the fields returned by _create_archive().
#         """
#         pass

#     @staticmethod
#     def read_recommender_description(filepath):
#         """
#         Read a recommender model description from file after it has
#         been saved by save(), without loading any additional
#         associated data into memory.

#         Parameters
#         ==========
#         filepath : str
#             The filepath to read from.
#         """
#         r = np.load(filepath,mmap_mode='r')
#         if isinstance(r,BaseRecommender):
#             model = r
#         else:
#             model = np.loads(str(r['model']))
#         return str(model)

#     def __str__(self):
#         if hasattr(self,'description'):
#             return self.description
#         return 'unspecified recommender: you should set self.description or implement __str__()'

#     def batch_recommend_items(self,
#                               dataset,
#                               max_items=10,
#                               return_scores=True,
#                               show_progress=False,
#                               item_features=None):
#         """
#         Recommend new items for all users in the training dataset.

#         Parameters
#         ==========
#         dataset : scipy.sparse.csr_matrix
#             User-item matrix containing known items.
#         max_items : int
#             Maximum number of recommended items to return.
#         return_scores : bool
#             If true return a score along with each recommended item.
#         show_progress: bool
#             If true print something to stdout to show progress.
#         item_features : array_like, shape = [num_items, num_features]
#             Optionally supply features for each item in the dataset.

#         Returns
#         =======
#         recs : list of lists
#             Each entry is a list of (idx,score) pairs if return_scores is True,
#             else just a list of idxs.

#         Notes
#         =====
#         This provides a default implementation, you will be able to optimize
#         this for most recommenders.
#         """
#         recs = []
#         for u in xrange(self.num_users):
#             if show_progress and u%1000 == 0:
#                print(u,'..',
#             recs.append(self.recommend_items(dataset,u,max_items,return_scores)))
#         if show_progress:
#             print
#         return recs

#     def range_recommend_items(self,
#                               dataset,
#                               user_start,
#                               user_end,
#                               max_items=10,
#                               return_scores=True,
#                               item_features=None):
#         """
#         Recommend new items for a range of users in the training dataset.

#         Parameters
#         ==========
#         dataset : scipy.sparse.csr_matrix
#             User-item matrix containing known items.
#         user_start : int
#             Index of first user in the range to recommend.
#         user_end : int
#             Index one beyond last user in the range to recommend.
#         max_items : int
#             Maximum number of recommended items to return.
#         return_scores : bool
#             If true return a score along with each recommended item.
#         item_features : array_like, shape = [num_items, num_features]
#             Optionally supply features for each item in the dataset.

#         Returns
#         =======
#         recs : list of lists
#             Each entry is a list of (idx,score) pairs if return_scores is True,
#             else just a list of idxs.

#         Notes
#         =====
#         This provides a default implementation, you will be able to optimize
#         this for most recommenders.
#         """
#         return [self.recommend_items(dataset,u,max_items,return_scores) for u in xrange(user_start,user_end)]

#     def _zero_known_item_scores(self,r,train):
#         """
#         Helper function to set predicted scores/ratings for training items
#         to zero or less, to avoid recommending already known items.

#         Parameters
#         ==========
#         r : numpy.ndarray or scipy.sparse.csr_matrix
#             Predicted scores/ratings.
#         train : scipy.sparse.csr_matrix
#             The training user-item matrix, which can include zero-valued entries.

#         Returns
#         =======
#         r_safe : scipy.sparse.csr_matrix
#             r_safe is equal to r except that r[u,i] <= 0 for all u,i with entries
#             in train.
#         """
#         col = train.indices
#         if isinstance(r,csr_matrix):
#             max_score = r.data.max()
#         else:
#             max_score = r.max()
#         data = max_score * np.ones(col.shape)
#         # build up the row (user) indices
#         # - we can't just use row,col = train.nonzero() as this eliminates
#         #   u,i for which train[u,i] has been explicitly set to zero
#         row = np.zeros(col.shape)
#         for u in xrange(train.shape[0]):
#             start,end = train.indptr[u],train.indptr[u+1]
#             if end > start:
#                 row[start:end] = u
#         return r - csr_matrix((data,(row,col)),shape=r.shape)

# class ItemSimilarityRecommender(BaseRecommender):
#     """
#     Abstract base class for recommenders that generate recommendations
#     from an item similarity matrix.  To implement a recommender you just
#     need to supply the compute_similarities() method.
#     """

#     def fit(self,dataset,item_features=None):
#         """
#         Learn the complete similarity matrix from a user-item matrix.

#         Parameters
#         ==========
#         dataset : scipy sparse matrix or mrec.sparse.fast_sparse_matrix, shape = [num_users, num_items]
#             The matrix of user-item counts, row i holds the counts for
#             the i-th user.
#         item_features : array_like, shape = [num_items, num_features]
#             Features for items in training set, ignored here.
#         """
#         if not isinstance(dataset,fast_sparse_matrix):
#             dataset = fast_sparse_matrix(dataset)
#         num_users,num_items = dataset.shape
#         # build up a sparse similarity matrix
#         data = []
#         row = []
#         col = []
#         for j in xrange(num_items):
#             w = self.compute_similarities(dataset,j)
#             for k,v in enumerate(w):
#                 if v != 0:
#                     data.append(v)
#                     row.append(j)
#                     col.append(k)
#         idx = np.array([row,col],dtype='int32')
#         self.similarity_matrix = csr_matrix((data,idx),(num_items,num_items))

#     def _create_archive(self):
#         """
#         Return fields to be serialized in a numpy archive.

#         Returns
#         =======
#         archive : dict
#             Fields to serialize, includes the model itself
#             under the key 'model'.
#         """
#         # pickle the model without its similarity matrix
#         # and use numpy to save the similarity matrix efficiently
#         tmp = self.similarity_matrix
#         self.similarity_matrix = None
#         m = pickle.dumps(self)
#         self.similarity_matrix = tmp
#         if isinstance(self.similarity_matrix,np.ndarray):
#             archive = {'mat':self.similarity_matrix,'model':m}
#         elif isinstance(self.similarity_matrix,csr_matrix):
#             d = self.similarity_matrix.tocoo(copy=False)
#             archive = {'row':d.row,'col':d.col,'data':d.data,'shape':d.shape,'model':m}
#         else:
#             # similarity matrix has unexpected type
#             archive = None
#         return archive

#     def _load_archive(self,archive):
#         """
#         Load fields from a numpy archive.
#         """
#         if 'mat' in archive.files:
#             self.similarity_matrix = archive['mat']
#         elif 'row' in archive.files:
#             data = archive['data']
#             row = archive['row']
#             col = archive['col']
#             shape = archive['shape']
#             self.similarity_matrix = coo_matrix((data,(row,col)),shape=shape).tocsr()
#         else:
#             raise IOError('unexpected serialization format, cannot find similarity matrix')

#     def load_similarity_matrix(self,filepath,num_items,offset=1):
#         """
#         Load a precomputed similarity matrix from tsv.

#         Parameters
#         ==========
#         filepath : str
#             Filepath to tsv file holding externally computed similarity matrix.
#         num_items : int
#             Total number of items (might exceed highest ID in a sparse similarity matrix).
#         offset : int
#             Item index offset i.e. 1 if indices in file are 1-indexed.
#         """
#         y = np.loadtxt(filepath)
#         row = y[:,0]
#         col = y[:,1]
#         data = y[:,2]
#         idx = np.array([row,col],dtype='int32')-offset
#         self.similarity_matrix = csr_matrix((data,idx),(num_items,num_items))

#     def compute_similarities(self,dataset,j):
#         """
#         Compute pairwise similarity scores between the j-th item and
#         every item in the dataset.

#         Parameters
#         ==========
#         j : int
#             Index of item for which to compute similarity scores.
#         dataset : mrec.sparse.fast_sparse_matrix
#             The user-item matrix.

#         Returns
#         =======
#         similarities : numpy.ndarray
#             Vector of similarity scores.
#         """
#         pass

#     def get_similar_items(self,j,max_similar_items=30,dataset=None):
#         """
#         Get the most similar items to a supplied item.

#         Parameters
#         ==========
#         j : int
#             Index of item for which to get similar items.
#         max_similar_items : int
#             Maximum number of similar items to return.
#         dataset : mrec.sparse.fast_sparse_matrix
#             The user-item matrix. Not required if you've already called fit()
#             to learn the similarity matrix.

#         Returns
#         =======
#         sims : list
#             Sorted list of similar items, best first.  Each entry is
#             a tuple of the form (i,score).
#         """
#         if hasattr(self,'similarity_matrix') and self.similarity_matrix is not None:
#             w = zip(self.similarity_matrix[j].indices,self.similarity_matrix[j].data)
#             sims = sorted(w,key=itemgetter(1),reverse=True)[:max_similar_items]
#             sims = [(i,f) for i,f in sims if f > 0]
#         else:
#             w = self.compute_similarities(dataset,j)
#             sims = [(i,w[i]) for i in w.argsort()[-1:-max_similar_items-1:-1] if w[i] > 0]
#         return sims

#     def recommend_items(self,dataset,u,max_items=10,return_scores=True,item_features=None):
#         """
#         Recommend new items for a user.  Assumes you've already called
#         fit() to learn the similarity matrix.

#         Parameters
#         ==========
#         dataset : scipy.sparse.csr_matrix
#             User-item matrix containing known items.
#         u : int
#             Index of user for which to make recommendations.
#         max_items : int
#             Maximum number of recommended items to return.
#         return_scores : bool
#             If true return a score along with each recommended item.
#         item_features : array_like, shape = [num_items, num_features]
#             Features for items in training set, ignored here.

#         Returns
#         =======
#         recs : list
#             List of (idx,score) pairs if return_scores is True, else
#             just a list of idxs.
#         """
#         try:
#             r = (self.similarity_matrix * dataset[u].T).toarray().flatten()
#         except AttributeError:
#             raise AttributeError('you must call fit() before trying to recommend items')
#         known_items = set(dataset[u].indices)
#         recs = []
#         for i in r.argsort()[::-1]:
#             if i not in known_items:
#                 if return_scores:
#                     recs.append((i,r[i]))
#                 else:
#                     recs.append(i)
#                 if len(recs) >= max_items:
#                     break
#         return recs

#     def batch_recommend_items(self,
#                               dataset,
#                               max_items=10,
#                               return_scores=True,
#                               show_progress=False,
#                               item_features=None):
#         """
#         Recommend new items for all users in the training dataset.  Assumes
#         you've already called fit() to learn the similarity matrix.

#         Parameters
#         ==========
#         dataset : scipy.sparse.csr_matrix
#             User-item matrix containing known items.
#         max_items : int
#             Maximum number of recommended items to return.
#         return_scores : bool
#             If true return a score along with each recommended item.
#         show_progress: bool
#             If true print something to stdout to show progress.
#         item_features : array_like, shape = [num_items, num_features]
#             Features for items in training set, ignored here.

#         Returns
#         =======
#         recs : list of lists
#             Each entry is a list of (idx,score) pairs if return_scores is True,
#             else just a list of idxs.
#         """
#         try:
#             r = dataset * self.similarity_matrix.T
#         except AttributeError:
#             raise AttributeError('you must call fit() before trying to recommend items')
#         return self._get_recommendations_from_predictions(r,dataset,0,r.shape[0],max_items,return_scores,show_progress)

#     def range_recommend_items(self,
#                               dataset,
#                               user_start,
#                               user_end,
#                               max_items=10,
#                               return_scores=True,
#                               item_features=None):
#         """
#         Recommend new items for a range of users in the training dataset.
#         Assumes you've already called fit() to learn the similarity matrix.

#         Parameters
#         ==========
#         dataset : scipy.sparse.csr_matrix
#             User-item matrix containing known items.
#         user_start : int
#             Index of first user in the range to recommend.
#         user_end : int
#             Index one beyond last user in the range to recommend.
#         max_items : int
#             Maximum number of recommended items to return.
#         return_scores : bool
#             If true return a score along with each recommended item.
#         item_features : array_like, shape = [num_items, num_features]
#             Features for items in training set, ignored here.

#         Returns
#         =======
#         recs : list of lists
#             Each entry is a list of (idx,score) pairs if return_scores is True,
#             else just a list of idxs.
#         """
#         try:
#             r = dataset[user_start:user_end,:] * self.similarity_matrix.T
#         except AttributeError:
#             raise AttributeError('you must call fit() before trying to recommend items')
#         return self._get_recommendations_from_predictions(r,dataset,user_start,user_end,max_items,return_scores)

#     def _get_recommendations_from_predictions(self,r,dataset,user_start,user_end,max_items,return_scores=True,show_progress=False):
#         """
#         Select recommendations given predicted scores/ratings.

#         Parameters
#         ==========
#         r : scipy.sparse.csr_matrix
#             Predicted scores/ratings for candidate items for users in supplied range.
#         dataset : scipy.sparse.csr_matrix
#             User-item matrix containing known items.
#         user_start : int
#             Index of first user in the range to recommend.
#         user_end : int
#             Index one beyond last user in the range to recommend.
#         max_items : int
#             Maximum number of recommended items to return.
#         return_scores : bool
#             If true return a score along with each recommended item.
#         show_progress: bool
#             If true print something to stdout to show progress.

#         Returns
#         =======
#         recs : list of lists
#             Each entry is a list of (idx,score) pairs if return_scores is True,
#             else just a list of idxs.
#         """
#         r = self._zero_known_item_scores(r,dataset[user_start:user_end,:])
#         recs = [[] for u in xrange(user_start,user_end)]
#         for u in xrange(user_start,user_end):
#             ux = u - user_start
#             if show_progress and ux%1000 == 0:
#                print ux,'..',
#             ru = r[ux,:]
#             if return_scores:
#                 recs[ux] = [(i,v) for v,i in sorted(izip(ru.data,ru.indices),reverse=True) if v > 0][:max_items]
#             else:
#                 recs[ux] = [i for v,i in sorted(izip(ru.data,ru.indices),reverse=True) if v > 0][:max_items]
#         if show_progress:
#             print
#         return recs


# class KNNRecommender(ItemSimilarityRecommender):
#     """
#     Abstract base class for k-nn recommenders.  You must supply an
#     implementation of the compute_all_similarities() method.

#     Parameters
#     ==========
#     k : int
#         The number of nearest neighbouring items to retain
#     """

#     def __init__(self,k):
#         self.k = k

#     def compute_similarities(self,dataset,j):
#         A = dataset.X
#         a = dataset.fast_get_col(j)
#         d = self.compute_all_similarities(A,a)
#         d[j] = 0  # zero out self-similarity
#         # now zero out similarities for all but top-k items
#         nn = d.argsort()[-1:-1-self.k:-1]
#         w = np.zeros(A.shape[1])
#         w[nn] = d[nn]
#         return w

#     def compute_all_similarities(self,A,a):
#         """
#         Compute similarity scores between item vector a
#         and all the rows of A.

#         Parameters
#         ==========
#         A : scipy.sparse.csr_matrix
#             Matrix of item vectors.
#         a : array_like
#             The item vector to be compared to each row of A.

#         Returns
#         =======
#         similarities : numpy.ndarray
#             Vector of similarity scores.
#         """
#         pass

# class DotProductKNNRecommender(KNNRecommender):
#     """
#     Similarity between two items is their dot product
#     (i.e. cooccurrence count if input data is binary).
#     """

#     def compute_all_similarities(self,A,a):
#         return A.T.dot(a).toarray().flatten()

#     def __str__(self):
#         return 'DotProductKNNRecommender(k={0})'.format(self.k)

# class CosineKNNRecommender(KNNRecommender):
#     """
#     Similarity between two items is their cosine distance.
#     """

#     def compute_all_similarities(self,A,a):
#         return cosine_similarity(A.T,a.T).flatten()

#     def __str__(self):
#         return 'CosineKNNRecommender(k={0})'.format(self.k)


