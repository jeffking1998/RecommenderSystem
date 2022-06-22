from sklearn.neighbors import NearestNeighbors




def cosine(X):
    knn = NearestNeighbors(n_neighbors=self.topK,
                            metric='cosine',
                            algorithm='auto',
                            n_jobs=-1,
                            ).fit(X)
    distances, indices = knn.kneighbors(X)
    return distances, indices 

def pearson(X):
    knn = NearestNeighbors(n_neighbors=self.topK,
                            metric='correlation',
                            algorithm='auto',
                            n_jobs=-1,
                            ).fit(X)
    distances, indices = knn.kneighbors(X)
    return distances, indices 


def adjust_cosine(X):
    pass 

def msd(X):
    pass 

def src(X):
    pass


if __name__ == "__main__":
    pass  