import pandas as pd
import numpy as np
import re
from sklearn.metrics import pairwise_distances_chunked
from scipy.spatial.distance import squareform, pdist
from itertools import chain


final_outcome= []



def reduce_func(self, D_chunk, start):
    neigh = np.array([np.flatnonzero(d/X[i][0]< 2/1000000)
                     for i, d in enumerate(D_chunk, start)])
    return neigh



class Roi():
    def __init__(self, chunk):
        super().__init__()
        self.df1 = pd.read_csv(chunk)
        self.df1= self.df1.rename({'Unnamed: 0': 'index'}, axis= 1, inplace= True)



    def mapping(self):
        mz= self.df.mz.values
        mapped= mz['index'].values
        X= vec2matrix(mz,ncol=1)
        self.D_chunk = next(pairwise_distances_chunked(X))
        self.neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
        a, b= uniqueRows(self.D_chunk)
        final_outcome.append([{x[0]:mapped[y]} for x, y in a, b])



    def uniqueRows(self, thresh=0.002, metric='euclidean'):
        "Returns subset of rows that are unique, in terms of Euclidean distance"
        distances = squareform(pdist(self.D_chunk, metric=metric))
        idxset = {tuple(np.nonzero(v)[0]) for v in distances <= thresh}
        return X[[x[0] for x in idxset]], self.neigh[[x[0] for x in idxset]]




def create_slaves(data_dir):
    try:
        gg= []
        with pd.read_csv(data_dir, sep=",", chunksize=100) as reader:
        for chunk in reader:
            obj123= Roi(chunk)
#            cc= obj123.uniqueRows()
            gg.append(obj123)

        return gg

    except Exception as e:
        print(e)
        warnings.warn(f"The error {e} has been found")



from threading import Thread

def execute(workers):
    try:
        threads = [Thread(target=w.mapping) for w in workers]
        for thread in threads: thread.start()
        for thread in threads: thread.join()

    except Exception as e:
        print(e)
        warnings.warn(f"The error {e} has been found")



def combine_all():
    workers= create_slaves(".")
    execute(workers)
    px= np.array([item for for item in chain(x.keys() for x in final_outcome)])
    tx= np.array([item for for item in chain(x.values() for x in final_outcome)])
    X= vec2matrix(px,ncol=1)
    D_chunk = next(pairwise_distances_chunked(X))
    neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
    distances = squareform(pdist(D_chunk, metric='euclidian'))
    idxset = {tuple(np.nonzero(v)[0]) for v in distances <= 0.002}
    makekey= X[[x[0] for x in idxset]]
    makevalue= neigh[[x[0] for x in idxset]]
    makevalue= [(tx[i]).flatten() for i in makevalue]
    return {x: y for x, y in zip(makekey, makevalue)}





def filter_roi(new_dict1):
    rm= dict()
    for k,v in new_dict1.items():
        if len(v) < 8:
            rm.update({k:v})
    return rm




if __name__ == "__main__":
    dict1= combine_all()
    rm= filter_roi(dict1)
    for l,m in rm.items():
       dict1.pop(l)
    import json
    with open('convertfinal2.txt', 'w') as convert_file:
         convert_file.write(json.dumps(dict1))
    with open('rmfinal2.txt', 'w') as convert_file:
         convert_file.write(json.dumps(rm))
