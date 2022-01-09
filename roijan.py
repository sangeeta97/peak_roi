import pandas as pd
import numpy as np
import re
from sklearn.metrics import pairwise_distances_chunked
from scipy.spatial.distance import squareform, pdist
from itertools import chain


a= []
b= []



def roi_seed(mz):
    points= mz
    clusters = []
    index= []
    eps = 3/1000000
    points_sorted = sorted(points)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    curr_index= [0]
    for k, point in enumerate(points_sorted[1:]):
        k= k+1
        if eps <= (curr_point - point)/point:
            curr_cluster.append(point)
            curr_index.append(k)

        else:
            clusters.append(curr_cluster)
            index.append(curr_index)
            curr_cluster = [point]
            curr_index= [k]
            curr_point = point
    clusters.append(curr_cluster)
    index.append(curr_index)
    return clusters, index





def duplicates(lst, item):
   return [i for i, x in enumerate(lst) if x == item]




def ppm(D_chunk, distances, ppm= 2):
    for l, v in zip(D_chunk, distances):
        ik= (v/l)*1000000 <= ppm
        mm= tuple(np.nonzero(ik)[0])
        mm= set(mm)
        return mm



def reduce_func(self, D_chunk, start):
    neigh = np.array([np.flatnonzero(d/X[i][0]< 2/1000000)
                     for i, d in enumerate(D_chunk, start)])
    return neigh




df1 = pd.read_csv(filename)
df1= df1.rename({'Unnamed: 0': 'index'}, axis= 1, inplace= True)
mz= df1.mz.values
mapped= df1['index'].values
X = np.reshape(mz, (-1, 1))
D_chunk = pairwise_distances_chunked(X)
gen_chunk= pairwise_distances_chunked(X, reduce_func=reduce_func)




class Roi():
    def __init__(self, gen_chunk, D_chunk):
        super().__init__()
        self.gen = gen_chunk
        self.D= D_chunk




    def uniqueRows(self, metric='euclidean'):
        "Returns subset of rows that are unique, in terms of Euclidean distance"
        self.distances = squareform(pdist(self.D_chunk, metric=metric))
        idxset= ppm(self.D_chunk, self.distances)
        xx, yy = X[[x[0] for x in idxset]], self.neigh[[x[0] for x in idxset]]
        final_outcome.append([{x[0]:mapped[y]} for x, y in a, b])
        a.append(xx)
        b.append(yy)




def create_slaves(data_dir):
    try:
        gg= []
        for q, w in zip(gen_chunk, D_chunk):
            obj123= Roi(q, w)
            gg.append(obj123)
    except Exception as e:
        print(e)



from threading import Thread

def execute(workers):
    try:
        threads = [Thread(target=w.uniqueRows) for w in workers]
        for thread in threads: thread.start()
        for thread in threads: thread.join()

    except Exception as e:
        print(e)



def combine_all():
    combine= []
    workers= create_slaves(".")
    execute(workers)
    a= np.flatten(a)
    xy= Counter(a)
    for i , j in xy.items():
        if j > 1:
            index_dup= duplicates(a, i)
            combine.append(index_dup)
    return set(combine)



new_dict1= dict()


def merge():
    b= np.array(b)
    combine= combine_all()
    for j in combine:
        j= np.array(j)
        cy= b[j]
        cy= np.flatten(cy)
        b[j]= cy
    a= set(a)
    a= np.array(a)
    b= set(b)
    b= np.array(b)
    dict1= {k:v for k,v in zip(a, b)}
    lk= list(dict1.keys())
    t, y= roi_seed(lk)
    for i, j in zip(t, y):
      if len(i)> 1:
        j= []
        for x in i:
          p= dict1[x]
          j.extend(p)
      i= np.mean(np.array(i))
      new_dict1[i]= j









def filter_roi(new_dict1):
    rm= dict()
    for k,v in new_dict1.items():
        if len(v) < 8:
            rm.update({k:v})
    return rm




if __name__ == "__main__":
    combine_all()
    merge()
    #
    # rm= filter_roi(dict1)
    # for l,m in rm.items():
    #    dict1.pop(l)
    # import json
    # with open('convertfinal2.txt', 'w') as convert_file:
    #      convert_file.write(json.dumps(dict1))
    # with open('rmfinal2.txt', 'w') as convert_file:
    #      convert_file.write(json.dumps(rm))
