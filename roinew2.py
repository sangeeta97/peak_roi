import pandas as pd
import numpy as np
import re



def roi_seed(mz):
    points= mz
    clusters = []
    index= []
    eps = 0.004
    points_sorted = sorted(points)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    curr_index= [0]
    for k, point in enumerate(points_sorted[1:]):
        k= k+1
        if point <= curr_point + eps:
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



class Roi():
    def __init__(self, path1):
        super().__init__()
        self.df1 = pd.read_csv(path1)
        self.df1= self.df1.rename({'Unnamed: 0': 'index'}, axis= 1, inplace= True)
        self.df2= self.df1[self.df1['scan']== 1]
        self.df3= self.df1[self.df1['scan']!= 1]
        self.mz= self.df2.mz.tolist()
        self.l, self.d= roi_seed(self.mz)
        self.grouped= self.df3.groupby('scan')


def mapping(self):
    for i, j in grouped:
      try:
        new_index= j['index'].tolist()
        new_mz= j['mz'].tolist()
        cd= dict(list(zip(j['index'], j['mz'])))
        dc= dict(list(zip(j['mz'], j['index'])))
        ui= np.array(list(map(np.mean, l)))
        ab= np.searchsorted(np.array(j.mz), ui)
        values= np.array(j.mz)
        ab= [x if x !=len(values) else 0 for x in ab]
        values= values[ab]
        mask= np.absolute(np.subtract(values, ui)) < 0.002
        mask= np.array(mask)
        index= np.where(mask)
        values= values[index]
        null= [self.l[x].append(y) for x, y in zip(index[0], values)]
        indexlist= [self.d[x].append(dc[v]) for x, v in zip(index[0], values)]
        notselected_mz= list(set(list(dc.keys())).difference(set(values)))
        notselected_index = [dc[x] for x in notselected_mz]
        last_scan= i
        for i, j in zip(notselected_mz, notselected_index):
          self.l.append([i])
          self.d.append([j])

      except Exception as e:
        print(e)
        last_scan= i
    ui= np.array(list(map(np.mean, self.l)))
    dict1= {k:v for k,v in zip(ui, self.d)}
    lk= list(dict1.keys())
    t, y= roi_seed(lk)
    self.new_dict1= dict()
    for i, j in zip(t, y):
      if len(i)> 1:
        j= []
        for x in i:
          p= dict1[x]
          j.extend(p)
      i= np.mean(np.array(i))
      self.new_dict1[i]= j


    def filter_roi(self):
        self.rm= dict()
        self.mapping()
        for k,v in self.new_dict1.items():
            if len(v) < 5:
                self.rm.update({k:v})
            else:
                testdf= self.df1.iloc[list(v), ]
                scannumber= testdf.scan.tolist()
                intensity= testdf.intensity.tolist()
                intensity = sorted(intensity, reverse=True)
                if intensity[4] < 1000:
                    self.rm.update({k:v})
                if len(scannumber)/last_scan > 0.5:
                    self.rm.update({k:v})



if __name__ == "__main__":
    obj123= Roi('/home/fs71579/skumari/699_ph1.csv')
    obj123.filter_roi()
    for l,m in obj123.rm.items():
       obj123.new_dict1.pop(l)
    import json
    with open('convertfinal2.txt', 'w') as convert_file:
         convert_file.write(json.dumps(obj123.new_dict1))
    with open('rmfinal2.txt', 'w') as convert_file:
         convert_file.write(json.dumps(obj123.rm))
