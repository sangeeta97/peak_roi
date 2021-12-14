import json
import pandas as pd
import numpy as np
import re
from scipy.signal import find_peaks, peak_prominences
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from typing import Callable, Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from dataclasses import asdict


def get_area(x, y):
    area = trapz(y, x)
    return max(0.0, area)


# A class for holding an employees content
@dataclass
class PeakData:
    mzmin: float
    mzmax: float
    mzmed: float
    rtmin: float
    rtmax: float
    rtmed: float
    area: float


outliers=[]
def detect_outlier(data_1):
    threshold=4
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    for y in data_1:
        z_score= (y - mean_1)/std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


def filter(xx):
    return savgol_filter(xx, 5, 2)



class Peaksworking():
    def __init__(self, path1, path2):
        super().__init__()
        self.rawdata = pd.read_csv(path1)
        self.roi= open(path2)
        self.roi= json.load(self.roi)


    def peakchoose(self, i):
        right= right_bases[i]
        left= left_bases[i]
        df3= df2[(df2.rt >= left) & (df2.rt <= right)]
        if df3.index.size > 3:
            intensity= np.array(df3.intensity)
            rt= np.array(df3.rt)
            mz= np.array(df3.mz)
            index= np.argmax(intensity)
            rtmed= rt[index]
            mzmed= mz[index]
            mzmin= mz.min()
            mzmax= mz.max()
            rtmin= rt.min()
            rtmax= rt.max()
            area= get_area(rt, intensity)
            return PeakData(mzmin, mzmax, mzmed, rtmin, rtmax, rtmed, area)



    def pick(self, ll):
        try:
            df2=self.rawdata[self.rawdata.index.isin(ll)]
            outlier_datapoints = detect_outlier(df2.scan.tolist())
            df2= df2[~df2['scan'].isin(outlier_datapoints)]
            tt= np.array(df2.rt)
            yy= np.array(df2.intensity)
            interpolator = interpolate.interp1d(tt, yy, kind= 'linear')
            b = interpolator(np.arange(tt.min(), tt.max(), 1))
            rr= np.arange(tt.min(), tt.max(), 1)
            x= filter(b)
            diff= np.absolute(b-x)
            noise= np.mean(diff)
            prominence= noise/4
            peaks, cx = find_peaks(x, prominence= prominence, wlen= 39.1)
            return peaks, cx

        except:
            pass


    def result(self):
        object_list= []
        for i in self.roi.values():
            peaks, cx= self.pick(i)
            if peaks.size > 0:
                right_bases= cx['right_bases']
                left_bases= cx['left_bases']
                finaldata= list(map(self.peakchoose, range(len(peaks))))
        object_list.extend(finaldata)
        objectnew= [asdict(x) for x in object_list if isinstance(x, PeakData)]
        df= pd.DataFrame(objectnew)
        df.to_csv('/home/fs71579/skumari/selected_peaks4.csv')


if __name__ == "__main__":
    obj123= Peaksworking('/home/fs71579/skumari/699_ph1.csv', '/home/fs71579/skumari/convertfinal.txt')
    obj123.result()
