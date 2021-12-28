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
from scipy import stats
import matplotlib.pyplot as plt
import scipy
from scipy import stats


def get_area(x, y):
    area = trapz(y, x)
    return max(0.0, area)


def find_figure(x, y, z, a, b):
    fig= plt.figure()
    plt.plot(a, b, 'x')
    fig.savefig(f'{z}.png')


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
    sn: int
    peak_gaussian: str


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
        print(self.roi.keys())
        self.b= np.array([10, 100, 1000, 2000, 200, 20])
        self.test= False



    def peakchoose(self, i):
        right= self.right_bases[i]
        left= self.left_bases[i]
        df3= self.df2[(self.df2.rt >= left) & (self.df2.rt <= right)]
        mz= np.array(df3.mz)
        if df3.index.size > 8:
            intensity= np.array(df3.intensity)
            rt= np.array(df3.rt)
            interpolator = interpolate.interp1d(rt, intensity, kind= 'linear')
            b = interpolator(np.arange(rt.min(), rt.max(), 1))
            x= filter(b)
            diff= b-x
            B = np.where(diff > 0)
            noise= diff[B]
            noise= noise.mean()
            signal= intensity.max()
            sn= int(signal/noise)
            mz= np.array(df3.mz)
            index= np.argmax(intensity)
            k2, p = stats.normaltest(intensity)
            alpha = 0.5
            p= 'gaussian' if p > alpha else 'Not_guassian'
            rtmed= rt[index]
            mzmed= mz[index]
            mzmin= mz.min()
            mzmax= mz.max()
            rtmin= rt.min()
            rtmax= rt.max()
            area= get_area(rt, intensity)
            return PeakData(mzmin, mzmax, mzmed, rtmin, rtmax, rtmed, area, sn, p)



    def pick(self, ll):
        try:
            if self.test:
                df2=self.rawdata[self.rawdata.index.isin(ll)]
                name= df2.mz.tolist()[0]
                outlier_datapoints = detect_outlier(df2.scan.tolist())
                self.df2= df2[~df2['scan'].isin(outlier_datapoints)]
                tt= np.array(self.df2.rt)
                yy= np.array(self.df2.intensity)
                self.interpolator = interpolate.interp1d(tt, yy, kind= 'linear')
                self.b = interpolator(np.arange(tt.min(), tt.max(), 1))

            else:
                df2=self.rawdata[self.rawdata.index.isin(ll)]
                name= df2.mz.tolist()[0]
                outlier_datapoints = detect_outlier(df2.scan.tolist())
                self.df2= df2[~df2['scan'].isin(outlier_datapoints)]
                tt= np.array(self.df2.rt)
                yy= np.array(self.df2.intensity)
                interpolator = interpolate.interp1d(tt, yy, kind= 'linear')
                b = interpolator(np.arange(tt.min(), tt.max(), 1))

            result = stats.kruskal(self.b, b)
            print(result)
            if result.pvalue > 0.0005:
                rr= np.arange(tt.min(), tt.max(), 1)
                find_figure(rr, b, name, tt, yy)
                x= filter(b)
                diff= b-x
                B = np.where(diff > 0)
                self.smooth= diff[B]
                first= self.smooth.min()
                prominence= first*2
                peaks, cx = find_peaks(x, prominence= prominence, wlen= 39.1)
                print(cx)
                return peaks, cx

        except:
            return np.array([]), 1


    def result(self):
        object_list= []
        for i, j in self.roi.items():
            self.test= True if i == "162.99727494969818" else False
            if self.pick(j):
                peaks, cx= self.pick(j)
                if peaks.size > 0:
                    self.right_bases= cx['right_bases']
                    self.left_bases= cx['left_bases']
                    finaldata= list(map(self.peakchoose, range(len(peaks))))
                    object_list.extend(finaldata)
            objectnew= [asdict(x) for x in object_list if isinstance(x, PeakData)]
            df= pd.DataFrame(objectnew)
            df.to_csv('s1.csv')


if __name__ == "__main__":
    obj123= Peaksworking('parsed_raw_data.csv', 'convert1.txt')
    obj123.result()
