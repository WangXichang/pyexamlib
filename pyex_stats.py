# -*- utf-8 -*-
# version 2017-09-16

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from scipy import stats
import pyex_lib as pl
import pyex_seg as ps


# ywmean = df.yw.mean(), wlmean = df.wl.mean()
# m = math.sqrt(sum([(x - ywmean)**2 for x in df.yw]))*math.sqrt(sum([(x - wlmean)**2 for x in df.wl]))
# c = sum([(x - ywmean)*(y - wlmean) for x, y in zip(df.yw, df.wl)])
# pearsonr = c/m
def show_relation(x, y):
    plt.scatter(x, y)
    plt.title(str(stats.pearsonr(x, y)[0]))
    plt.xlabel('first var')
    plt.ylabel('second var')
    return


def float_str(x, d1, d2):
    return f'%{d1}.{d2}f' % x


def int_str(x,d):
    return f'%{d}d' % x


def exp_r(noise=10):
    """
    show a regression example
    deviation 10, and R is big
    """
    tf = pl.exp_scoredf_normal(mean=60, std=10, size=1000)
    tf['sf2'] = tf.sf.apply(lambda v: v + np.random.rand()*noise)
    rs = stats.pearsonr(tf.sf, tf.sf2)
    maxdiff = max(abs(tf.sf - tf.sf2))
    plt.scatter(tf.sf, tf.sf2, label='relation')
    plt.title('noise={n}   PearsonR={r}   MaxDiff={d}'.
              format(n=noise, r=float_str(rs, 2, 4), d=float_str(maxdiff, 2, 4)))


class gkdf():
    """
    read gk17 data from csv
    include kl, ysw, wl, hx, sw
    """
    def __init__(self):
        self.df17lk = None
        self.df17wk = None
        self.wkdatafile = 'f:/studies/xkdata/gkscore/wzxj2017.csv'
        self.lkdatafile = 'f:/studies/xkdata/gkscore/lzxj2017.csv'

    def read_17lk(self):
        self.df17lk = pd.read_csv(self.lkdatafile, # 'd:/work/newgk/gkdata/lkcj17.csv', sep='\t',
                              usecols=['wl', 'hx', 'sw'])
        self.df17lk.loc[:, 'wl100'] = self.df17lk['wl'].apply(lambda x: self.xround(x*10/11))
        self.df17lk.loc[:, 'sw100'] = self.df17lk['sw'].apply(lambda x: self.xround(x*10/9))
        self.smoothdata(self.df17lk, 'wl100')
        self.smoothdata(self.df17lk, 'sw100')

    def read_17wk(self):
        self.df17wk = pd.read_csv(self.wkdatafile,
                                 usecols=['zz', 'ls', 'dl'])
        self.df17wk = self.df17wk.applymap(lambda x: self.xround(x))
        self.smoothdata(self.df17wk, 'dl')

    def xround(self,x):
        if np.random.randint(0, 1000) % 2 ==0:
            return int(np.floor(x))
        else:
            return int(np.ceil(x))

    def smoothdata(self,df, field):
        tplist = df[field].tolist()
        for i in range(len(tplist)):
            if (i > 0) & (tplist[i]>2000):
                if tplist[i] < tplist[i-1] * 1.2:
                    tplist[i] = tplist[i-1]
            if (i > 0) & (i < len(df)-1):
                if (tplist[i] == 0) & (tplist[i-1]*tplist[i+1] != 0):
                    tplist[i]=tplist[i-1]
        df.loc[:, field] = tplist
        return df


class Relation:
    """
    compute pearson and group relation coefficients
    pearsonr_all: pearson and group coefficients
    pearsonr: only pearson coefficients
    group_r only group coefficients
    """
    def __init__(self):
        self.pearsonr_all = None
        self.pearsonr = None
        self.group_r = dict()
        self._dataframe = None
        self._fields = None
        self._remove_small_samples = True
        self._remove_zero_value = True
        self.outdf = None
        self.meandf = None

    def set_data(self,
                 df: pd.DataFrame,
                 fields: list,
                 remove_small_samples=True,
                 remove_zero_value=True):
        if len(fields) == 0:
            print('must to set field names list!')
            return
        else:
            self._fields = fields
        self._dataframe = df.copy(deep=True)
        self.pearsonr = df[fields].corr()
        self._remove_small_samples = remove_small_samples
        self._remove_zero_value = remove_zero_value

    def _dfmean(self, df, field_value, value, field_mean):
        tdf = df[df[field_value]==value][field_mean]
        return tdf.mean() if tdf.count() > 10 else -1

    def run(self):
        if len(self._fields) == 0:
            print('must to set field names list!')
            return
        # self.seg = ps.SegTable()
        tempdf = self._dataframe[self._fields].copy(deep=True)
        # tempdf = self.df
        meandict = dict()
        maxscoredict = dict()
        for _f in self._fields:
            maxscoredict[_f] = int(self._dataframe[_f].max())
        for i, _f1 in enumerate(self._fields):
            for j, _f2 in enumerate(self._fields):
                if j > i:
                    stime = time.clock()
                    print(f'calculating: {_f1}--{_f2}')
                    meandict[(_f1, _f2)] = [self._dfmean(self._dataframe, _f2, x, _f1) for x in range(maxscoredict[_f2] + 1)]
                    meandict[(_f2, _f1)] = [self._dfmean(self._dataframe, _f1, x, _f2) for x in range(maxscoredict[_f1] + 1)]
                    meanf1_for_f2 = list(map(lambda x: meandict[(_f1, _f2)][x],
                                             self._dataframe[_f2].values.astype(int)))
                    meanf2_for_f1 = list(map(lambda x: meandict[(_f2, _f1)][x],
                                             self._dataframe[_f1].values.astype(int)))
                    tempdf['gmean_'+_f1 + '_for_' + _f2] = meanf1_for_f2
                    tempdf['gmean_'+_f2 + '_for_' + _f1] = meanf2_for_f1
                    self.group_r[_f1 + '_gr_' + _f2] = stats.pearsonr(tempdf[_f1].values, meanf2_for_f1)[0]
                    self.group_r[_f2 + '_gr_' + _f1] = stats.pearsonr(tempdf[_f2].values, meanf1_for_f2)[0]
                    print(f'consume time = {round(time.clock()-stime,2)}')
        if self._remove_small_samples:
            self.pearsonr_all = tempdf[tempdf > 0].corr()
        else:
            self.pearsonr_all = tempdf[tempdf != 0].corr()
        self.outdf = tempdf
        self.meandf = meandict
        return

def group_relation(df, f1, f2, nozero=True):
    """
    :param df: input dataframe
    :param f1: scorefield1
    :param f2: scorefield2
    :param nozero:  remove zero values in field1, field2
    :return: dict['f1_f2_grouprelation',  'f2_f1_grouprelation']
    """
    if nozero:
        df = df[(df[f1] > 0) & (df[f2] > 0)]
    f1scope = [int(df[f1].min()), int(df[f1].max())]
    f2scope = [int(df[f2].min()), int(df[f2].max())]
    df1 = pd.DataFrame({f2+'_mean': [df[df[f1] == x][f2].mean() for x in range(*f1scope)],
                        f1: [x for x in range(*f1scope)]})
    df2 = pd.DataFrame({f1+'_mean': [df[df[f2] == x][f1].mean() for x in range(*f2scope)],
                        f2: [x for x in range(*f2scope)]})
    df1.fillna(0, inplace=True)
    df2.fillna(0, inplace=True)
    r = dict()
    r[f1+'_'+f2] = stats.pearsonr(df1[f1], df1[f2+'_mean'])[0]
    r[f2+'_'+f1] = stats.pearsonr(df2[f2], df2[f1+'_mean'])[0]
    return r
