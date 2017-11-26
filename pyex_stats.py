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
    return stats.pearsonr(x, y)[0]


def float_str(x, d1, d2):
    return f'%{d1}.{d2}f' % x


def int_str(x,d):
    return f'%{d}d' % x


def exp_r(noise=10):
    tf = pl.exp_scoredf_normal(mean=60, std=10, size=1000)
    tf['sf2'] = tf.sf.apply(lambda v: v + np.random.rand()*noise)
    rs = relation(tf.sf, tf.sf2)
    maxdiff = max(abs(tf.sf - tf.sf2))
    #plt.figure()
    plt.scatter(tf.sf, tf.sf2, label='relation')
    plt.title('noise={n}   PearsonR={r}   MaxDiff={d}'.
              format(n=noise, r=float_str(rs, 2, 4), d=float_str(maxdiff, 2, 4)))


class ScoreData():
    """
    read gk data from csv
    include kl, ysw, wl, hx, sw
    """

    def __init__(self):
        self.fs_file = 'd:/work/newgk/shanghai1711/cj17b.csv'
        self.rawdf = None

    def read_rawdf(self, filepath='d:/work/newgk/shanghai1711/cj17b.csv'):
        # self.df = pd.read_csv('d:/work/newgk/shanghai1711/cj17b.csv', sep='\t', index_col=0)
        self.rawdf = pd.read_csv(filepath, sep='\t', index_col=0)
        return


class Relation():
    def __init__(self):
        self.pearson_r = None
        # self.group_r = dict()
        self.df = None
        self.fields = None
        self.maxscore = 150
        self.remove_small_samples = True
        self.remove_zero_value = True
        self.outdf = None

    def set_data(self, df, fields='', remove_small_samples=True, remove_zero_value=True):
        if len(fields) == 0:
            #self.fields = df.columns.values
            print('must to set field names list!')
            return
        else:
            self.fields = fields
        self.df = df.copy(deep=True)
        self.remove_small_samples = remove_small_samples
        self.remove_zero_value = remove_zero_value

    def dfmean(self, df, field_value, value, field_mean):
        tdf = df[df[field_value]==value][field_mean]
        return tdf.mean() if tdf.count() > 10 else -1

    def run(self):
        if len(self.fields) == 0:
            print('must to set field names list!')
            return
        self.seg = ps.SegTable()
        tempdf = self.df[self.fields].copy(deep=True)
        # tempdf = self.df
        meandict = dict()
        maxscoredict = dict()
        for _f in self.fields:
            # print(_f)
            maxscoredict[_f] = int(self.df[_f].max())
        for i, _f1 in enumerate(self.fields):
            for j, _f2 in enumerate(self.fields):
                if j > i:
                    stime = time.clock()
                    print(f'calculating: {_f1}--{_f2}')
                    '''
                    meandict[(_f1, _f2)] = [self.df[self.df[_f2] == x][_f1].mean()
                                            for x in range(maxscoredict[_f2] + 1)]
                    meandict[(_f2, _f1)] = [self.df[self.df[_f1] == x][_f2].mean()
                                            for x in range(maxscoredict[_f1] + 1)]'''
                    meandict[(_f1, _f2)] = [self.dfmean(self.df, _f2, x, _f1) for x in range(maxscoredict[_f2]+1)]
                    meandict[(_f2, _f1)] = [self.dfmean(self.df, _f1, x, _f2) for x in range(maxscoredict[_f1]+1)]
                    f1 = lambda x: meandict[(_f1, _f2)][x]
                    f2 = lambda x: meandict[(_f2, _f1)][x]
                    meanf1_for_f2 = list(map(f1, self.df[_f2].values.astype(int)))
                    meanf2_for_f1 = list(map(f2, self.df[_f1].values.astype(int)))
                    tempdf['gmean_'+_f1 + '_for_' + _f2] = meanf1_for_f2
                    tempdf['gmean_'+_f2 + '_for_' + _f1] = meanf2_for_f1
                    # self.group_r[_f1 + '_gr_' + _f2] = stats.pearsonr(tempdf[_f1].values, meanf2_for_f1)[0]
                    # self.group_r[_f2 + '_gr_' + _f1] = stats.pearsonr(tempdf[_f2].values, meanf1_for_f2)[0]
                    print(f'consume time = {round(time.clock()-stime,2)}')
        if self.remove_small_samples:
            self.pearson_r = tempdf[tempdf > 0].corr()
        else:
            self.pearson_r = tempdf[tempdf != 0].corr()
        self.outdf = tempdf
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
    # r['df_'+f1+'_'+f2+'_mean'] = df1
    # r['df_'+f2+'_'+f1+'_mean'] = df2
    return r


def df_format(dfsource, intlen=2, declen=4, strlen=8):
    df = dfsource[[dfsource.columns[0]]]
    fdinfo = dfsource.dtypes
    for fs in fdinfo.index:
        if fdinfo[fs] in [np.float, np.float16, np.float32, np.float64]:
            df[fs+'_str'] = dfsource[fs].apply(lambda x: float_str(x, intlen, declen))
        elif fdinfo[fs] in [np.int, np.int8, np.int16, np.int32, np.int64]:
            df[fs+'_str'] = dfsource[fs].apply(lambda x: int_str(x, 6))
        elif fdinfo[fs] in [str]:
            df[fs+'_fmt'] = dfsource[fs].apply(lambda x: x.rjust(strlen))
    df.sort_index(axis=1)
    return df


def ref_stm(df, fkey, f1, f2, adj_rate_points=(0.35, 0.75)):
    segmodel = ps.SegTable()
    segmodel.set_data(df, [f1, f2])
    segmodel.set_parameters(segmax=max(df[f1]))
    segmodel.run()
    segf1 = segmodel.segdf
    segmodel.set_parameters(segmax=max(df[f2]))
    segf2 = segmodel.segdf
    f1points = []
    for p in adj_rate_points:
        f2count = segf2.loc[segf2[f2+'_percent'] >= p, f2+'_cumsum'].head(1)['seg']


def cross_seg(source_dataframe,
              key_field,
              cross_field,
              cross_seg_list=(50, 60, 70, 80, 90, 100)):
    segmodel = ps.SegTable()
    segmodel.set_data(source_dataframe, key_field)
    segmodel.set_parameters(segmax=max(source_dataframe[key_field]))
    segmodel.run()
    dfseg = segmodel.segdf
    dfcount = dfseg[key_field + '_cumsum'].tail(1).values[0]
    vfseg = {x:[] for x in cross_seg_list}
    vfper = {x:[] for x in cross_seg_list}
    seglen = dfseg['seg'].count()
    for sv, step in zip(dfseg['seg'], range(seglen)):
        if (step % 20 == 0) | (step == seglen-1):
            print('='* int((step+1)/seglen * 30) + '>>' + f'{float_str((step+1)/seglen, 1, 2)}')
        segv = []
        for vfv in cross_seg_list:
            segcount = source_dataframe.loc[(source_dataframe[key_field] >= sv) & (source_dataframe[cross_field] >= vfv), cross_field].count()
            vfseg[vfv].append(segcount)
            vfper[vfv].append(segcount/dfcount)
    for vs in cross_seg_list:
        dfseg[cross_field + str(vs) + '_cumsum'] = vfseg[vs]
        dfseg[cross_field + str(vs) + '_percent'] = vfper[vs]
    return dfseg


