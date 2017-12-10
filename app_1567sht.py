# -*- utf-8 -*-
# version 2017-09-16

import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
# import time
import pyex_stm as stm
import glob


class Data:
    def __init__(self):
        self.pf15 = 'f:/studies/xkdata/gkscore/2015/csv/*.csv'
        self.pf16 = 'f:/studies/xkdata/gkscore/2016/csv/*.csv'
        self.pf17 = 'f:/studies/xkdata/gkscore/2017/*.csv'
        self.df17wk = None
        self.df17lk = None
        self.df16wk = None
        self.df16lk = None
        self.df15wk = None
        self.df15lk = None
        self.std_00 = [0, 15, 30, 50, 70, 85, 100]
        self.std_20 = [20, 30, 45, 60, 75, 90, 100]
        self.std_30 = [30, 38, 41, 65, 79, 92, 100]
        self.std_40 = [40, 48, 59, 70, 81, 92, 100]

        self.stdpoints00 = [0, 10, 30, 50, 70, 90, 100]  # std=20
        self.stdpoints20 = [20, 28, 44, 60, 76, 92, 100]  # std=16
        self.stdpoints30 = [30, 43, 51, 65, 79, 93, 100]  # std=14
        self.stdpoints40 = [40, 46, 58, 70, 82, 94, 100]  # std=12

    def set_office_data(self):
        self.pf15 = 'f:/studies/xkdata/gkscore/2015/csv/*.csv'
        self.pf16 = 'f:/studies/xkdata/gkscore/2016/csv/*.csv'
        self.pf17 = 'f:/studies/xkdata/gkscore/2017/*.csv'

    def set_dell_data(self):
        self.pf15 = 'd:/work/newgk/gkdata/2015/*.csv'
        self.pf16 = 'd:/work/newgk/gkdata/2016/*.csv'
        self.pf17 = 'd:/work/newgk/gkdata/2017/*2017.csv'

    def read_data(self):
        fs = glob.glob(self.pf15)
        if len(fs) > 0:
            print(f'reading {fs}')
            self.df15lk = pd.read_csv(fs[0])
            self.df15lk['wl100'] = self.df15lk.wlcj.apply(lambda x: int(x*10/11))
            self.df15lk['sw100'] = self.df15lk.swcj.apply(lambda x: int(x*10/9))

            self.df15wk = pd.read_csv(fs[1])
            self.df15wk = self.df15wk.applymap(lambda x: int(x))
            self.df15wk.applymap(lambda x: int(x))

        fs = glob.glob(self.pf16)
        if len(fs) > 0:
            print(f'reading {fs}')
            self.df16lk = pd.read_csv(fs[0])
            self.df16lk['wl100'] = self.df16lk.wlcj.apply(lambda x: int(x*10/11))
            self.df16lk['sw100'] = self.df16lk.swcj.apply(lambda x: int(x*10/9))

            self.df16wk = pd.read_csv(glob.glob(self.pf16)[1])
            self.df16wk = self.df16wk.applymap(lambda x: int(x))
            self.df16wk.applymap(lambda x: int(x))

        fs = glob.glob(self.pf17)
        if len(fs) > 0:
            print(f'reading {fs}')
            self.df17lk = pd.read_csv(glob.glob(self.pf17)[0])
            self.df17lk['wl100'] = self.df17lk.wl.apply(lambda x: int(x*10/11))
            self.df17lk['sw100'] = self.df17lk.sw.apply(lambda x: int(x*10/9))

            self.df17wk = pd.read_csv(glob.glob(self.pf17)[1])
            self.df17wk = self.df17wk.applymap(lambda x: int(x))
            self.df17wk.applymap(lambda x: int(x))

    @staticmethod
    def test_shift(df,
                   fieldlist=('wl', 'hx', 'sw'),
                   stdpoints=(20, 30, 45, 60, 75, 90, 100),
                   rawpoints=[0, .15, .30, .50, .70, .85, 1.00]
                   ):
        # rawpoints = [0, 0.0229, 0.1596, 0.50, 0.8423, 0.9774, 1.00]
        # rawpoints = [0, 0.02275, 0.158655, 0.5, 0.841345, 0.97725, 1.00]   # from scipy.stats.norm(0,1).sf
        mddict = dict()
        for fs in fieldlist:
            print(f'---< {fs} >---')
            # md = stm.test_model(df=df, fieldnames=fs, stdpoints=list(stdpoints), rawpoints=rawpoints)
            md = stm.test_model(df=df,
                                fieldnames=fs,
                                stdpoints=stdpoints,
                                rawpoints=rawpoints)
            mddict[fs] = md
        return mddict

    def test_shift_all(self, df, field: str, newname:str):
        # set to 1 std step in stdscore
        rawpoints1 = [0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.00]
        # rawpoints1 = [0, 0.02275, 0.158655, 0.5, 0.841345, 0.97725, 1.00]
        # rawpoints = [0, .15, .30, .50, .70, .85, 1.00]    # for 0.5 std step
        mdd00 = Data.test_shift(df, fieldlist=[field],
                                stdpoints=self.stdpoints00,
                                rawpoints=rawpoints1
                                )[field]
        mdd20 = Data.test_shift(df, fieldlist=[field],
                                stdpoints=self.stdpoints20,
                                rawpoints=rawpoints1
                                )[field]
        mdd30 = Data.test_shift(df, fieldlist=[field],
                                stdpoints=self.stdpoints30,
                                rawpoints=rawpoints1
                                )[field]
        mdd40 = Data.test_shift(df, fieldlist=[field],
                                stdpoints=self.stdpoints40,
                                rawpoints=rawpoints1
                                )[field]
        import pyex_seg as psg
        seg = psg.SegTable()
        seg.set_parameters(segmax=100, segmin=1)
        seg.set_data(mdd00.outdf[[field, field+'_plt']].apply(round).astype(int), [field, field+'_plt'])
        seg.run()
        dfseg = seg.segdf[['seg', field+'_count', field+'_plt_count']].copy(deep=True)
        # dfseg.drop(labels=[field+'_cumsum', field+'_percent',
        #                   field+'_plt_cumsum', field+'_plt_percent'], axis=1, inplace=True)

        seg.set_data(mdd20.outdf[[field+'_plt']].apply(round).astype(int), field+'_plt')
        seg.run()
        dfseg[field+'_20_count'] = seg.segdf[field+'_plt_count']

        seg.set_data(mdd30.outdf[[field+'_plt']].apply(round).astype(int), field+'_plt')
        seg.run()
        dfseg[field+'_30_count'] = seg.segdf[field+'_plt_count']

        seg.set_data(mdd40.outdf[[field+'_plt']].apply(round).astype(int), field+'_plt')
        seg.run()
        dfseg[field+'_40_count'] = seg.segdf[field+'_plt_count']

        dfdict = {field+'_count': newname+'_raw', field+'_plt_count': newname+'_0_100',
                  field + '_20_count': newname + '_20_100', field + '_30_count': newname + '_30_100',
                  field + '_40_count': newname + '_40_100'
                  }
        dfseg = dfseg[list(dfdict.keys())]
        dfseg.rename(columns=dfdict, inplace=True)
        return (dfseg, mdd00, mdd20, mdd30, mdd40)

    def smooth_field(self, df, field, scope:list):
        lastindex = df.index[0]
        for index, row in df.iterrows():
            if index in range(scope[0], scope[1]):
                if row[field] == 0:
                    df.loc[index, field] = df.loc[lastindex, field]
                    print(index,lastindex)
            lastindex = index
        # remove peak
        for index, row in df.iterrows():
            if index in range(scope[0]+1, scope[1]-1):
                if row[field] > df.loc[index-1,field]*1.5:
                    df.loc[index, field] = int((df.loc[index-1,field] + df.loc[index+1,field])/2)
                if row[field] > df.loc[index+1,field]*1.5:
                    df.loc[index, field] = int((df.loc[index-1,field] + df.loc[index+1,field])/2)

    def plot_df(self, df, fignum=1):
        from matplotlib.font_manager import FontProperties as fp
        font_simhei = fp(fname=r'C:\Windows\Fonts\simhei.ttf',size=14)
        plt.figure(fignum)
        plt.title(u'分数转换结果显示', fontproperties=font_simhei)
        for f in df.columns:
            plt.plot(df.index, df[f])
