# -*- utf-8 -*-
# version 2017-09-16

import pandas as pd
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
        self.std_20 = (20, 30, 45, 60, 75, 90, 100)
        self.std_30 = (30, 38, 41, 65, 79, 92, 100)
        self.std_40 = (40, 48, 59, 70, 81, 92, 100)

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

    def test_shift(self,
                   df,
                   fieldlist=['wl', 'hx', 'sw'],
                   stdpoints=[20, 30, 45, 60, 75, 90, 100]
                   ):
        mddict = dict()
        for fs in fieldlist:
            print(f'---< {fs} >---')
            md = stm.test_model(df=df, fieldnames=fs, stdpoints=stdpoints)
            mddict[fs] = md
        return mddict
