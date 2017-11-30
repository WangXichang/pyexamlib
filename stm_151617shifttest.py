# -*- utf-8 -*-
# version 2017-09-16

import pandas as pd
import numpy as np
import time
import pyex_stm as stm

df15lk = pd.read_csv('g:/2015/csv/lzcj.csv')
df15lk['wl100']=df15lk.wlcj.apply(lambda x:int(x*10/11))
df15lk['sw100']=df15lk.swcj.apply(lambda x:int(x*10/9))

df15wk = pd.read_csv('g:/2015/csv/wzcj.csv')
df15wk=df15wk.applymap(lambda x:int(x))
df15wk.applymap(lambda x:int(x))

df16lk = pd.read_csv('g:/2016/csv/lzcj.csv')
df16lk['wl100']=df16lk.wlcj.apply(lambda x:int(x*10/11))
df16lk['sw100']=df16lk.swcj.apply(lambda x:int(x*10/9))

df16wk = pd.read_csv('g:/2016/csv/wzcj.csv')
df16wk=df16wk.applymap(lambda x:int(x))
df16wk.applymap(lambda x:int(x))

df17lk = pd.read_csv('f:/studies/xkdata/gkscore/2017/lzxj2017.csv')
df17lk['wl100']=df17lk.wl.apply(lambda x:int(x*10/11))
df17lk['sw100']=df17lk.sw.apply(lambda x:int(x*10/9))

df17wk = pd.read_csv('f:/studies/xkdata/gkscore/2017/wzxj2017.csv')
df17wk=df17wk.applymap(lambda x:int(x))
df17wk.applymap(lambda x:int(x))

def test_shift(df,
               fieldlist=['wl', 'hx', 'sw'],
               stdpoints=[20, 30, 45, 60, 75, 90, 100]):
    for fs in fieldlist:
        print(f'---km={fs}---')
        md = stm.test_model(df=df, fieldnames=fs, stdpoints=stdpoints)
