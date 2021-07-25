# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by ben 2020/1/13 下午3:33
import random
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from collections import Iterable

def dictData(datas):
    dictDatas = {i: data for i, data in enumerate(datas)}
    return dictDatas

def tupleData(datas):
    tupleDatas = [(i, data) for i, data in enumerate(datas)]
    return tupleDatas

def tupleDf(df):
    datas = [tuple(data) for data in df.values]
    return datas

def tupleDictData(datas):
    tupleDatas = [(k, v) for k, v in datas.items()]
    return tupleDatas

# ========================================

def df_split(df, splits, seed=None):
    datas = data_split(df.to_numpy(), splits, seed=seed)
    dfs = [pd.DataFrame(data, columns=df.columns) for data in datas]
    return dfs

def data_split(data, splits, seed=None):
    dataSize = len(data)
    splits = np.array(splits)
    if np.sum(splits) == 1.0:
        splits_int = [int(dataSize * i) for i in splits]
        splits_int[-1] = dataSize - np.sum(splits_int[:-1])
    elif np.sum(splits) == dataSize:
        splits_int = splits
    else:
        raise Exception(f'sum of splits is not 1 or {dataSize}')
    splits_int = np.array(splits_int)
    splits_int_accumulative = [0] + [np.sum(splits_int[:i + 1]) for i in range(len(splits))]
    ids = list(range(dataSize))
    random.seed(seed)
    random.shuffle(ids)
    datas = []
    for i in range(len(splits)):
        iData = []
        for j in range(splits_int_accumulative[i], splits_int_accumulative[i + 1]):
            iData.append(data[ids[j]])
        datas.append(iData)
    return datas

    # ========================================

def df_addCol(df, col, data, dropna=False):
    df[col] = pd.Series(data)
    if dropna:
        df.dropna(inplace=True)
    return df

def df_apply(df, col, func, newCol=None, cpu=1, dropna=False, bar=False, **kwargs):
    datas_col = df[col].to_list()
    datas_newCol = multi(datas_col, func, cpu=cpu, filterNone=False, bar=bar, **kwargs)
    if newCol:
        df[newCol] = datas_newCol
    else:
        df[col] = datas_newCol
    if dropna:
        if newCol:
            df.dropna(inplace=True, subset=[newCol])
        else:
            df.dropna(inplace=True, subset=[col])

def df_addName(df, col='name', prefix=None):
    if prefix is None:
        names = [f'{i}' for i in range(df.shape[0])]
    else:
        names = [f'{prefix}_{i}' for i in range(df.shape[0])]
    df.insert(0, col, names)
    return df

def df_valueCounts(df, col):
    counts = dict(df[col].value_counts())
    df[f'{col}_repeat'] = [counts[k] for k in df[col].to_list()]
    return df

def df_group(df, col):
    ks = set(df[col].to_list())
    group = df.groupby(col)
    dfs = {}
    for k in ks:
        dfs[k] = group.get_group(k)
    return dfs

# ========================================
def multi(datas, func, cpu=None, chunksize=1, filterNone=True, bar=False, tupleK=False, **kwargs):
    if tupleK:
        return __multiProcess_tupleK(datas, func, cpu=cpu, chunksize=chunksize, filterNone=filterNone, bar=bar, **kwargs)
    elif type(datas) is dict:
        return __multiProcess_dict(datas, func, cpu=cpu, chunksize=chunksize, filterNone=filterNone, bar=bar, **kwargs)
    elif isinstance(datas, Iterable):
        return __multiProcess_list(datas, func, cpu=cpu, chunksize=chunksize, filterNone=filterNone, bar=bar, **kwargs)
    else:
        raise Exception('multi,data type is not iterable')

def multiProcess_batch(datas, func, batchSize=None, cpu=None, filterNone=True, bar=False, yieldMode=None, **kwargs):
    if batchSize == None:
        if cpu == 0:
            batchSize = multiprocessing.cpu_count() * 4
        else:
            batchSize = cpu * 4

    def multiProcess_batch_yield():
        if type(datas) == dict:
            if yieldMode == 'single':
                raise Exception('yieldMode=single is not supported for dict data')
            rs = {}
        elif isinstance(datas, Iterable):
            rs = []
        else:
            raise Exception('datas is not iterable')
        for batchData in getBatchData(datas, batchSize, bar=bar):
            batchR = multi(batchData, func, cpu=cpu, chunksize=1, filterNone=filterNone, bar=False, **kwargs)
            if yieldMode == 'single':
                for r in batchR:
                    yield r
            elif yieldMode == 'batch':
                yield batchR
            elif yieldMode is None:
                if type(rs) == dict:
                    rs.update(batchR)
                else:
                    rs.extend(batchR)
            else:
                raise Exception(f'yieldMode {yieldMode} is not supported')
        if yieldMode is None:
            yield rs

    if yieldMode is None:
        return next(multiProcess_batch_yield())
    else:
        return multiProcess_batch_yield()

def __multiProcess_list(datas, func, cpu=None, chunksize=1, filterNone=True, bar=False, **kwargs):
    if cpu is None:
        cpu = 0
    if kwargs:
        func = partial(func, **kwargs)
    if bar:
        if type(datas) is not list and cpu != 1:
            datas = list(datas)
        datas = tqdm(datas)
    if cpu == 1:
        datas = map(func, datas)
    elif cpu == 0:
        with multiprocessing.Pool() as pool:
            datas = pool.map(func, datas, chunksize=chunksize)
    elif cpu > 1:
        with multiprocessing.Pool(cpu) as pool:
            datas = pool.map(func, datas, chunksize=chunksize)
    else:
        raise Exception(f'the number of cpu can not be {cpu} ')
    if filterNone:
        datas = filter(lambda x: x is not None, datas)
    return list(datas)

def __multiProcess_dict(datas, func, cpu=None, chunksize=1, filterNone=True, bar=False, **kwargs):
    ks = datas.keys()
    vs = datas.values()
    vs = __multiProcess_list(vs, func, cpu=cpu, chunksize=chunksize, filterNone=False, bar=bar, **kwargs)
    datas = zip(ks, vs)
    if filterNone:
        def notNone(x):
            if x[1] is None:
                return False
            else:
                return True

        datas = filter(notNone, datas)
    return dict(datas)

def __multiProcess_tupleK(datas, func, cpu=None, chunksize=1, filterNone=True, bar=False, **kwargs):
    ks, vs = zip(*datas)
    vs = __multiProcess_list(vs, func, cpu=cpu, chunksize=chunksize, filterNone=False, bar=bar, **kwargs)
    datas = zip(ks, vs)
    if filterNone:
        datas = filter(lambda x: x[1], datas)
    return list(datas)

def getBatchData(datas, batchSize, bar=False):
    if type(datas) == dict:
        bDatas = {}
        if bar:
            pBar = tqdm(total=len(datas))
        else:
            pBar = None
        for k in datas:
            bDatas[k] = datas[k]
            if bar:
                pBar.update(1)
            if len(bDatas) == batchSize:
                yield bDatas
                bDatas = {}
        if bar:
            pBar.close()
        if len(bDatas) > 0:
            yield bDatas
    elif isinstance(datas, Iterable):
        bDatas = []
        if bar:
            datas = tqdm(datas)
        for data in datas:
            bDatas.append(data)
            if len(bDatas) == batchSize:
                yield bDatas
                bDatas = []
        if len(bDatas) > 0:
            yield bDatas
    else:
        raise Exception('datas is not iterable')

# ========================================
def dataMerge_listOfList(datas):
    datas1 = []
    for data in datas:
        if data is not None:
            for iData in data:
                if iData is not None:
                    datas1.append(iData)
    return datas1

def getRankIndex(data, shift=0, reverse=False):
    if reverse:
        sortIndex = np.argsort(-np.array(data))
    else:
        sortIndex = np.argsort(np.array(data))
    pairs = list(zip(sortIndex, list(range(len(sortIndex)))))
    pairs.sort(key=lambda x: x[0])
    rankIndex = [x[1] + shift for x in pairs]

    return rankIndex
