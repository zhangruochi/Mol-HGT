# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben  2020/3/13 上午9:40
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, \
    mean_absolute_error, precision_recall_curve, auc, mean_squared_error, r2_score
import numpy as np

try:
    import tensorflow as tf
except:
    pass

def maskNan(yPred, yMask, logger=None):
    if logger and np.sum(np.isnan(yPred)) > 0:
        logger.warn('yPred existed nan,mask')
    yMask = np.clip(yMask - np.isnan(yPred), 0, 1)
    return yMask

def prc_auc_score(yTrue, yPred):
    precision, recall, _ = precision_recall_curve(yTrue, yPred)
    prc_auc = auc(recall, precision)
    return prc_auc

def root_mean_squared_error(yTrue, yPred):
    rmse = mean_squared_error(yTrue, yPred) ** 0.5
    return rmse

def fn_roc_auc(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.int64)
    yPred = np.array(yPred, dtype=np.float64)
    yMask = np.array(yMask, dtype=np.int64)
    # =====================================
    roc_auc_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        if np.mean(yMask_i) > 0.0:
            yTrue_i = yTrue[:, i][yMask_i != 0]
            yPred_i = yPred[:, i][yMask_i != 0]
            # ----------------------------------------
            yTrue_mean = np.mean(yTrue_i)
            if yTrue_mean > 0.0 and yTrue_mean < 1.0:
                roc_auc_list.append(roc_auc_score(yTrue_i, yPred_i))
    if len(roc_auc_list) > 0:
        roc_auc = np.mean(roc_auc_list)
    else:
        roc_auc = 0.0
    return roc_auc

def fn_accuracy(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.int64)
    yPred = np.array(yPred, dtype=np.float64)
    yPred_int = np.int64(yPred > 0.5)
    yMask = np.array(yMask, dtype=np.int64)
    # =====================================
    accuracy_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        if np.mean(yMask_i) > 0.0:
            yTrue_i = yTrue[:, i][yMask_i != 0]
            yPred_int_i = yPred_int[:, i][yMask_i != 0]
            accuracy_list.append(accuracy_score(yTrue_i, yPred_int_i))
    if len(accuracy_list) > 0:
        accuracy = np.mean(accuracy_list)
    else:
        accuracy = 0.0
    return accuracy

def fn_prc_auc(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.int64)
    yPred = np.array(yPred, dtype=np.float64)
    yMask = np.array(yMask, dtype=np.int64)
    # =====================================
    prc_auc_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        if np.mean(yMask_i) > 0.0:
            yTrue_i = yTrue[:, i][yMask_i != 0]
            yPred_i = yPred[:, i][yMask_i != 0]
            yTrue_mean = np.mean(yTrue_i)
            if yTrue_mean > 0.0 and yTrue_mean < 1.0:
                prc_auc_list.append(prc_auc_score(yTrue_i, yPred_i))
    # =====================================
    if len(prc_auc_list) > 0:
        prc_auc = np.mean(prc_auc_list)
    else:
        prc_auc = 0.0
    return prc_auc

def fn_recall(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.int64)
    yPred = np.array(yPred, dtype=np.float64)
    yPred_int = np.int64(yPred > 0.5)
    yMask = np.array(yMask, dtype=np.int64)
    # =====================================
    recall_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        if np.mean(yMask_i) > 0.0:
            yTrue_i = yTrue[:, i][yMask_i != 0]
            yPred_int_i = yPred_int[:, i][yMask_i != 0]
            # ----------------------------------------
            recall_list.append(recall_score(yTrue_i, yPred_int_i))
    if len(recall_list) > 0:
        recall = np.mean(recall_list)
    else:
        recall = 0.0
    return recall

def fn_precision(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.int64)
    yPred = np.array(yPred, dtype=np.float64)
    yPred_int = np.int64(yPred > 0.5)
    yMask = np.array(yMask, dtype=np.int64)
    # =====================================
    precision_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        if np.mean(yMask_i) > 0.0:
            yTrue_i = yTrue[:, i][yMask_i != 0]
            yPred_int_i = yPred_int[:, i][yMask_i != 0]
            precision_list.append(precision_score(yTrue_i, yPred_int_i))
    if len(precision_list) > 0:
        precision = np.mean(precision_list)
    else:
        precision = 0.0
    return precision

def fn_f1(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.int64)
    yPred = np.array(yPred, dtype=np.float64)
    yPred_int = np.int64(yPred > 0.5)
    yMask = np.array(yMask, dtype=np.int64)
    # =====================================
    f1_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        if np.mean(yMask_i) > 0.0:
            yTrue_i = yTrue[:, i][yMask_i != 0]
            yPred_int_i = yPred_int[:, i][yMask_i != 0]
            f1_list.append(f1_score(yTrue_i, yPred_int_i))
    if len(f1_list) > 0:
        f1 = np.mean(f1_list)
    else:
        f1 = 0.0
    return f1

def fn_rmse(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.float64)
    yPred = np.array(yPred, dtype=np.float64)
    yMask = np.array(yMask, dtype=np.int64)
    rmse_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        yTrue_i = yTrue[:, i][yMask_i != 0]
        yPred_i = yPred[:, i][yMask_i != 0]
        if np.mean(yMask_i) > 0.0:
            rmse_list.append(root_mean_squared_error(yTrue_i, yPred_i))
    if len(rmse_list) > 0:
        rmse = np.mean(rmse_list)
    else:
        rmse = 1e3
    return rmse

def fn_mae(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.float64)
    yPred = np.array(yPred, dtype=np.float64)
    yMask = np.array(yMask, dtype=np.int64)
    mae_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        yTrue_i = yTrue[:, i][yMask_i != 0]
        yPred_i = yPred[:, i][yMask_i != 0]
        if np.mean(yMask_i) > 0.0:
            mae_list.append(mean_absolute_error(yTrue_i, yPred_i))
    if len(mae_list) > 0:
        mae = np.mean(mae_list)
    else:
        mae = 1e3
    return mae

def fn_r2(yTrue, yPred, yMask):
    yTrue = np.array(yTrue, dtype=np.float64)
    yPred = np.array(yPred, dtype=np.float64)
    yMask = np.array(yMask, dtype=np.int64)
    r2_list = []
    for i in range(yTrue.shape[1]):
        yMask_i = yMask[:, i]
        yTrue_i = yTrue[:, i][yMask_i != 0]
        yPred_i = yPred[:, i][yMask_i != 0]
        if np.mean(yMask_i) > 0.0:
            r2_list.append(r2_score(yTrue_i, yPred_i))
    if len(r2_list) > 0:
        r2 = np.mean(r2_list)
    else:
        r2 = -1.0
    return r2

metric_fns = \
    {'roc_auc': fn_roc_auc,
     'accuracy': fn_accuracy,
     'prc_auc': fn_prc_auc,
     'recall': fn_recall,
     'precision': fn_precision,
     'f1': fn_f1,
     'rmse': fn_rmse,
     'mae': fn_mae,
     'r2': fn_r2}

metric_betters = \
    {'roc_auc': 'max',
     'accuracy': 'max',
     'prc_auc': 'max',
     'recall': 'max',
     'precision': 'max',
     'f1': 'max',
     'rmse': 'min',
     'mae': 'min',
     'r2': 'max'}

class Metric():
    def __init__(self, metricTypes):
        self.metricTypes = metricTypes
        self.meanFns = {'loss': tf.metrics.Mean()}
        for k in metricTypes:
            self.meanFns[k] = tf.metrics.Mean()
        self.yTrues = []
        self.yPreds = []
        self.yMasks = []

    def stepUpdate(self, yTrue, yPred, yMask, loss=None, writer=None, step=None, prefix='step', calc=True):
        yMask = maskNan(yPred, yMask)
        self.yTrues.append(yTrue)
        self.yPreds.append(yPred)
        self.yMasks.append(yMask)
        results = {}
        if loss is not None:
            results['loss'] = loss
        if calc:
            for k in self.metricTypes:
                results[k] = metric_fns[k](yTrue, yPred, yMask)
        for k in results:
            self.meanFns[k](results[k])
        if writer is not None:
            with writer.as_default():
                for k in results:
                    tf.summary.scalar(f'{prefix}/{k}', self.meanFns[k].result(), step=tf.cast(step, dtype=tf.int64))
        return self.getMeanResult()

    def epochCalc(self):
        yTrue = np.concatenate(self.yTrues, axis=0)
        yPred = np.concatenate(self.yPreds, axis=0)
        yMask = np.concatenate(self.yMasks, axis=0)
        yMask = maskNan(yPred, yMask)
        results = {}
        for k in self.metricTypes:
            results[k] = metric_fns[k](yTrue, yPred, yMask)
        for k in results:
            self.meanFns[k].reset_states()
            self.meanFns[k](results[k])
        return self.getMeanResult()

    def getMeanResult(self):
        meanResults = {k: self.meanFns[k].result().numpy() for k in self.meanFns}
        return meanResults

    def resetState(self):
        self.yTrues = []
        self.yPreds = []
        self.yMasks = []
        for k in self.meanFns:
            self.meanFns[k].reset_states()
