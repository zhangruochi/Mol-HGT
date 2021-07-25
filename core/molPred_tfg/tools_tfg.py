# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by ben  2020/3/30 下午1:28
import tensorflow as tf
import numpy as np
from collections import Counter
from tensorflow.keras import layers

epsilon = 1e-5

# ===================================================
def segment_gather_with_pad(x, id, batch_size, max_num):
    x_list = []
    xMask_list = []
    for i in range(batch_size):
        index = tf.squeeze(tf.where(tf.equal(id, i)))
        if tf.rank(index) == 0:
            index = tf.expand_dims(index, 0)
        size_i = tf.shape(index)[0]

        x_i = tf.gather(x, index)
        x_i = tf.pad(x_i, [[0, max_num - size_i], [0, 0]])
        x_list.append(x_i)

        xMask_i = tf.ones(size_i, dtype=tf.float32)
        xMask_i = tf.pad(xMask_i, [[0, max_num - size_i]])
        xMask_list.append(xMask_i)
    x = tf.stack(x_list)
    xMask = tf.stack(xMask_list)
    xMask = tf.expand_dims(xMask, axis=2)
    return x, xMask

# ===================================================
def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf

# ===================================================
def segment_sum(x, id):
    return tf.math.segment_sum(x, id)

def segment_mean(x, id):
    return tf.math.segment_mean(x, id)

def segment_max(x, id):
    return tf.math.segment_max(x, id)

def segment_min(x, id):
    return tf.math.segment_min(x, id)

def segment_std(x, id):
    return tf.sqrt(segment_var(x, id) + epsilon)

def segment_var(x, id):
    x_mean_squares = segment_mean(x * x, id)
    x_mean = segment_mean(x, id)
    var = tf.nn.relu(x_mean_squares - x_mean * x_mean)
    return var

def segment_softmax(x, id, epsilon=epsilon):
    x_exp = tf.exp(x)
    x_exp_sum = segment_sum(x_exp, id)
    x_exp_sum = tf.gather(x_exp_sum, id)
    x = x_exp / (x_exp_sum + epsilon)
    return x

def segment_moment(x, id, n=3):
    x_mean = segment_mean(x, id)
    x_n = segment_mean(tf.pow(x - x_mean, n))
    rooted_h_n = tf.sign(x_n) * tf.pow(tf.abs(x_n) + epsilon, 1.0 / n)
    return rooted_h_n

# ===================================================
def degreeScale_amplification(x, nodeLogDegree, avgNodeLogDegree):
    return x * (nodeLogDegree / avgNodeLogDegree)

def degreeScale_attenuation(x, nodeLogDegree, avgNodeLogDegree):
    return x * (avgNodeLogDegree / nodeLogDegree)

def getNodeLogDegree(id):
    nodeLogDegree = tf.math.segment_sum(tf.ones_like(id), id)
    nodeLogDegree = tf.math.log(tf.cast(nodeLogDegree, tf.float32) + 1.0)
    return nodeLogDegree

def getNodeLogDegree_np(id):
    nodeLogDegree = dict(Counter(id))
    nodeLogDegree = np.array(list(nodeLogDegree.values()), dtype=np.float32)
    nodeLogDegree = np.log(nodeLogDegree + 1.0)
    return nodeLogDegree

# ===================================================
class Model_ensemble(tf.keras.Model):
    def __init__(self, modelClass, args):
        super(Model_ensemble, self).__init__()
        self.models = [modelClass(args) for i in range(args.modelEnsemble)]
        self.ensemble = args.modelEnsemble

    def call(self, batchGraph, training=False):
        ids = list(range(self.ensemble))
        if training:
            ids = tf.random.shuffle(ids)
            ids = ids[0:1]
        xs = []
        for i in ids:
            xs.append(self.models[i](batchGraph, training=training))
        x = tf.reduce_mean(tf.stack(xs, axis=0), axis=0)
        return x

class Layer_ensemble(layers.Layer):
    def __init__(self, layerClass, *args, ensemble=1, **kwargs):
        super(Layer_ensemble, self).__init__()
        self.ls = [layerClass(*args, **kwargs) for _ in range(ensemble)]
        self.ensemble = ensemble

    def call(self, *args, training=None):
        ids = list(range(self.ensemble))
        if training and self.ensemble > 1:
            ids = tf.random.shuffle(ids)
            ids = ids[0:1]
        xs = []
        for i in ids:
            xs.append(self.ls[i](*args, training=training))
        x = tf.reduce_mean(tf.stack(xs, axis=0), axis=0)
        return x

class Output_block(layers.Layer):
    def __init__(self, labels=1, taskType='c', hiddenSize=100, extFeat=False, dropout=0.0,
                 yMean_train=0.0, yStd_train=1.0, rYScale=0, fcEnsemble=1):
        super(Output_block, self).__init__()
        self.extFeat = extFeat
        if self.extFeat:
            self.denseExtFeat = layers.Dense(200, activation='relu')
            self.lnExtFeat = layers.LayerNormalization(trainable=True)
        self.fc = Layer_ensemble(Fc_block,
                                 labels=labels,
                                 taskType=taskType,
                                 hiddenSize=hiddenSize,
                                 dropout=dropout,
                                 yMean_train=yMean_train,
                                 yStd_train=yStd_train,
                                 rYScale=rYScale,
                                 ensemble=fcEnsemble)

    def call(self, x, feat, training=False):
        if self.extFeat:
            feat = self.denseExtFeat(feat)
            feat = self.lnExtFeat(feat, training=training)
            x1 = tf.concat([x, feat], axis=1)
        else:
            x1 = x
        x = self.fc(x1, training=training)
        return x

class Fc_block(layers.Layer):
    def __init__(self, labels=1, taskType='c', hiddenSize=100, dropout=0.0,
                 yMean_train=0.0, yStd_train=1.0, rYScale=0):
        super(Fc_block, self).__init__()
        self.taskType = taskType
        self.yMean_train = tf.expand_dims(tf.convert_to_tensor(yMean_train, dtype=tf.float32), axis=0)
        self.yStd_train = tf.expand_dims(tf.convert_to_tensor(yStd_train, dtype=tf.float32), axis=0)
        self.rYScale = rYScale
        if self.taskType == 'c':
            activation = 'sigmoid'
        else:
            activation = None
        self.fcs = tf.keras.Sequential([
            layers.Dense(int(hiddenSize / 1.4), activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(int(hiddenSize / 2.0), activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(labels, activation=activation)
        ])

    def call(self, x, training=None):
        x = self.fcs(x, training=training)
        if self.taskType == 'r' and self.rYScale:
            x = x * self.yStd_train + self.yMean_train
        return x
