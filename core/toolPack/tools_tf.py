# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben  2020/4/30 上午11:45
import math
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import MeanMetricWrapper
from multiprocessing import Process

class LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, trainSteps, warmupSteps=0, offsetSteps=0,
                 lrStart=0.0, lrMax=1e-3, lrEnd=1e-6,
                 curveType='cos', writer=None, name='lr'):
        super(LearningRate, self).__init__()
        self.writer = writer
        self.name = name
        self.offsetSteps = offsetSteps
        self.curveFns = {'cos': decayCurve_cos, 'exp': decayCurve_exp}
        self.curve = self.curveFns[curveType](
            trainSteps, warmupSteps=warmupSteps, lrStart=lrStart, lrMax=lrMax, lrEnd=lrEnd
        )

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.int64) + self.offsetSteps
        lr = self.curve[step]
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar(self.name, lr, step=step)
        return lr

class LearningRate_epoch(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, epochs, warmupEpochs=0,
                 lrStart=1e-4, lrMax=1e-3, lrEnd=1e-5,
                 curveType='cos', writer=None, name='lr'):
        super(LearningRate_epoch, self).__init__()
        self.writer = writer
        self.curveFns = {'cos': decayCurve_cos, 'exp': decayCurve_exp}
        self.curve = self.curveFns[curveType](
            epochs, warmupSteps=warmupEpochs, lrStart=lrStart, lrMax=lrMax, lrEnd=lrEnd
        )
        self.name = name
        self.epoch = 0

    def __call__(self, step):
        lr = self.curve[self.epoch]
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar(self.name, lr, step=tf.cast(step, dtype=tf.int64))
        return lr

    def updateEpoch(self):
        self.epoch = self.epoch + 1

class LearningRate_epoch_batch(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, epochs, dataSize, warmupEpochs=0,
                 lrStart=0.0, lrMax=1e-3, lrEnd=1e-5,
                 curveType='cos', writer=None, name='lr'):
        super(LearningRate_epoch_batch, self).__init__()
        self.dataSize = tf.cast(dataSize, dtype=tf.float32)
        self.writer = writer
        self.name = name
        self.epoch = 0
        self.count = 0.0
        self.curveFns = {'cos': decayCurve_cos, 'exp': decayCurve_exp}
        self.curve = self.curveFns[curveType](
            epochs + 1, warmupSteps=warmupEpochs, lrStart=lrStart, lrMax=lrMax, lrEnd=lrEnd
        )
        self.curve = tf.convert_to_tensor(self.curve, dtype=tf.float32)

    def __call__(self, step):
        lr0 = self.curve[self.epoch]
        lr1 = self.curve[self.epoch + 1]
        lr = lr0 + self.count / self.dataSize * (lr1 - lr0)
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar(self.name, lr, step=tf.cast(step, dtype=tf.int64))
        return lr

    def updateEpoch(self):
        self.epoch = self.epoch + 1
        self.count = 0.0

    def updateBatchSize(self, batchSize):
        self.count = self.count + tf.cast(batchSize, dtype=tf.float32)

class PiecewiseConstantDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, trainStep, lrs, bounds=None, writer=None, name='lr'):
        super(PiecewiseConstantDecay, self).__init__()
        self.trainStep = trainStep
        self.lrs = lrs
        self.writer = writer
        self.name = name
        if bounds == None:
            self.bounds = [float((i + 1) / len(lrs)) for i in range(len(lrs) - 1)]
        else:
            self.bounds = bounds

    def __call__(self, step):
        stepRatio = step / self.trainStep
        lr = self.lrs[-1]
        for i, b in enumerate(self.bounds):
            if stepRatio <= b:
                lr = self.lrs[i]
                break
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar(self.name, lr, step=tf.cast(step, dtype=tf.int64))
        return lr

def decayCurve_exp(steps, warmupSteps=0.0, lrStart=0.0, lrMax=1e-3, lrEnd=1e-5, show=False):
    decayRate = math.pow(lrEnd / lrMax, 1.0 / (steps - warmupSteps - 1))
    curve = []
    for step in range(steps):
        if warmupSteps == 0:
            lr = lrMax * np.power(decayRate, (step - warmupSteps))
        else:
            lr1 = lrStart + step * (lrMax - lrStart) / warmupSteps
            lr2 = lrMax * np.power(decayRate, (step - warmupSteps))
            lr = np.minimum(lr1, lr2)
        curve.append(lr)
    if show:
        plt.plot(curve)
        plt.show()
    return curve

def decayCurve_cos(steps, warmupSteps=0.0, lrStart=0.0, lrMax=1e-3, lrEnd=1e-5, show=False):
    curve = []
    for step in range(steps):
        if step <= warmupSteps:
            lr = lrStart + step * (lrMax - lrStart) / warmupSteps
        else:
            lr = lrEnd + 0.5 * (lrMax - lrEnd) * \
                 (1.0 + np.cos((step - warmupSteps) / (steps - 1 - warmupSteps) * np.pi))
        curve.append(lr)
    if show:
        plt.plot(curve)
        plt.show()
    return curve

# =====================================================
def lossFunc_crossentropy(yTrue, yPred, yMask=None):
    loss = - (1.0 - yTrue) * tf.math.log(1.0 - yPred + 1e-7) - yTrue * tf.math.log(yPred + 1e-7)
    if yMask is not None:
        loss = loss * yMask
    loss = tf.reduce_mean(loss)
    return loss

def lossFunc_focal(yTrue, yPred, yMask=None, gamma=2.0):
    loss = - (1 - yTrue) * tf.math.log(1.0 - yPred + 1e-7) * tf.pow(yPred, gamma) \
           - yTrue * tf.math.log(yPred + 1e-7) * tf.pow(1.0 - yPred, gamma)
    # =================================
    if yMask is not None:
        loss = loss * yMask
    loss = tf.reduce_mean(loss)
    return loss

def lossFunc_cut(yTrue, yPred, yMask=None, cut=0.3):
    loss = - (1.0 - yTrue) * tf.math.log(1.0 - tf.math.maximum(yPred, cut) + 1e-7) - yTrue * tf.math.log(tf.math.minimum(yPred, 1.0 - cut) + 1e-7)
    # =================================
    if yMask is not None:
        loss = loss * yMask
    loss = tf.reduce_mean(loss)
    return loss

def lossFunc_precision(yTrue, yPred, yMask=None, l=1):
    loss = - (1.0 - yTrue) * tf.math.log(1.0 - tf.math.maximum(yPred, 0.1) + 1e-7) - yTrue * tf.math.log(tf.math.minimum(yPred, 0.9) + 1e-7)
    loss += l * yPred
    # =================================
    if yMask is not None:
        loss = loss * yMask
    loss = tf.reduce_mean(loss)
    return loss

def lossFunc_mae(yTrue, yPred, yMask=None):
    loss = tf.keras.losses.mean_absolute_error(yTrue, yPred)
    if yMask is not None:
        loss = loss * yMask
    loss = tf.reduce_mean(loss)
    return loss

def lossFunc_mse(yTrue, yPred, yMask=None):
    loss = tf.keras.losses.mean_squared_error(yTrue, yPred)
    if yMask is not None:
        loss = loss * yMask
    loss = tf.reduce_mean(loss)
    return loss

def lossFunc_huber(yTrue, yPred, yMask=None):
    loss = tf.keras.losses.huber(yTrue, yPred, delta=0.5)
    if yMask is not None:
        loss = loss * yMask
    loss = tf.reduce_mean(loss)
    return loss

def getLossFunc(args):
    fns = {
        'crossentropy': lossFunc_crossentropy,
        'focal': lossFunc_focal,
        'cut': lossFunc_cut,
        'precision': lossFunc_precision,
        'mae': lossFunc_mae,
        'mse': lossFunc_mse,
        'huber': lossFunc_huber,
    }
    if args.taskType == 'c':
        return fns[args.cLossFn]
    else:
        return fns[args.rLossFn]

# =====================================================
def tensorboard(logdir, port=6006, host=None, log=True):
    cmd = f'tensorboard --logdir {logdir} --port {port}'
    if host is not None:
        cmd += f' --host {host}'
    if not log:
        cmd += ' >/dev/null 2>&1'
    p = Process(target=os.system, args=[cmd])
    p.start()
    return p
