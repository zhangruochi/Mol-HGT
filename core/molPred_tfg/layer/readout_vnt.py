# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by ben  2020/6/23 上午11:50
import tensorflow as tf
from tensorflow.keras import layers
from ..tools_tfg import segment_softmax


class Readout_vnt(layers.Layer):
    def __init__(self, hiddenSize, heads=10, vntResidual=True):
        super(Readout_vnt, self).__init__()
        self.hiddenSize = hiddenSize
        self.heads = heads
        self.vntResidual = vntResidual
        self.relu = layers.ReLU()
        assert self.hiddenSize % self.heads == 0
        self.dk = self.hiddenSize // self.heads
        self.scale = tf.sqrt(tf.cast(self.dk, dtype=tf.float32))
        init_rn = tf.random_normal_initializer()
        self.nf_virtualNode = tf.Variable(initial_value=init_rn([1, self.hiddenSize]), trainable=False, name=self.name + '_nf_vitrual')
        self.denseQ = layers.Dense(self.hiddenSize, use_bias=False)
        self.denseK = layers.Dense(self.hiddenSize, use_bias=False)
        self.denseV = layers.Dense(self.hiddenSize, use_bias=False)
        self.denseO = layers.Dense(self.hiddenSize, activation='relu')
        self.ln1 = layers.LayerNormalization(trainable=True)
        self.ln2 = layers.LayerNormalization(trainable=True)

    def call(self, nf, nId, training=False):
        q = self.denseQ(self.nf_virtualNode)
        k = self.denseK(nf)
        v = self.denseV(nf)
        q = tf.reshape(q, [1, self.heads, self.dk])
        k = tf.reshape(k, [-1, self.heads, self.dk])
        v = tf.reshape(v, [-1, self.heads, self.dk])
        att = tf.reduce_sum(q * k, axis=2)
        att = att / self.scale
        att = segment_softmax(att, nId)

        x = v * tf.expand_dims(att, axis=2)
        x = tf.reshape(x, [-1, self.hiddenSize])
        x = tf.math.segment_mean(x, nId)

        x = self.ln1(x, training=training)
        x = self.denseO(x)
        x = self.ln2(x, training=training)
        if self.vntResidual:
            x0 = tf.math.segment_sum(nf, nId)
            x = x + x0
        return x
