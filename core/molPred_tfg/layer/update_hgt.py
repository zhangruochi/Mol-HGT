# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by ben  2020/8/17 上午10:19
import tensorflow as tf
from tensorflow.keras import layers

class Update_hgt(layers.Layer):
    def __init__(self, hiddenSize, nTypeNum):
        super(Update_hgt, self).__init__()
        init_rn = tf.random_normal_initializer()
        init_one = tf.ones_initializer()
        init_zero = tf.zeros_initializer()
        self.wa = tf.Variable(initial_value=init_rn([nTypeNum, hiddenSize, hiddenSize]), trainable=True, name=self.name + '_wa')
        self.ba = tf.Variable(initial_value=init_zero([1, nTypeNum, hiddenSize]), trainable=True, name=self.name + '_ba')
        self.skip = tf.Variable(initial_value=init_one([nTypeNum, 1]), trainable=True, name=self.name + '_skip')
        self.ln_nf = layers.LayerNormalization(trainable=False)

    def call(self, nf, nType, nType_id, msg, training=None):
        nf1 = tf.einsum('ni,tio->nto', msg, self.wa) + self.ba
        nf1 = tf.einsum('nto,nt->no', nf1, nType)
        # =========================================
        alpha = tf.sigmoid(self.skip)
        alpha = tf.gather(alpha, nType_id)
        nf2 = nf * (1.0 - alpha) + nf1 * alpha
        nf2 = self.ln_nf(nf2, training=training)
        # =========================================
        return nf2
