# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben 2020/8/17 上午10:19
import tensorflow as tf
from tensorflow.keras import layers

class Update_gru(layers.Layer):
    def __init__(self, hiddenSize):
        super(Update_gru, self).__init__()
        self.denseR = layers.Dense(hiddenSize, activation='sigmoid')
        self.denseZ = layers.Dense(hiddenSize, activation='sigmoid')
        self.denseH = layers.Dense(hiddenSize, activation='tanh')
        self.ln_msg = layers.LayerNormalization(trainable=False)
        self.ln_nf = layers.LayerNormalization(trainable=False)

    def call(self, nf, msg, training=False):
        msg = self.ln_msg(msg, training=training)
        msg_nf = tf.concat([msg, nf], axis=1)
        r = self.denseR(msg_nf)
        z = self.denseZ(msg_nf)
        # =========================================
        nf1 = self.denseH(tf.concat([nf * r, msg], axis=1))
        # =========================================
        nf = nf * (1.0 - z) + nf1 * z
        nf = self.ln_nf(nf, training=training)
        # =========================================
        return nf
