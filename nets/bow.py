# coding=utf-8
import logging
import tensorflow as tf
import layers.tf_layers as layers

class BOW(object):
    """
    bow init function
    类似微软的 DSSM, 即 MLP在 match 问题上的应用
    """
    def __init__(self, config):
        self.vocab_size = int(config['vocabulary_size'])
        self.emb_size = int(config['embedding_dim'])
        self.bow_size = int(config["bow_size"])
        self.hidden_size = int(config['hidden_size'])
        self.left_name, self.seq_len = config["left_slots"][0]
        self.right_name, self.seq_len = config["right_slots"][0]
        self.emb_layer = layers.EmbeddingLayer(self.vocab_size, self.emb_size)
        self.seq_pool_layer = layers.SequencePoolingLayer()
        self.softsign_layer = layers.SoftsignLayer()
        self.bow_layer = layers.FCLayer(self.emb_size, self.bow_size)
        self.n_class = int(config['n_class'])
        self.fc_layer = layers.FCLayer(2 * self.hidden_size, self.n_class)
     
    def predict(self, left_slots, right_slots):
        """
        predict graph of this net
        """
        try:
            left = left_slots[self.left_name]
            right = right_slots[self.right_name]
        except:
            left, right = left_slots, right_slots
        left_emb = self.emb_layer.ops(left)
        right_emb = self.emb_layer.ops(right)
        left_pool = self.seq_pool_layer.ops(left_emb)
        right_pool = self.seq_pool_layer.ops(right_emb)
        left_soft = self.softsign_layer.ops(left_pool)
        right_soft = self.softsign_layer.ops(right_pool)
        left_bow = self.bow_layer.ops(left_soft)
        right_bow = self.bow_layer.ops(right_soft)
        concat = tf.concat([left_bow, right_bow], -1)
        pred = self.fc_layer.ops(concat)
        return pred


# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
