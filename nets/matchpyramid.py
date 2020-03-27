# coding=utf-8
import logging
import tensorflow as tf
import layers.tf_layers as layers
from utils.utility import get_cross_mask

class MatchPyramid(object):
    """
    MatchPyramid init function
    """
    def __init__(self, config):
        self.vocab_size = int(config['vocabulary_size'])
        self.emb_size = int(config['embedding_dim'])
        self.kernel_size = int(config['num_filters'])
        self.win_size1 = int(config['window_size_left'])
        self.win_size2 = int(config['window_size_right'])
        self.dpool_size1 = int(config['dpool_size_left'])
        self.dpool_size2 = int(config['dpool_size_right'])
        self.hidden_size = int(config['hidden_size'])
        self.left_name, self.seq_len1 = config['left_slots'][0]
        self.right_name, self.seq_len2 = config['right_slots'][0]
        self.emb_layer = layers.EmbeddingEnhancedLayer(self.vocab_size, self.emb_size, zero_pad=True, scale=False)
        self.cnn_layer = layers.CNNDynamicPoolingLayer(self.seq_len1, self.seq_len2, self.win_size1, self.win_size2,
                                                       self.dpool_size1, self.dpool_size2, self.kernel_size)
        self.relu_layer = layers.ReluLayer()
        self.tanh_layer = layers.TanhLayer()
        if 'match_mask' in config and config['match_mask'] != 0:
            self.match_mask = True
        else:
            self.match_mask = False
        self.fc1_layer = layers.FCLayer(self.kernel_size * self.dpool_size1 * self.dpool_size2, self.hidden_size)
        self.n_class = int(config['n_class'])
        self.fc2_layer = layers.FCLayer(self.hidden_size, self.n_class)
        
    def predict(self, left_slots, right_slots):
        """
        predict graph of this net
        """
        try:
            left = left_slots[self.left_name]
            right = right_slots[self.right_name]
        except:
            left, right = left_slots, right_slots
        left_emb = self.emb_layer.ops(left) # (N, len, D)
        right_emb = self.emb_layer.ops(right) # (N, len, D)
        cross = tf.matmul(left_emb, tf.transpose(right_emb, [0, 2, 1])) # (N, len, len)
        if self.match_mask:
            cross_mask = get_cross_mask(left, right)
        else:
            cross_mask = None
        conv_pool = self.cnn_layer.ops(cross, mask=cross_mask)
        pool_relu = self.relu_layer.ops(conv_pool)
        relu_hid1 = self.fc1_layer.ops(pool_relu)
        hid1_tanh = self.tanh_layer.ops(relu_hid1)
        pred = self.fc2_layer.ops(hid1_tanh)
        return pred
