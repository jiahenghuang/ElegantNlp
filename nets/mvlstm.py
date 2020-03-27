# coding=utf-8
import logging
import tensorflow as tf
import layers.tf_layers as layers 
from utils.utility import seq_length

class MVLSTM(object):
    """
    体现了交互, 不是向量表示后直接压缩
    An implementation of A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations
    """
    def __init__(self, config):
        self.vocab_size = int(config['vocabulary_size'])
        self.emb_size = int(config['embedding_dim'])
        self.k_max_num = int(config['k_max_num'])
        self.hidden_size = int(config['hidden_size'])
        self.left_name, self.seq_len1 = config['left_slots'][0]
        self.right_name, self.seq_len2 = config['right_slots'][0]
        self.emb_layer = layers.EmbeddingEnhancedLayer(self.vocab_size, self.emb_size, zero_pad=True, scale=False)
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, state_is_tuple=True)
        self.bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, state_is_tuple=True)
        if 'match_mask' in config and config['match_mask'] != 0:
            self.match_mask = True
        else:
            self.match_mask = False
        self.n_class = int(config['n_class'])
        self.fc2_layer = layers.FCLayer(self.k_max_num, self.n_class)
        
    def predict(self, left_slots, right_slots):
        """
        predict graph of this net
        """
        left = left_slots[self.left_name]
        right = right_slots[self.right_name]
        left_emb = self.emb_layer.ops(left) # (N, len, D)
        right_emb = self.emb_layer.ops(right) # (N, len, D)
        bi_left_outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, left_emb,
                                                             sequence_length=seq_length(left), dtype=tf.float32)
        left_seq_encoder = tf.concat(bi_left_outputs, -1)
        bi_right_outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, right_emb,
                                                              sequence_length=seq_length(right), dtype=tf.float32)
        right_seq_encoder = tf.concat(bi_right_outputs, -1)

        cross = tf.matmul(left_seq_encoder, tf.transpose(right_seq_encoder, [0, 2, 1])) # (N, len, len)
        # if self.match_mask:
        #    cross_mask = get_cross_mask(left, right)
        #    cross = tf.multiply(cross, tf.cast(cross_mask, tf.float32))
        #    paddings = tf.ones_like(cross)*(-2**32+1)
        #    cross = tf.where(tf.equal(cross_mask, 0), paddings, cross)
        cross_reshape = tf.reshape(cross, [-1, self.seq_len1 * self.seq_len2])
        k_max_match = tf.nn.top_k(cross_reshape, k=self.k_max_num, sorted=True)[0]
        #k_max_match = tf.Print(k_max_match, [k_max_match], "k_max_match")
        pred = self.fc2_layer.ops(k_max_match)
        return pred

