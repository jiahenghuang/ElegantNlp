#coding=utf-8
import logging
import tensorflow as tf
import layers.tf_layers as layers
from utils.utility import seq_length

class LSTM(object):
    """
    mlp cnn init function
    """
    def __init__(self, config):
        self.vocab_size = int(config['vocabulary_size'])
        self.emb_size = int(config['embedding_dim'])
        self.rnn_hidden_size = int(config['rnn_hidden_size'])
        self.hidden_size = int(config['hidden_size'])
        self.left_name, self.seq_len1 = config['left_slots'][0]
        self.right_name, self.seq_len2 = config['right_slots'][0]
        self.emb_layer = layers.EmbeddingEnhancedLayer(self.vocab_size, 
                                self.emb_size, zero_pad=True, scale=False)
        self.rnn = layers.LSTMLayer(self.rnn_hidden_size)
        self.extract = layers.ExtractLastLayer()
        self.n_class = int(config['n_class'])
        self.fc1_layer = layers.FCLayer(self.rnn_hidden_size * 2, self.hidden_size)
        self.fc2_layer = layers.FCLayer(self.hidden_size, self.n_class)
        
    def predict(self, left_slots, right_slots):
        """
        predict graph of this net
        """
        left = left_slots[self.left_name]
        right = right_slots[self.right_name]
        left_emb = self.emb_layer.ops(left) # (N, len, D)
        right_emb = self.emb_layer.ops(right) # (N, len, D)
        ## left
        left_length = seq_length(left)
        left_encoder = self.rnn.ops(left_emb, left_length)
        left_rep = self.extract.ops(left_encoder, left_length)
        right_length = seq_length(right)
        right_encoder = self.rnn.ops(right_emb, right_length)
        right_rep = self.extract.ops(right_encoder, right_length)
        rep_concat = tf.concat([left_rep, right_rep], -1)
        hidden1 = self.fc1_layer.ops(rep_concat)
        pred = self.fc2_layer.ops(hidden1)
        return pred

