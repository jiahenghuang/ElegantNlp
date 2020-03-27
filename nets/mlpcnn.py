#coding=utf-8
import logging
import layers.tf_layers as layers

class MLPCnn(object):
    """
    mlp cnn init function
    cnn-mlp, 即 CDSSM 算法
    """
    def __init__(self, config):
        self.vocab_size = int(config['vocabulary_size'])
        self.emb_size = int(config['embedding_dim'])
        self.kernel_size = int(config['num_filters'])
        self.win_size = int(config['window_size'])
        self.hidden_size = int(config['hidden_size'])
        self.left_name, self.seq_len = config['left_slots'][0]
        self.right_name, self.seq_len = config['right_slots'][0]
        self.emb_layer = layers.EmbeddingLayer(self.vocab_size, self.emb_size)
        self.cnn_layer = layers.CNNLayer(self.seq_len, self.emb_size,
                                         self.win_size, self.kernel_size)
        self.relu_layer = layers.ReluLayer()
        self.tanh_layer = layers.TanhLayer()
        self.concat_layer = layers.ConcatLayer()
        self.fc1_layer = layers.FCLayer(self.kernel_size, self.hidden_size)
        self.n_class = int(config['n_class'])
        self.fc2_layer = layers.FCLayer(2 * self.hidden_size, self.n_class)
        
    def predict(self, left_slots, right_slots):
        """
        predict graph of this net
        """
        left = left_slots[self.left_name]
        right = right_slots[self.right_name]
        left_emb = self.emb_layer.ops(left)
        right_emb = self.emb_layer.ops(right)
        left_cnn = self.cnn_layer.ops(left_emb)
        right_cnn = self.cnn_layer.ops(right_emb)
        left_relu = self.relu_layer.ops(left_cnn)
        right_relu = self.relu_layer.ops(right_cnn)
        hid1_left = self.fc1_layer.ops(left_relu)
        hid1_right = self.fc1_layer.ops(right_relu)
        left_tanh = self.tanh_layer.ops(hid1_left)
        right_tanh = self.tanh_layer.ops(hid1_right)
        concat = self.concat_layer.ops([left_tanh, right_tanh], self.hidden_size * 2)
        pred = self.fc2_layer.ops(concat)
        return pred

