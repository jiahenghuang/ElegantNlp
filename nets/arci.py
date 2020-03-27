# coding=utf-8
import logging
import layers.tf_layers as layers

class ARCI(object):
    """
    mlp cnn init function
    cnn-mlp, 即 CDSSM 算法
    """

    def __init__(self, config):
        self.vocab_size = int(config['vocabulary_size'])
        self.emb_size = int(config['embedding_dim'])
        self.kernel_num = int(config['num_filters'])
        self.win_size = int(config['window_size'])
        self.pool_size = int(config['pool_size'])
        self.hidden_size = int(config['hidden_size'])
        self.dropout_rate = float(config['dropout_rate'])
        self.drop_out = layers.DropoutLayer(drop_rate=self.dropout_rate)
        self.tanh_layer = layers.TanhLayer()
        self.left_name, self.seq_len = config['left_slots'][0]
        self.right_name, self.seq_len = config['right_slots'][0]
        self.emb_layer = layers.EmbeddingLayer(self.vocab_size, self.emb_size)
        self.cnn_layer = layers.CNNLayerConfig(seq_len=self.seq_len, emb_dim=self.emb_size, win_height=1,
                                               win_width=self.win_size, kernel_num=self.kernel_num, same_conv=True,
                                               activate=False, pooling=True, pool_size=self.pool_size)
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
        concat = self.concat_layer.ops([left_cnn, right_cnn], self.hidden_size * 2)
        concat_drop = self.drop_out.ops(concat)
        pred = self.fc2_layer.ops(concat_drop)
        return pred
