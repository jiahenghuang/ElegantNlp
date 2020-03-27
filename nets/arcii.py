# coding=utf-8

import layers.tf_layers as layers
from utils.utility import cross_match, tf


class ARCII(object):
    """
    implement of arcii
    """
    def __init__(self, config):
        self.vocab_size = int(config['vocabulary_size'])
        self.emb_size = int(config['embedding_dim'])
        self.dropout_rate = float(config['dropout_rate'])

        self.left_name, self.seq_len = config['left_slots'][0]
        self.right_name, self.seq_len = config['right_slots'][0]

        self.kernel_num1 = int(config['num_filters_1'])
        self.win_size1 = int(config['window_size_1'])
        self.kernel_num2 = int(config['num_filters_2'])
        self.win_size2 = int(config['window_size_2'])
        self.pool_size2 = int(config['pool_size2'])

        self.drop_out = layers.DropoutLayer(drop_rate=self.dropout_rate)
        self.emb_layer = layers.EmbeddingLayer(self.vocab_size, self.emb_size)
        self.cnn1_layer = layers.CNNLayerConfig(seq_len=self.seq_len, emb_dim=self.emb_size,
                                                win_height=self.win_size1, win_width=self.emb_size, kernel_num=self.kernel_num1,
                                                same_conv=False, pooling=False, activate=False)
        self.cnn_out_len = self.seq_len-self.win_size1+1
        self.cnn2_layer = layers.CNNLayerConfig(seq_len=self.seq_len, emb_dim=self.seq_len, win_height=self.win_size2,
                                                win_width=self.win_size2, kernel_num=self.kernel_num2,
                                                same_conv=True, pooling=True, activate=True, pool_size=self.pool_size2)
        self.cnn2_out_size = (self.cnn_out_len-self.pool_size2+1)*(self.cnn_out_len-self.pool_size2+1)*self.kernel_num2
        self.n_class = int(config['n_class'])
        self.fc_layer = layers.FCLayer(self.cnn2_out_size, self.n_class)

    def predict(self, left_slots, right_slots):
        left = left_slots[self.left_name]
        right = right_slots[self.right_name]
        left_emb = self.emb_layer.ops(left)
        right_emb = self.emb_layer.ops(right)
        left_cnn = self.cnn1_layer.ops(left_emb)
        print(left_cnn.get_shape())
        right_cnn = self.cnn1_layer.ops(right_emb)
        cross = cross_match([left_cnn, right_cnn], match_type='plus')
        batch_size = cross.get_shape().as_list()[0]
        print(cross.get_shape())
        cross_ = tf.reshape(cross, [batch_size, self.cnn_out_len, self.cnn_out_len, -1])
        print(cross_.get_shape())
        cross_cnn = self.cnn2_layer.ops(cross_)
        print(cross_cnn.get_shape())
        cross_cnn_drop = self.drop_out.ops(tf.reshape(cross_cnn, [batch_size, -1]))
        print(cross_cnn_drop.get_shape())
        pred = self.fc_layer.ops(cross_cnn_drop)
        return pred
