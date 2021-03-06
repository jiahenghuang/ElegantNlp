# coding=utf-8
import sys
sys.path.append('../')

import numpy as np
import math

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn


class VSumLayer(object):
    """
    向量加和
    """
    def __init__(self):
        pass

    def ops(self, input_x):
        return tf.reduce_sum(input_x, 1)


class ConcatLayer(object):
    """
    concat
    """
    def __init__(self): 
        pass
    
    def ops(self, blobs, concat_size):
        return tf.reshape(tf.concat(blobs, 1), [-1, concat_size])


class CosineLayer(object): 
    """
    a layer class: cosine layer
    """
    def __init__(self): 
        pass

    def ops(self, input_a, input_b):
        norm_a = tf.nn.l2_normalize(input_a, dim=1)
        norm_b = tf.nn.l2_normalize(input_b, dim=1)
        # multiply is dot-product
        cos_sim = tf.expand_dims(tf.reduce_sum(tf.multiply(norm_a, norm_b), 1), -1)
        return cos_sim


class AttentionLayer(object): 
    """
    a layer class: attention layer
    """
    def __init__(self, hidden_size, attention_size): 
        """
        init function
        """
        self.hidden_size = hidden_size
        self.W = tf.Variable(tf.random.normal([hidden_size, attention_size], stddev=0.1))
        self.b = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
        self.u = tf.Variable(tf.random.normal([attention_size], stddev=0.1))

    def ops(self, input_x):
        """
        operation
        """
        input_shape = input_x.shape
        sequence_length = input_shape[1].value
        # suppose input_x is not time major
        v = tf.tanh(tf.matmul(tf.reshape(input_x, [-1, self.hidden_size]), self.W) + tf.reshape(self.b, [1, -1]))
        vu = tf.matmul(v, tf.reshape(self.u, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(
            input_x * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        return output


class ExtractLastLayer(object): 
    """
    a layer class: get the last step layer
    """
    def __init__(self): 
        """
        init function
        """
        pass

    def ops(self, input_hidden, seq_length=None):
        """
        operation
        """
        if seq_length is not None:
            output = input_hidden
            batch_size = tf.shape(output)[0]
            max_length = tf.shape(output)[1]
            out_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (seq_length - 1)
            flat = tf.reshape(output, [-1, out_size])
            relevant = tf.gather(flat, index)
            return relevant
        else:
            output = tf.transpose(input_hidden, [1, 0, 2])
            return tf.gather(output, int(output.get_shape()[0]) - 1)


class ERnnLayer(object): 
    """
    a layer class: ernn layer
    """
    def __init__(self, input_size, hidden_size): 
        """
        init function
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wx = tf.Variable(tf.zeros([self.input_size, self.hidden_size]))
        self.Wh = tf.Variable(tf.zeros([self.hidden_size, self.hidden_size]))
        self.bi = tf.Variable(tf.zeros([self.hidden_size]))

    def ernn_func(self, pre_hidden, emb_t):
        """
        ernn operation
        """
        hidden = tf.tanh(
            tf.matmul(pre_hidden, self.Wh) +
            tf.matmul(emb_t, self.Wx) + self.bi)
        return hidden

    def ops(self, input_emb):
        """
        operation
        """
        init_hidden = input_emb[:][0]
        init_hidden = tf.matmul(init_hidden, tf.zeros(
            [self.input_size, self.hidden_size]))
        input_emb_ = tf.transpose(input_emb, perm=[1, 0, 2])
        all_hidden_states = tf.scan(self.ernn_func,
                                    input_emb_,
                                    initializer=init_hidden,
                                    name='states')
        all_hidden_states = tf.transpose(all_hidden_states, perm=[1, 0, 2])
        return all_hidden_states


class GRULayer(object): 
    """
    a layer class: a GRU layer
    """
    def __init__(self, hidden_size): 
        """
        init function
        """
        self.hidden_size = hidden_size

    def ops(self, input_emb):
        """
        operation
        """
        rnn_outputs, _ = rnn(GRUCell(self.hidden_size), inputs=input_emb,
                             dtype=tf.float32)
        return rnn_outputs


class LSTMLayer(object): 
    """
    a layer class: a LSTM layer
    """
    def __init__(self, hidden_size): 
        """
        init function
        """
        self.hidden_size = hidden_size
        self.cell = LSTMCell(self.hidden_size)

    def ops(self, input_emb, seq_length=None):
        """
        operation
        """
        rnn_outputs, _ = rnn(self.cell, inputs=input_emb,
                             dtype=tf.float32, sequence_length=seq_length)
        return rnn_outputs


class BiDirectionalRNNLayer(object): 
    """
    a layer class: Bi-directional LSTM/gru
    """
    def __init__(self, hidden_size, rnn_type='lstm'): 
        """
        init function
        """
        self.hidden_size = hidden_size
        type_cast = {'lstm': LSTMCell, 'gru': GRUCell}
        if rnn_type not in type_cast:
            cell_type = GRUCell
        else:
            cell_type = type_cast[rnn_type]
        self.fw_cell = cell_type(
            num_units=self.hidden_size, state_is_tuple=True)
        self.bw_cell = cell_type(
            num_units=self.hidden_size, state_is_tuple=True)

    def ops(self, input_emb, seq_length=None):
        """
        operation
        """
        bi_outputs, bi_left_state = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell,
                                    input_emb, sequence_length=seq_length, dtype=tf.float32)
        seq_encoder = tf.concat(bi_outputs, -1)
        return seq_encoder


class AdamUpdater(object): 
    """
    a layer class: adam optimization method
    """
    def __init__(self, lr): 
        """
        init function
        """
        self.lr = lr

    def ops(self, loss):
        """
        operation
        """
        return tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)


class FCLayer(object):
    """
    a layer class: a fc layer implementation in tensorflow
    """
    def __init__(self, num_in, num_out): 
        """
        init function
        """
        self.num_in = num_in
        self.num_out = num_out
        self.weight = tf.Variable(tf.random.normal([num_in, num_out]))
        self.bias = tf.Variable(tf.random.normal([num_out]))

    def ops(self, input_x):
        """
        operation
        """
        out_without_bias = tf.matmul(input_x, self.weight)
        output = tf.nn.bias_add(out_without_bias, self.bias)
        return output


class TanhLayer(object):
    """
    a layer class: tanh Activation function
    """
    def __init__(self): 
        """
        init function
        """
        pass

    def ops(self, input_x):
        """
        operation
        """
        return tf.tanh(input_x)


class ReluLayer(object):
    """
    a layer class: relu Activation function
    """
    def __init__(self): 
        """
        init function
        """
        pass

    def ops(self, input_x):
        """
        operation
        """
        return tf.nn.relu(input_x)


class SigmoidLayer(object):
    """
    a layer class: sigmoid Activation function
    """
    def __init__(self): 
        """
        init function
        """
        pass

    def ops(self, input_x):
        """
        operation
        """
        return tf.sigmoid(input_x)


class SoftsignLayer(object):
    """
    a layer class: softsign Activation function
    """
    def __init__(self):
        """
        init function
        """
        pass

    def ops(self, input_x):
        """
        operation
        """
        return tf.nn.softsign(input_x)


class DropoutLayer(object):
    """
    a layer class: a dropout layer implementation in tensorflow
    """
    def __init__(self, drop_rate): 
        """
        init function
        """
        self.drop_rate = drop_rate

    def ops(self, input_x):
        """
        operation
        """
        return tf.nn.dropout(input_x, self.drop_rate)


class EmbeddingEnhancedLayer(object): 
    """
    a layer class: embedding enhanced layer
    """
    def __init__(self, vocab_size, emb_dim, zero_pad=True, scale=True): 
        """
        init function
        """
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        emb_shape = [self.vocab_size, self.emb_dim]
        self.embedding = tf.get_variable('embedding', dtype=tf.float32, shape=emb_shape,
                                         initializer=tf.contrib.layers.xavier_initializer())
        # 随机初始化
        if zero_pad:
            self.embedding = tf.concat((tf.zeros(shape=[1, emb_dim]), self.embedding[1:]), 0)
        # embedding第一行全部替换成pad，v=0，原始的embedding第一行如何处理呢
        self.scale = scale

    def ops(self, input_x):
        """
        operation
        """
        outputs = tf.nn.embedding_lookup(self.embedding, input_x)
        if self.scale:
            outputs = outputs * (self.emb_dim ** 0.5)
        return outputs


class EmbeddingLayer(object): 
    """
    a layer class: embedding layer
    """
    def __init__(self, vocab_size, emb_dim): 
        """
        init function
        """
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        init_scope = math.sqrt(6.0 / (vocab_size + emb_dim))
        emb_shape = [self.vocab_size, self.emb_dim]
        self.embedding = tf.Variable(tf.random.uniform(emb_shape, -init_scope, init_scope))

    def ops(self, input_x):
        """
        operation
        """
        return tf.nn.embedding_lookup(self.embedding, input_x)


class EmbeddingWithVSumLayer(object): 
    """
    a layer class: sum embedding layer
    """
    def __init__(self, vocab_size, emb_dim):
        """
        init function
        """
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        init_scope = math.sqrt(6.0 / (vocab_size + emb_dim))
        emb_shape = [self.vocab_size, self.emb_dim]
        self.embedding = tf.Variable(tf.random.uniform(emb_shape, -init_scope, init_scope))

    def ops(self, input_x):
        """
        operation
        """
        return tf.nn.embedding_lookup_sparse(self.embedding, input_x[0], input_x[1], combiner="sum")


class CNNMultiKernelLayer(object): 
    """
    multi cnn for text
    """
    def __init__(self, seq_len, emb_dim, win_array, kernel_array):
        """
        init function
        """
        self.max_seq_len = seq_len
        self.emb_dim = emb_dim
        self.win_array = win_array
        self.kernel_array = kernel_array
        self.conv_w_array = []
        self.bias_array = []
        for i in range(len(self.kernel_array)):
            # why emb_dim in filter shape, how differ win_array to kernel_array
            filter_shape = [self.win_array[i], self.emb_dim, 1, self.kernel_array[i]]
            conv_w = tf.Variable(tf.random.uniform(filter_shape))
            bias = tf.Variable(tf.constant(0.0, shape=[self.kernel_array[i]]))
            self.conv_w_array.append(conv_w)
            self.bias_array.append(bias)

    def ops(self, emb):
        emb_expanded = tf.expand_dims(emb, -1)
        pooled_out = []
        # dim=4
        for i in range(len(self.kernel_array)):
            conv_local = tf.nn.conv2d(emb_expanded, self.conv_w_array[i], strides=[1, 1, 1, 1], padding="VALID")
            h_local = tf.nn.bias_add(conv_local, self.bias_array[i])
            pool_local = tf.nn.max_pool(h_local, ksize=[1, self.max_seq_len - self.win_array[i] + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID")
            pooled_out.append(pool_local)
        total_kernel_size = sum(self.kernel_array)
        return tf.reshape(tf.concat(pooled_out, 3), [-1, total_kernel_size])


class CNNDynamicPoolingLayer(object):
    def __init__(self, seq_len1, seq_len2, win_size1, win_size2, dpool_size1, dpool_size2, kernel_size):
        """
        init function
        """
        self.seq_len1 = seq_len1
        self.seq_len2 = seq_len2
        self.win_size1 = win_size1
        self.win_size2 = win_size2
        self.dpool_size1 = dpool_size1
        self.dpool_size2 = dpool_size2
        self.kernel_size = kernel_size
        filter_shape = [self.seq_len1, self.seq_len2, 1, self.kernel_size]
        self.conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv")
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.kernel_size]), name="bias")

    def ops(self, cross, mask=None):
        emb_expanded = tf.expand_dims(cross, -1)
        conv = tf.nn.conv2d(emb_expanded, self.conv_w, strides=[1, 1, 1, 1], padding="SAME", name="conv_op")
        # shape(cross)*kernel_size
        h = tf.nn.bias_add(conv, self.bias)
        if mask is not None:
            paddings = tf.ones_like(h) * (-2 ** 32 + 1) # neg info
            mask_kernel = tf.tile(tf.expand_dims(mask, -1), [1, 1, 1, self.kernel_size])
            h = tf.where(tf.equal(mask_kernel, 0), paddings, h)
        pool = tf.nn.max_pool(h, ksize=[1, self.seq_len1/self.dpool_size1, self.seq_len2/self.dpool_size2, 1],
                              strides=[1, self.seq_len1/self.dpool_size1, self.seq_len2/self.dpool_size2, 1],
                              padding="VALID", name="pool")
        pool_flat = tf.reshape(pool, [-1, self.dpool_size1*self.dpool_size2*self.kernel_size])
        return pool_flat


class CNNLayer(object):
    def __init__(self, seq_len, emb_dim, win_size, kernel_size):
        """
        init function
        """
        self.max_seq_len = seq_len
        self.emb_dim = emb_dim
        self.win_size = win_size
        self.kernel_size = kernel_size
        filter_shape = [self.win_size, self.emb_dim, 1, self.kernel_size]
        self.conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv")
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.kernel_size]), name="bias")

    def ops(self, emb):
        emb_expanded = tf.expand_dims(emb, -1)
        conv = tf.nn.conv2d(emb_expanded, self.conv_w, strides=[1, 1, 1, 1], padding="VALID", name="conv_op")
        h = tf.nn.bias_add(conv, self.bias)
        pool = tf.nn.max_pool(h, ksize=[1, self.max_seq_len - self.win_size + 1, 1, 1],
                              strides=[1, 1, 1, 1], padding="VALID", name="pool")
        pool_flat = tf.reshape(pool, [-1, self.kernel_size])
        return pool_flat


class CNNLayerConfig(object):
    def __init__(self, seq_len, emb_dim, win_height, kernel_num, win_width=None, pooling=True, pool_size=None,
                 same_conv=False, activate=False):
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.num_kernels = kernel_num
        self.conv_pad = "SAME" if same_conv else "VALID"
        self.activate = activate
        self.pooling = pooling
        self.pool_size = pool_size
        self.win_width = emb_dim if win_width is None else win_width
        self.win_height = win_height
        self.kernel_num = kernel_num
        self.bias = tf.Variable(tf.constant(0.1, shape=[kernel_num]), name="bias")

    def ops(self, emb):
        assert len(emb.shape) in [3, 4]
        channel_num = 1 if len(emb.shape) == 3 else emb.get_shape().as_list()[-1]
        emb_expanded = tf.expand_dims(emb, -1) if len(emb.shape) == 3 else emb
        filter_shape = [self.win_height, self.win_width, channel_num, self.kernel_num]
        conv_weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv")
        conv = tf.nn.conv2d(emb_expanded, conv_weight, strides=[1, 1, 1, 1], padding=self.conv_pad, name="conv_op")
        h = tf.nn.bias_add(conv, self.bias)
        if self.activate:
            h = tf.nn.relu(h, name="cnn_relu_")
        if not self.pooling:
            return h
        elif self.pool_size is None and self.win_width == self.emb_dim:
            pool_h = self.seq_len-self.win_height+1 if self.conv_pad == "VALID" else self.seq_len
            pool = tf.nn.max_pool(h, ksize=[1, pool_h, 1, 1],
                                  strides=[1, 1, 1, 1], padding="VALID", name="pool")
            return pool
        elif self.pool_size is not None and self.win_width != self.emb_dim:
            pool = tf.nn.max_pool(h, ksize=[1, self.pool_size, self.pool_size, 1],
                                  strides=[1, 1, 1, 1], padding="VALID", name="pool")
            return pool


class KernelPoolingLayer(object):
    """
    a layer class: A kernel pooling implementation in tensorflow
    """
    def __init__(self, kernel_num, lamb): 
        """
        init function
        """
        self.kernel_num = kernel_num
        self.lamb = lamb

        def get_mus(kernel_num):
            """
            get mus
            """
            mu_vals = [1]
            if kernel_num == 1:
                return tf.constant(value=mu_vals, name="mus")
            bin_size = 2.0 / (kernel_num - 1)
            mu_vals.append(1 - bin_size / 2)
            for i in xrange(1, kernel_num - 1):
                mu_vals.append(mu_vals[i] - bin_size)
            return tf.constant(value=mu_vals, name="mus")

        def get_sigmas(kernel_num, lamb):
            """
            get sigmas
            """
            bin_size = 2.0 / (kernel_num - 1)
            sigma_vals = [0.00001]
            if kernel_num == 1:
                return tf.constant(value=sigma_vals, name="sigmas")
            sigma_vals += [bin_size * lamb] * (kernel_num - 1)
            return tf.constant(value=sigma_vals, name="sigmas")

        self.mus = get_mus(self.kernel_num)
        self.sigmas = get_sigmas(self.kernel_num, self.lamb)

    def ops(self, sim):
        """
        operation
        """
        # compute gaussian kernel
        batch_size, query_len, title_len = sim.get_shape()
        reshape_sim = tf.expand_dims(sim, -1)
        # compute Gaussian score of each kernel
        tmp = tf.exp(-tf.square(tf.subtract(reshape_sim, self.mus)) / tf.multiply(tf.square(self.sigmas), 2))
        feats = []  # store the soft-TF features from each field
        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, 2)
        kde = tf.log(tf.maximum(kde, 1e-10)) * 0.01
        aggregated_kde = tf.reduce_sum(kde, 1)
        feats.append(aggregated_kde)
        feats_concat = tf.concat(feats, 1)
        return feats_concat


class SimilarityMatrixLayer(object):
    """
    a layer class: A similarity matrix implementation in tensorflow
    """
    def __init__(self): 
        """
        init function
        """
        pass

    def ops(self, query_emb, title_emb):
        """
        operation
        """
        query_norm = tf.sqrt(tf.reduce_sum(
            tf.square(query_emb), 2, keep_dims=True))
        normalized_query_emb = query_emb / query_norm
        title_norm = tf.sqrt(tf.reduce_sum(
            tf.square(title_emb), 2, keep_dims=True))
        normalized_title_emb = title_emb / title_norm
        sim_mat = tf.matmul(normalized_query_emb, tf.transpose(
            normalized_title_emb, perm=[0, 2, 1]), name="similarity_matrix")
        return sim_mat


class SequencePoolingLayer(object):
    """
    a layer class: A sequence pooling implementation in tensorflow
    """
    def __init__(self): 
        """
        init function
        """
        pass

    def ops(self, emb):
        """
        operation
        """
        reduce_sum = tf.reduce_sum(emb, axis=1)
        return reduce_sum


if __name__ == "__main__":
    None
