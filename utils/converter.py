#coding=utf-8
from collections import Counter
import numpy
import errno
import time
import sys
import os
import tensorflow as tf

my_int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))


class TFConverter(object):
    """
    TFConverter init, with config
    """
    def __init__(self, config):
        data_id_file = config['data_id_file']
        data_tfrecord_file = config['data_tfrecord_file']
        self.source_file = []
        self.target_file = []
        if os.path.isdir(data_id_file):
            try:
                os.makedirs(data_tfrecord_file)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(data_tfrecord_file):
                    pass
                else:
                    raise
            data_parts = os.listdir(data_id_file)
            for part in data_parts:
                self.source_file.append(os.path.join(data_id_file, part))
                self.target_file.append(os.path.join(data_tfrecord_file, part))
        else:
            self.source_file.append(data_id_file)
            self.target_file.append(data_tfrecord_file)
        data_mode = config['training_mode']
        self.left_slots = config["left_slots"]
        self.right_slots = config["right_slots"]
        self.pad_id = 0
        
        self.n_class = config["n_class"]
        self.func = self.convert_pointwise
        self.all_slots = self.left_slots + self.right_slots
            
    def convert_pointwise(self, line):
        """
        convert pointwise data,pointwise parse
        """
        # left_ids \t right_ids \t label
        group = line.strip().split("\t")
        if len(group) != 1 + len(self.all_slots):
            print >> sys.stderr, "convert error, slots doesn't match"
            sys.exit(-1)
        label = [0 for i in range(self.n_class)]
        all_ids = []
        label[int(group[-1])] = 1
        feature = {"label": my_int_feature(label)}
        for i in range(len(self.all_slots)):
            slot_name, seq_len = self.all_slots[i]
            tmp_ids = [int(t) for t in group[i].strip().split(" ")]
            if len(tmp_ids) < seq_len:
                pad_len = seq_len - len(tmp_ids)
                tmp_ids = tmp_ids + [self.pad_id] * pad_len
            feature[slot_name] = my_int_feature(tmp_ids[:seq_len])
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example
    
    def write_data_to_tf(self, filename, tfrecord_name):
        """
        write to tfrecord file, write data
        """
        writer = tf.python_io.TFRecordWriter(tfrecord_name)
        with open(filename) as fin_data:
            for line in fin_data:
                example = self.func(line)
                writer.write(example.SerializeToString())
        writer.close()

    def convert(self):
        """
        convert all files
        """
        print >> sys.stderr, "writing tf record"
        for i in range(len(self.source_file)):
            self.write_data_to_tf(self.source_file[i], self.target_file[i])
            print >> sys.stderr, self.source_file[i], "-->", self.target_file[i]
        print >> sys.stderr, "all done"


def run_convert(config):
    """
    run convert
    """
    tf_conv = TFConverter(config)
    tf_conv.convert()


if __name__ == "__main__":
    file_conf = {"train": "data/train_ids.txt.tf",
                 "test": "data/test_ids.txt.tf"}
    for k, v in file_conf.items():
        tf_tran = TFConverter(data_id_file=v, data_tfrecord_file=v + ".tf",n_class=2,
                              seq_len_config={"left": 50, "right": 50})
        tf_tran.convert()