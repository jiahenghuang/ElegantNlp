# coding=utf-8
import argparse
import logging
import json
import sys
import os

import tensorflow as tf

from utils import datafeeds
from utils import controler
from utils import utility
from utils import converter


def load_config(config_file):
    """
    load config
    """
    with open(config_file, "r") as f:
        try:
            conf = json.load(f)
        except Exception:
            logging.error("load json file %s error" % config_file)
    conf_dict = {}
    unused = [conf_dict.update(conf[k]) for k in conf]
    logging.debug("\n".join(["%s=%s" % (u, conf_dict[u]) for u in conf_dict]))
    return conf_dict

def train(conf_dict):
    tf.compat.v1.reset_default_graph()
    net = utility.import_object(conf_dict["net_py"], conf_dict["net_class"])(conf_dict)
    datafeed = datafeeds.TFPointwisePaddingData(conf_dict)
    input_l, input_r, label_y = datafeed.ops()
    pred = net.predict(input_l, input_r)
    loss_layer = utility.import_object(
        conf_dict["loss_py"], conf_dict["loss_class"])()
    loss = loss_layer.ops(pred, label_y)
    # define optimizer
    lr = float(conf_dict["learning_rate"])
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    # run_trainer
    controler.run_trainer(loss, optimizer, conf_dict)

def predict(conf_dict):
    tf.compat.v1.reset_default_graph()
    net = utility.import_object(
        conf_dict["net_py"], conf_dict["net_class"])(conf_dict)
    conf_dict.update({"batch_size": "1", "shuffle": "0", "train_file": conf_dict["test_file"]})
    if "dropout_rate" in conf_dict:
        conf_dict.update({"dropout_rate": 1.0})
    test_datafeed = datafeeds.TFPointwisePaddingData(conf_dict)
    test_l, test_r, test_y = test_datafeed.ops()
    pred = net.predict(test_l, test_r)
    controler.run_predict(pred, test_y, conf_dict)

if __name__ == "__main__": 
    # model_configs = os.listdir('config')
    # model_configs = ['config/'+config for config in model_configs] 
    # for model_config in model_configs:
    #     config = load_config(model_config)
    #     config.update({'train_file': 'data/train_ids.txt.tf',
    #                'test_file': 'data/test_ids.txt.tf',
    #                'vocabulary_size': 39625})
    #     train(config)
    #     print("- "*20)
    #     predict(config)
        config = load_config('config/arci.json')
        config.update({'train_file': 'data/train_ids.txt.tf',
                   'test_file': 'data/test_ids.txt.tf',
                   'vocabulary_size': 39625})
        train(config)
        print("- "*20)
        predict(config)