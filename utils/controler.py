# coding=utf-8
import sys
import platform
import tensorflow as tf
from sklearn.metrics import classification_report


def run_predict(pred, label, config):
    """
    run classification predict function handle
    """
    mean_acc = 0.0
    saver = tf.compat.v1.train.Saver()
    label_index = tf.argmax(label, 1)
    pred_prob = tf.nn.softmax(pred, -1)
    score = tf.reduce_max(pred_prob, -1)
    pred_index = tf.argmax(pred_prob, 1)
    correct_pred = tf.equal(pred_index, label_index)
    acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
    result_file = open(config["test_result"], "w")
    pred_list, y_list = [], []
    step = 0
    init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1)) as sess:
        sess.run(init)
        try:
            saver.restore(sess, config["test_model_file"]+"final")
        except:
            saver.restore(sess, config["test_model_file"] + "epoch{}".format(config["num_epochs"]))
        coord = tf.train.Coordinator()
        read_thread = tf.train.start_queue_runners(sess=sess, coord=coord)
        while not coord.should_stop():
            step += 1
            try:
                ground, pi, a, prob = sess.run([label_index, pred_index, acc, score])
                pred_list += [pi[0]]
                y_list += [ground[0]]
                mean_acc += a
                for i in range(len(prob)):
                    result_file.write("%d\t%d\t%f\n" % (ground[i], pi[i], prob[i]))
            except tf.errors.OutOfRangeError:
                coord.request_stop()
        coord.join(read_thread)
    sess.close()
    result_file.close()
    mean_acc = mean_acc / step
    print(sys.stderr, "accuracy: %4.2f" % (mean_acc * 100))
    print(classification_report(y_true=y_list, y_pred=pred_list))


def run_trainer(loss, optimizer, config):
    tf.compat.v1.summary.scalar('loss', loss)
    merged = tf.compat.v1.summary.merge_all()
    thread_num = int(config["thread_num"])
    model_path = config["model_path"]
    model_file = config["model_prefix"]
    print_iter = int(config["print_iter"])
    data_size = int(config["data_size"])
    batch_size = int(config["batch_size"])
    epoch_iter = int(data_size / batch_size)
    avg_cost = 0.0
    saver = tf.compat.v1.train.Saver(max_to_keep=None)
    init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=thread_num, 
                                          inter_op_parallelism_threads=thread_num)) as sess:
        pi_list = []
        for k in ["model_prefix", "embedding_dim", "batch_size", "learning_rate"]:
            pi_list += [str(config[k])]
        tb_path = "graph/{}".format("_".join(pi_list))
        writer = tf.compat.v1.summary.FileWriter(tb_path, sess.graph)
        sess.run(init)
        coord = tf.train.Coordinator()
        read_thread = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        epoch_num = 1
        while not coord.should_stop():
            try:
                step += 1
                c, _, summary = sess.run([loss, optimizer, merged])
                avg_cost += c

                if step % print_iter == 0:
                    print("iter %d , loss: %f" % (step, avg_cost/print_iter))
                    avg_cost = 0.0
                    writer.add_summary(summary, step)
                if step % epoch_iter == 0:
                    print("iter %d save model epoch%d" % (step, epoch_num))
                    saver.save(sess, "%s/%s.epoch%d" % (model_path, model_file, epoch_num))
                    epoch_num += 1
            except tf.errors.OutOfRangeError:
                print("iter %d : %s" % (step, "OutOfRangeError"))
                saver.save(sess, "%s/%s.final" % (model_path, model_file))
                coord.request_stop()
        coord.join(read_thread)
        writer.close()
    sess.close()


def graph_save(pred, config):
    """
    run classify predict
    """
    graph_path = config["graph_path"]
    graph_name = config["graph_name"]
    pred_prob = tf.nn.softmax(pred, -1, name="output_prob")
    saver = tf.compat.v1.train.Saver()
    step = 0
    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1)) \
                    as sess:
        sess.run(init)
        tf.train.write_graph(sess.graph_def, graph_path, graph_name, as_text=True)
    sess.close()

