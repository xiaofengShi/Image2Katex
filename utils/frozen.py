import tensorflow as tf
import os

project_path = os.path.abspath(os.path.basename(os.path.join(os.path.realpath(__file__), '..')))
checkpoints = tf.train.get_checkpoint_state(project_path + '/checkpoint')
input_checkpoint = checkpoints.model_checkpoint_path
print('input_checkpoint:', input_checkpoint)

absolute_model = '/'.join(input_checkpoint.split('/')[:-1])
print('absolute_model:', absolute_model)
out_graph = absolute_model+'/fozen_model.pb'

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    saver.restore(sess, input_checkpoint)

    for op in tf.get_default_graph().get_operations():
        print(op.name, op.values())
    # NOTE: 模型的输出tensor名称是什么？
    output_graph = tf.graph_util.convert_variables_to_constants(
        sess, tf.get_default_graph().as_graph_def(), output_node_names=[''])
