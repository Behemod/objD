import tensorflow as tf

with tf.Graph().as_default() as graph:
    with tf.compat.v1.gfile.FastGFile('ImageAI/mytest/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        for i, node in enumerate(graph_def.node):
            print(f"{i}  {node.name}")
