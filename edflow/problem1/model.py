import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope


import nn as nn
from nn import conv2D, dense


# mnist example form
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
# adapted for EDFlow


def mnist_model(x):
    with arg_scope([conv2D, dense], activation=tf.nn.relu):
        features = conv2D(x, 32, 5, padding="valid")
        features = tf.layers.max_pooling2d(features, 2, strides=1)
        features = conv2D(features, 64, 3, padding="valid")
        features = tf.layers.max_pooling2d(features, 2, strides=1)
        features = conv2D(features, 64, 3, padding="valid")
        features = tf.layers.max_pooling2d(features, 2, strides=1)

        y = tf.layers.flatten(features)
        y = dense(y, 1024)
        y = dense(y, 512)
    logits = dense(y, 10)  # 10 classes for mnist
    probs = tf.nn.softmax(logits, dim=-1)
    return probs, logits


class TrainModel(object):
    def __init__(self, config):
        self.config = config
        self.define_graph()
        self.variables = tf.global_variables()


    @property
    def inputs(self):
        '''
        A dictionary of inputs to the model at inference time.
        The keys have to be the same as the returned keys from the example class.
        The values have to be tensorflow placeholders.

        Returns
        -------

        '''
        _inputs = # TODO: fill this in
        return _inputs


    @property
    def outputs(self):
        '''
        outputs of model at inference time
        Returns
        -------

        '''
        return {'probs' : self.probs,
                "classes" : self.classes}


    def define_graph(self):
        # inputs
        self.image = tf.placeholder(
            tf.float32,
            shape = (
                self.config["batch_size"],
                self.config["spatial_size"],
                self.config["spatial_size"],
                1),
            name = "image_in")
        self.targets = tf.placeholder(
            tf.float32,
            shape=(
                self.config["batch_size"], # 10 classes in mnist # TODO maybe move this to config
            ))

        # model definition
        model = nn.make_model("model", mnist_model) # <--- remember the model name for the trainer later
        probs, logits = model(self.image)

        # outputs
        self.probs = probs
        self.logits = logits
        self.classes = tf.argmax(probs, axis=1)



