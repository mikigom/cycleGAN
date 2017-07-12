import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import layer_utils

ngf = 64
ndf = 64

class Generator():
    def __init__(self, x, name):
        assert x.get_shape().as_list()[1:4] == [256, 256, 3]
        self.x = x
        self.ngf = ngf
        self.name = name
        self.y = self.model_define()

    def model_define(self):
        # Johnson, Justin, Alexandre Alahi, and Li Fei-Fei.
        #"Perceptual losses for real-time style transfer and super-resolution."(2016)
        # https://github.com/antlerros/tensorflow-fast-neuralstyle/blob/master/net.py#L29
        def resnet_block(x, n_filters, idx):
            with tf.variable_scope('resnet_block' + str(idx)):
                conv1 = slim.conv2d(x,     n_filters, (3, 3), scope = 'res1')
                conv2 = slim.conv2d(conv1, n_filters, (3, 3), scope = 'res2', activation_fn = None)
                res = x + conv2
                return res

        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L130
        # >> Without Dropout <<
        with tf.variable_scope('generator_' + self.name):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                padding = 'SAME',
                                #normalizer_fn = tf.contrib.layers.batch_norm,
                                normalizer_fn = layer_utils.instance_norm,
                                activation_fn = tf.nn.relu):
                conv1 = slim.conv2d(self.x, self.ngf, (7, 7), scope = 'conv1')
                with tf.variable_scope('downsampling'):
                    down1 = slim.conv2d(conv1, 2*self.ngf, (3, 3), (2, 2), scope = 'down1')
                    down2 = slim.conv2d(down1, 4*self.ngf, (3, 3), (2, 2), scope = 'down2')
                with tf.variable_scope('resnet'):
                    resnet1 = resnet_block(down2,   4*self.ngf, 1)
                    resnet2 = resnet_block(resnet1, 4*self.ngf, 2)
                    resnet3 = resnet_block(resnet2, 4*self.ngf, 3)
                    resnet4 = resnet_block(resnet3, 4*self.ngf, 4)
                    resnet5 = resnet_block(resnet4, 4*self.ngf, 5)
                    resnet6 = resnet_block(resnet5, 4*self.ngf, 6)
                    resnet7 = resnet_block(resnet6, 4*self.ngf, 7)
                    resnet8 = resnet_block(resnet7, 4*self.ngf, 8)
                    resnet9 = resnet_block(resnet8, 4*self.ngf, 9)
                with tf.variable_scope('upsampling'):
                    up1 = slim.conv2d_transpose(resnet9, 2*self.ngf, (3, 3), (2, 2), scope = 'up1')
                    up2 = slim.conv2d_transpose(up1,     self.ngf,   (3, 3), (2, 2), scope = 'up2')
                conv2 = slim.conv2d(up2, 3, (7, 7), activation_fn = tf.nn.tanh, scope = 'conv2',\
                                    biases_initializer = None, normalizer_fn = None)
                #conv2 = slim.conv2d(up2, 3, (7, 7), activation_fn = None, scope = 'conv2')
                return conv2

class Discriminator():
    def __init__(self, x, name, use_patchGAN = False, use_sigmoid = False):
        if use_patchGAN:
            assert x.get_shape().as_list()[1:4] == [70, 70, 3]
        self.x = x
        self.ndf = ndf
        self.name = name
        self.use_sigmoid = use_sigmoid
        self.y = self.model_define()

    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L287
    def model_define(self):
        with tf.variable_scope('discriminator_' + self.name):
            with slim.arg_scope([slim.conv2d],
                                padding = 'SAME',
                                normalizer_fn = tf.contrib.layers.batch_norm,
								#normalizer_fn = layer_utils.instance_norm,
                                activation_fn = layer_utils.lrelu):
                conv1 = slim.conv2d(self.x, self.ndf,   (4, 4), (2, 2), scope = 'conv1',\
                                    normalizer_fn = None, biases_initializer = None)
                conv2 = slim.conv2d(conv1,  2*self.ndf, (4, 4), (2, 2), scope = 'conv2')
                conv3 = slim.conv2d(conv2,  4*self.ndf, (4, 4), (2, 2), scope = 'conv3')
                conv4 = slim.conv2d(conv3,  8*self.ndf, (4, 4), (2, 2), scope = 'conv4')
                conv5 = slim.conv2d(conv4,  1,          (4, 4), (2, 2), scope = 'conv5',\
                                    normalizer_fn = None, activation_fn = None)

                if self.use_sigmoid:
                    return tf.nn.sigmoid(conv5)
                else:
                    return conv5
