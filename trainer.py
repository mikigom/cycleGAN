import tensorflow as tf
import numpy as np
import os, sys
import logging
import cycleGAN as cycleGAN
import tensorflow.contrib.slim as slim
import buffer as buffer
from reader import Reader
X_dir = 'data/tfrecords/apple.tfrecords'
Y_dir = 'data/tfrecords/orange.tfrecords'
image_size = 256
batch_size = 1
lamb = 10
epochs = 100
buffer_memory_epoch = 50
summ_dir = "./summ/apple2orange"
check_dir = "./ckpts/apple2orange"
is_loaded = False

class Trainer():
    def __init__(self):
        self.image_size = image_size
        self.batch_size = batch_size
        self.X_dir = X_dir
        self.Y_dir = Y_dir
        self.lamb = lamb
        self.epochs = epochs
        self.buffer_memory_epoch = buffer_memory_epoch
        self.summ_dir = summ_dir
        self.check_dir = check_dir
        self.is_loaded = is_loaded

        self.reader_define()
        self.model_construct()
        self.loss_define()
        self.optim_define()
        self.summary_define()

    def reader_define(self):
        X_reader = Reader(self.X_dir, name='apple',
            image_size=self.image_size, batch_size=self.batch_size)
        Y_reader = Reader(self.Y_dir, name='orange',
            image_size=self.image_size, batch_size=self.batch_size)
        self.X = X_reader.feed()
        self.Y = Y_reader.feed()

    def model_construct(self):
        self.buffer_F_Y = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'buffer_F_Y')
        self.buffer_G_X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'buffer_G_X')

        with tf.variable_scope("Model") as scope:
            self.G_X = cycleGAN.Generator(self.X, name = "G").y
            self.F_Y = cycleGAN.Generator(self.Y, name = "F").y
            self.Dx_X = cycleGAN.Discriminator(self.X, "Dx").y
            self.Dy_Y = cycleGAN.Discriminator(self.Y, "Dy").y

            scope.reuse_variables()

            self.Dx_F_Y = cycleGAN.Discriminator(self.F_Y, name = "Dx").y
            self.Dy_G_X = cycleGAN.Discriminator(self.G_X, name = "Dy").y
            self.F_G_X = cycleGAN.Generator(self.G_X, name = "F").y
            self.G_F_Y = cycleGAN.Generator(self.F_Y, name = "G").y

            scope.reuse_variables()

            self.Dx_buffer_F_Y = cycleGAN.Discriminator(self.buffer_F_Y, name = "Dx").y
            self.Dy_buffer_G_X = cycleGAN.Discriminator(self.buffer_G_X, name = "Dy").y

    def loss_define(self):
        with tf.variable_scope("loss"):
            self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

            disc_loss_X = tf.reduce_mean(tf.squared_difference(self.Dx_F_Y, 0.9))
            disc_loss_Y = tf.reduce_mean(tf.squared_difference(self.Dy_G_X, 0.9))

            cyc_loss = tf.reduce_mean(tf.abs(self.F_G_X - self.X)) + tf.reduce_mean(tf.abs(self.Y - self.G_F_Y))

            self.F_loss = self.lamb*cyc_loss + disc_loss_Y
            self.G_loss = self.lamb*cyc_loss + disc_loss_X

            self.Dx_loss = tf.reduce_mean(tf.square(self.Dx_buffer_F_Y)) + tf.reduce_mean(tf.squared_difference(self.Dx_X, 0.9))
            self.Dy_loss = tf.reduce_mean(tf.square(self.Dy_buffer_G_X)) + tf.reduce_mean(tf.squared_difference(self.Dy_Y, 0.9))

    def optim_define(self):
        optimizer = tf.train.AdamOptimizer(self.lr, beta1 = 0.5)
        self.model_vars = tf.trainable_variables()

        G_vars = [var for var in self.model_vars if 'generator_G' in var.name]
        F_vars = [var for var in self.model_vars if 'generator_F' in var.name]
        Dx_vars = [var for var in self.model_vars if 'discriminator_Dx' in var.name]
        Dy_vars = [var for var in self.model_vars if 'discriminator_Dy' in var.name]

        self.G_optim = optimizer.minimize(self.G_loss, var_list = G_vars)
        self.F_optim = optimizer.minimize(self.F_loss, var_list = F_vars)
        self.Dx_optim = optimizer.minimize(self.Dx_loss, var_list = Dx_vars)
        self.Dy_optim = optimizer.minimize(self.Dy_loss, var_list = Dy_vars)

    def summary_define(self):
        slim.model_analyzer.analyze_vars(self.model_vars, print_info = True)

        self.F_loss_summ = tf.summary.scalar("F_loss", self.F_loss)
        self.G_loss_summ = tf.summary.scalar("G_loss", self.G_loss)
        self.Dx_loss_summ = tf.summary.scalar("Dx_loss", self.Dx_loss)
        self.Dy_loss_summ = tf.summary.scalar("Dy_loss", self.Dy_loss)

        self.X_summ = tf.summary.image('X', self.X)
        self.Y_summ = tf.summary.image('Y', self.Y)
        self.G_X_summ = tf.summary.image('X_to_Y', self.G_X)
        self.F_Y_summ = tf.summary.image('Y_to_X', self.F_Y)
        self.F_G_X_summ = tf.summary.image('X_from_F_G_X', self.F_G_X)
        self.G_F_Y_summ = tf.summary.image('Y_from_G_F_Y', self.G_F_Y)
        self.buffer_F_Y_summ = tf.summary.image('buffer_F_Y', self.buffer_F_Y)
        self.buffer_G_X_summ = tf.summary.image('buffer_G_X', self.buffer_G_X)

        self.step = tf.Variable(0, trainable = False, name = 'step')

    def train(self):
        writer = tf.summary.FileWriter(self.summ_dir)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            if self.is_loaded:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord=coord)

            X_curr_lr = 0.0002
            Y_curr_lr = 0.0002
            try:
                G_X_buffer = buffer.Buffer(self.buffer_memory_epoch*self.batch_size, self.batch_size)
                F_Y_buffer = buffer.Buffer(self.buffer_memory_epoch*self.batch_size, self.batch_size)

                step = 0
                while not coord.should_stop():
                    if step % 10000 == 0:
                        saver.save(sess, self.check_dir)
                    __ = sess.run(self.G_optim, feed_dict = {self.lr : X_curr_lr})
                    _, G_X, summ, summ_, summ__, summ___ = sess.run([self.G_optim, self.G_X, self.G_loss_summ,\
                                                                     self.X_summ, self.G_X_summ, self.F_G_X_summ],\
                                                                     feed_dict={self.lr : X_curr_lr})
                    
                    writer.add_summary(summ, step)
                    writer.add_summary(summ_, step)
                    writer.add_summary(summ__, step)
                    writer.add_summary(summ___, step)
                    
                    G_X_buffer.push(G_X)
                    buffer_G_X_images = G_X_buffer.sample(self.batch_size)
                    
                    _, summ, summ_ = sess.run([self.Dy_optim, self.Dy_loss_summ, self.buffer_G_X_summ],
                                        feed_dict={self.buffer_G_X : buffer_G_X_images, self.lr : Y_curr_lr})
                    writer.add_summary(summ, step)
                    writer.add_summary(summ_, step)

                    _ = sess.run(self.F_optim, feed_dict = {self.lr : Y_curr_lr})
                    _, F_Y, summ, summ_, summ__, summ___ = sess.run([self.F_optim, self.F_Y, self.F_loss_summ,\
                                                                     self.Y_summ, self.F_Y_summ, self.G_F_Y_summ],
                                                                     feed_dict={self.lr : Y_curr_lr})

                    writer.add_summary(summ, step)
                    writer.add_summary(summ_, step)
                    writer.add_summary(summ__, step)
                    writer.add_summary(summ___, step)

                    F_Y_buffer.push(F_Y)
                    buffer_F_Y_images = F_Y_buffer.sample(self.batch_size)
                    
                    _, summ, summ_ = sess.run([self.Dx_optim, self.Dx_loss_summ, self.buffer_F_Y_summ],\
                                        feed_dict={self.buffer_F_Y : buffer_F_Y_images, self.lr : X_curr_lr})

                    writer.add_summary(summ, step)
                    writer.add_summary(summ_, step)
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
                coord.request_stop()
            except KeyboardInterrupt:
                logging.info("Interrupted")
                coord.request_stop()
            except Exception as e:
                print(e)
                print("@ line {}".format(sys.exc_info()[-1].tb_lineno))
                coord.request_stop()
            finally:
                print('Stop')
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
