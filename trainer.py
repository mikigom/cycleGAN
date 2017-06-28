import tensorflow as tf
import numpy as np
import os

import cycleGAN as cycleGAN
import tensorflow.contrib.slim as slim
import buffer as buffer

X_dir = "./dataset/apple2orange/trainX"
Y_dir = "./dataset/apple2orange/trainY"
image_size = 256
patch_size = 70
batch_size = 1
min_queue_examples = 100
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
        self.min_queue_examples = min_queue_examples
        self.lamb = lamb
        self.epochs = epochs
        self.buffer_memory_epoch = buffer_memory_epoch
        self.summ_dir = summ_dir
        self.check_dir = check_dir
        self.is_loaded = is_loaded

        self.data_setup()
        self.model_construct()
        self.loss_define()
        self.optim_define()
        self.summary_define()

    def data_setup(self):
        with tf.variable_scope('data_setup'):
            self.filenames_X = tf.train.match_filenames_once(self.X_dir + "/*.jpg")
            self.filenames_Y = tf.train.match_filenames_once(self.Y_dir + "/*.jpg")

            self.filename_queue_X = tf.train.string_input_producer(self.filenames_X, num_epochs = self.epochs)
            self.filename_queue_Y = tf.train.string_input_producer(self.filenames_Y, num_epochs = self.epochs)

            image_reader = tf.WholeFileReader()
            _, image_file_X = image_reader.read(self.filename_queue_X)
            _, image_file_Y = image_reader.read(self.filename_queue_Y)

            images_X_origin = tf.image.decode_jpeg(image_file_X, channels = 3)
            images_Y_origin = tf.image.decode_jpeg(image_file_Y, channels = 3)
            images_X = tf.image.resize_images(images_X_origin, [self.image_size, self.image_size])
            images_Y = tf.image.resize_images(images_Y_origin, [self.image_size, self.image_size])
            images_X.set_shape((self.image_size, self.image_size, 3))
            images_Y.set_shape((self.image_size, self.image_size, 3))
            images_X = tf.image.convert_image_dtype(images_X, tf.float32)/127.5 - 1.0
            images_Y = tf.image.convert_image_dtype(images_Y, tf.float32)/127.5 - 1.0

            self.X_batch = tf.train.shuffle_batch([images_X],
                                    batch_size = self.batch_size,
                                    num_threads = 1,
                                    capacity = self.min_queue_examples + 3 * self.batch_size,
                                    min_after_dequeue = 50)
            self.Y_batch = tf.train.shuffle_batch([images_Y],
                                    batch_size = self.batch_size,
                                    num_threads = 1,
                                    capacity = self.min_queue_examples + 3 * self.batch_size,
                                    min_after_dequeue = 50)

    def model_construct(self):
        self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'X')
        self.Y = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'Y')

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
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

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

    def train(self):
        writer = tf.summary.FileWriter(self.summ_dir)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord=coord)

            if self.is_loaded:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            try:
                X_data_len = len(os.listdir(self.X_dir))
                Y_data_len = len(os.listdir(self.Y_dir))

                G_X_buffer = buffer.Buffer(self.buffer_memory_epoch*self.batch_size, self.batch_size)
                F_Y_buffer = buffer.Buffer(self.buffer_memory_epoch*self.batch_size, self.batch_size)

                X_epoch = 0
                X_batch_iter = 0
                Y_epoch = 0
                Y_batch_iter = 0

                while not coord.should_stop():
                    if X_epoch % 10 == 0 or Y_epoch % 10 == 0:
                        saver.save(sess, self.check_dir)

                    if (self.batch_size*X_batch_iter)/X_data_len >= 1:
                        X_batch_iter = 0
                        X_epoch += 1

                    if (self.batch_size*Y_batch_iter)/Y_data_len >= 1:
                        Y_batch_iter = 0
                        Y_epoch += 1

                    if(X_epoch < 100) :
                        X_curr_lr = 0.0002
                    else:
                        X_curr_lr = 0.0002 - 0.0002*(Y_epoch-100)/100

                    if(Y_epoch < 100) :
                        Y_curr_lr = 0.0002
                    else:
                        Y_curr_lr = 0.0002 - 0.0002*(Y_epoch-100)/100

                    X_images, Y_images = sess.run([trainer.X_batch, trainer.Y_batch])

                    _, G_X, summ, summ_, summ__, summ___ = sess.run([self.G_optim, self.G_X, self.G_loss_summ,\
                                                                     self.X_summ, self.G_X_summ, self.F_G_X_summ],\
                                                                     feed_dict={self.X : X_images,\
                                                                     self.Y : Y_images,\
                                                                     self.lr : X_curr_lr})
                    
                    writer.add_summary(summ, self.batch_size*X_epoch + X_batch_iter)
                    writer.add_summary(summ_, self.batch_size*X_epoch + X_batch_iter)
                    writer.add_summary(summ__, self.batch_size*X_epoch + X_batch_iter)
                    writer.add_summary(summ___, self.batch_size*X_epoch + X_batch_iter)
                    
                    G_X_buffer.push(G_X)
                    buffer_G_X_images = G_X_buffer.sample(self.batch_size)                   
                    #buffer_G_X_images = [tf.random_crop(buffer_G_X_images, [70, 70, 3]) for i in range(10)]
                    
                    _, summ, summ_ = sess.run([self.Dy_optim, self.Dy_loss_summ, self.buffer_G_X_summ],
                                        feed_dict={self.buffer_G_X : buffer_G_X_images,\
                                                   self.Y : Y_images,\
                                                   self.lr : Y_curr_lr})
                    writer.add_summary(summ, self.batch_size*Y_epoch + Y_batch_iter)
                    writer.add_summary(summ_, self.batch_size*Y_epoch + Y_batch_iter)                      

                    _, F_Y, summ, summ_, summ__, summ___ = sess.run([self.F_optim, self.F_Y, self.F_loss_summ,\
                                                                     self.Y_summ, self.F_Y_summ, self.G_F_Y_summ],
                                                                     feed_dict={self.X : X_images,\
                                                                     self.Y : Y_images,\
                                                                     self.lr : Y_curr_lr})

                    writer.add_summary(summ, self.batch_size*Y_epoch + Y_batch_iter)
                    writer.add_summary(summ_, self.batch_size*Y_epoch + Y_batch_iter)
                    writer.add_summary(summ__, self.batch_size*Y_epoch + Y_batch_iter)
                    writer.add_summary(summ___, self.batch_size*Y_epoch + Y_batch_iter)

                    F_Y_buffer.push(F_Y)
                    buffer_F_Y_images = F_Y_buffer.sample(self.batch_size)
                    #buffer_F_Y_images = [tf.random_crop(buffer_F_Y_images, [70, 70, 3]) for i in range(10)]
                    
                    _, summ, summ_ = sess.run([self.Dx_optim, self.Dx_loss_summ, self.buffer_F_Y_summ],\
                                        feed_dict={self.buffer_F_Y : buffer_F_Y_images,\
                                                   self.X : X_images,\
                                                   self.lr : X_curr_lr})

                    writer.add_summary(summ, self.batch_size*X_epoch + X_batch_iter)
                    writer.add_summary(summ_, self.batch_size*X_epoch + X_batch_iter)

                    X_batch_iter += 1
                    Y_batch_iter += 1

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                print('Stop')
                coord.request_stop()

            coord.join(threads)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
