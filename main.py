from __future__ import absolute_import, division, print_function
import math
import numpy as np
import tensorflow as tf
import os
import scipy.misc
from scipy.misc import imsave
from progressbar import ETA, Bar, Percentage, ProgressBar
from vae import VAE
from tensorflow.examples.tutorials.mnist import input_data
from data_reader import load_data, get_next_batch

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 200, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "/tempspace/hyuan/VAE", "the file directory")
flags.DEFINE_integer("hidden_size", 3, "size of the hidden VAE unit")
flags.DEFINE_integer("channel", 128, "size of initial channel in decoder")
flags.DEFINE_integer("iterations", 100, "number of iterations in generating")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot= True)
  #  Train_set , Test_set = load_data('freyface', FLAGS.working_directory, 0.9, 0)
    model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.channel, FLAGS.iterations)

    for epoch in range(FLAGS.max_epoch):
        training_loss = 0.0
        pbar = ProgressBar()

        for i in pbar(range(FLAGS.updates_per_epoch)):
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
       #     images = get_next_batch(Train_set, FLAGS.batch_size)
            loss_value= model.update_params(images, epoch*FLAGS.updates_per_epoch + i)
            training_loss += loss_value
        #    print ("=============KL loss", kl_loss)
     #       print ("==============rec loss", rec_loss)
        model.save(epoch)
        training_loss = training_loss/ (FLAGS.updates_per_epoch * FLAGS.batch_size)
        print ("Loss %f" % training_loss)
        model.generate_and_save_images(FLAGS.working_directory)