from __future__ import absolute_import, division, print_function
import math
import numpy as np
import tensorflow as tf
import argparse
import os
import scipy.misc
from scipy.misc import imsave
from progressbar import ETA, Bar, Percentage, ProgressBar
from vae import VAE
from tensorflow.examples.tutorials.mnist import input_data
from data_reader import load_data, get_next_batch
from celeba import celeba
from cifar_reader import cifar_reader
flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 550, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_integer("max_test_epoch", 100, "max  test epoch")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("working_directory", "/tempspace/hyuan/VAE", "the file directory")
flags.DEFINE_integer("hidden_size", 3, "size of the hidden VAE unit")
flags.DEFINE_integer("channel", 128, "size of initial channel in decoder")
flags.DEFINE_integer("checkpoint", 99, "number of epochs to be reloaded")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '11'
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, or test')
    args = parser.parse_args()
    if args.action not in ['train', 'test']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test")    
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):    
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot= True)
  #  Train_set , Test_set = load_data('freyface', FLAGS.working_directory, 0.9, 0)
    model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.channel)
  #  data = celeba()
    #data =cifar_reader()
 #   images = data.next_batch(FLAGS.batch_size)
    if args.action == 'train':
      for epoch in range(FLAGS.max_epoch): 
          training_loss = 0.0
          pbar = ProgressBar()
          for i in pbar(range(FLAGS.updates_per_epoch)):
              images, _ = mnist.train.next_batch(FLAGS.batch_size)
          #    images = tf.reshape(images, [FLAGS.batch_size,])
          #     images = get_next_batch(Train_set, FLAGS.batch_size)
          #     images = data.next_batch(FLAGS.batch_size)
          #    images = data.next_batch(FLAGS.batch_size)
              loss_value, kl_loss, rec_loss = model.update_params(images, epoch*FLAGS.updates_per_epoch + i)
              training_loss += loss_value
            #   print ("=============KL loss", kl_loss)
            #   print ("==============rec loss", rec_loss)
          model.save(epoch)
          training_loss = training_loss/ (FLAGS.updates_per_epoch * FLAGS.batch_size)
          print ("Loss %f" % training_loss)
          model.generate_and_save_images(FLAGS.batch_size, FLAGS.working_directory)
    elif args.action == 'test':
        model.reload(FLAGS.checkpoint)

        sum_ll = 0
        for epoch in range(FLAGS.max_test_epoch):
            test_images, _ = mnist.test.next_batch(FLAGS.batch_size)
            sum_ll = sum_ll + model.evaluate(test_images)
            print("current sum ll ================", sum_ll)
        sum_ll = sum_ll/ FLAGS.max_test_epoch
        print("======================NLL: %d"% sum_ll)
    