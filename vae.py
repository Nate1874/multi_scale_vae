import os
import tensorflow as tf
import numpy as np
import math
from ops import encoder, decoder
from generator import Generator

class VAE(Generator):

    def __init__(self, hidden_size, batch_size, learning_rate):
        self.working_directory = '/tempspace/hyuan/VAE'
        self.height = 28
        self.width = 28
        self.modeldir = './modeldir'
        self.logdir = './logdir'
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate =learning_rate
        self.input_tensor =  tf.placeholder(
            tf.float32, [None,  self.height* self.width])
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.configure_networks()
    
    def configure_networks(self):
        with tf.variable_scope('VAE') as scope:
            self.train_summary = self.build_network('train',  self.input_tensor)
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)

    def build_network(self, name, input_tensor):
        summarys = []
        with tf.variable_scope('model') as scope:
            encode_out = encoder(input_tensor, self.hidden_size*4)
           # print (encode_out.get_shape())
            mean = encode_out[:,:self.hidden_size]
            
            stddev = tf.sqrt(tf.exp(encode_out[:,self.hidden_size:2*self.hidden_size]))
            mean2 =encode_out[:,self.hidden_size*2:self.hidden_size*3]
            stddev2 = tf.sqrt(tf.exp(encode_out[:,self.hidden_size*3:self.hidden_size*4]))


            new_mean = tf.reshape(mean,[tf.shape(mean)[0], self.hidden_size, 1])
       #     print (new_mean.get_shape())
            new_mean2 = tf.reshape(mean2, [tf.shape(mean2)[0], 1,  self.hidden_size])
            new_mean = new_mean* new_mean2 
       #     print(new_mean.get_shape())
            new_mean = tf.contrib.layers.flatten(new_mean)
            # how to cal the kron product?


            new_stddev = tf.reshape(stddev,[tf.shape(stddev)[0], self.hidden_size, 1])
            new_stddev2 = tf.reshape(stddev2, [tf.shape(stddev2)[0], 1, self.hidden_size])
            new_stddev = new_stddev * new_stddev2
       #     print(new_stddev.get_shape())
            new_stddev = tf.contrib.layers.flatten(new_stddev)



            epsilon = tf.random_normal([tf.shape(mean)[0], tf.square(self.hidden_size)])

            new_sample= epsilon * new_stddev + new_mean

         #   new_sample= tf.reshape(new_sample,[tf.shape(new_sample)[0], self.hidden_size, self.hidden_size])
            print(new_sample.get_shape())
            out_put = decoder(new_sample)
         #   print (out_put.get_shape())
        with tf.variable_scope("model", reuse=True) as scope:
            test_sample = tf.random_normal([self.batch_size, tf.square(self.hidden_size)])
         #   test_sample = tf.reshape(test_sample, [self.batch_size, self.hidden_size, self.hidden_size])
            self.sample_out = decoder(test_sample)        
        
        kl_loss = self.get_loss(new_mean, new_stddev)
        rec_loss = self.get_rec_loss(out_put, input_tensor)
        total_loss = kl_loss + rec_loss
        summarys.append(tf.summary.scalar('/KL-loss', kl_loss))
        summarys.append(tf.summary.scalar('/Rec-loss', rec_loss))
        summarys.append(tf.summary.scalar('/loss', total_loss))

        summarys.append(tf.summary.image('input', tf.reshape(input_tensor, [-1, 28, 28, 1]), max_outputs = 20))

        summarys.append(tf.summary.image('output', tf.reshape(out_put, [-1, 28, 28,1 ]), max_outputs = 20))
        
        self.train = tf.contrib.layers.optimize_loss(total_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.learning_rate, optimizer='Adam', update_ops=[])
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge(summarys)
        return summary

    
    def get_loss(self, mean, stddev, epsilon=1e-8):
        return tf.reduce_sum(0.5*(tf.square(mean)+
            tf.square(stddev)-2.0*tf.log(stddev+epsilon)-1.0))

    def get_rec_loss(self, out_put, target_out, epsilon=1e-8):
        return tf.reduce_sum(-target_out*tf.log(out_put+epsilon)
            -(1.0-target_out)*tf.log(1.0-out_put+epsilon))
    
    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.modeldir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
    
    def update_params(self, input_tensor, step):
        loss, summary =  self.sess.run([self.train, self.train_summary], {self.input_tensor: input_tensor})
       # print('---->summarying', step)
        self.writer.add_summary(summary, step)
        return loss






        

