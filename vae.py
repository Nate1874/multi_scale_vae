import os
import tensorflow as tf
import numpy as np
import math
from ops import encoder, decoder
from generator import Generator

class VAE(Generator):

    def __init__(self, hidden_size, batch_size, learning_rate, channel, iterations):
        self.working_directory = '/tempspace/hyuan/VAE'
        self.height = 28
        self.width = 28                             
        self.modeldir = './modeldir'
        self.logdir = './logdir'
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate =learning_rate
        self.channel = channel
        self.iterations = iterations
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
            encode_out = encoder(input_tensor, self.hidden_size*4*self.channel, reuse= False)
      #      print (encode_out.get_shape())
            encode_out = tf.reshape(encode_out, [self.batch_size ,self.channel, 4*self.hidden_size])
      #      print (encode_out.get_shape())
            mean1 = encode_out[ :, : , :self.hidden_size] #10*128*d*1
            stddev1 = tf.sqrt(tf.exp(encode_out[:,:,self.hidden_size:2*self.hidden_size]))
            mean2 = encode_out[:,:,2*self.hidden_size:3*self.hidden_size]
            stddev2 = tf.sqrt(tf.exp(encode_out[:,:,3*self.hidden_size:4*self.hidden_size]))
            new_mean= tf.expand_dims(mean1, -1) * tf.expand_dims(mean2, -2)
     #       print(new_mean.get_shape()) # 10 * 128 *3 * 3
            new_std = tf.expand_dims(stddev1, -1) *tf.expand_dims(stddev2, -2)
     #       print(new_std.get_shape())
            new_mean = tf.reshape(new_mean, [self.batch_size, self.channel, self.hidden_size * self.hidden_size])
            new_std = tf.reshape(new_std, [self.batch_size, self.channel, self.hidden_size * self.hidden_size])
            epsilon = tf.random_normal([self.batch_size, self.channel, self.hidden_size*self.hidden_size])
            new_sample = new_mean + epsilon*new_std
     #       print(new_sample.get_shape())
            new_sample = tf.reshape(new_sample,[self.batch_size, self.channel, self.hidden_size, self.hidden_size] )
            out_put = decoder(new_sample, reuse= False)
        self.prediction()
        self.rec_loss = self.get_rec_loss(out_put, input_tensor)
    #    total_loss = self.kl_loss + self.rec_loss

        total_loss = self.rec_loss
     #   summarys.append(tf.summary.scalar('/KL-loss', self.kl_loss))
        summarys.append(tf.summary.scalar('/Rec-loss', self.rec_loss))
        summarys.append(tf.summary.scalar('/loss', total_loss))

        summarys.append(tf.summary.image('input', tf.reshape(input_tensor, [-1, self.height, self.width, 1]), max_outputs = 20))

        summarys.append(tf.summary.image('output', tf.reshape(out_put, [-1, self.height, self.width ,1 ]), max_outputs = 20))
        
        self.train = tf.contrib.layers.optimize_loss(total_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.learning_rate, optimizer='Adam', update_ops=[])
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge(summarys)
        return summary

    
    def get_loss(self, mean, stddev, epsilon=1e-8):
        return tf.reduce_sum(0.5*(tf.square(mean)+
            tf.square(stddev)-2.0*tf.log(stddev+epsilon)-1.0))

    # def get_rec_loss(self, out_put, target_out, epsilon=1e-8):
    #     return tf.reduce_sum(-target_out*tf.log(out_put+epsilon)
    #         -(1.0-target_out)*tf.log(1.0-out_put+epsilon))
    def get_rec_loss(self, output, target_out, epsilon=1e-8):
        return tf.reduce_sum(tf.square(output - target_out+ epsilon))

    
    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.modeldir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
    
    def update_params(self, input_tensor, step):
        loss, summary=  self.sess.run([self.train, self.train_summary], {self.input_tensor: input_tensor})
       # print('---->summarying', step)
        self.writer.add_summary(summary, step)
        return loss

    def prediction(self):
        self.test_images= []
        input_test = tf.random_uniform([self.height* self.width, 1], minval= 0.0, maxval = 1.0, dtype= tf.float32)
        for i in range(self.iterations):
            encode_out = encoder(input_test, self.hidden_size*4*self.channel, reuse=True)
            encode_out = tf.reshape(encode_out, [1 ,self.channel, 4*self.hidden_size])
            mean1 = encode_out[ :, : , :self.hidden_size] #10*128*d*1
            stddev1 = tf.sqrt(tf.exp(encode_out[:,:,self.hidden_size:2*self.hidden_size]))
            mean2 = encode_out[:,:,2*self.hidden_size:3*self.hidden_size]
            stddev2 = tf.sqrt(tf.exp(encode_out[:,:,3*self.hidden_size:4*self.hidden_size]))
            new_mean= tf.expand_dims(mean1, -1) * tf.expand_dims(mean2, -2)
         #   print(new_mean.get_shape()) # 10 * 128 *3 * 3
            new_std = tf.expand_dims(stddev1, -1) *tf.expand_dims(stddev2, -2)
        #    print(new_std.get_shape())
            new_mean = tf.reshape(new_mean, [1, self.channel, self.hidden_size * self.hidden_size])
            new_std = tf.reshape(new_std, [1, self.channel, self.hidden_size * self.hidden_size])
            epsilon = tf.random_normal([1, self.channel, self.hidden_size*self.hidden_size])
            new_sample = new_mean + epsilon*new_std
            new_sample = tf.reshape(new_sample,[1, self.channel, self.hidden_size, self.hidden_size] )
            out_put = decoder(new_sample, reuse=True)
            self.test_images.append(out_put)
            input_test = out_put
    

