import tensorflow as tf

conv_size = 5
deconv_size_first = 2
deconv_size_second = 3
deconv_size = 5

def encoder(input_tensor, output_size): 
 #   output = tf.reshape(input_tensor, [-1, 28, 28, 1])
    print (input_tensor.get_shape())
    output = tf.contrib.layers.conv2d(
        input_tensor, 32, conv_size, scope='convlayer1', stride =2, 
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
 #   print(output.get_shape())
    output = tf.contrib.layers.conv2d(
        output, 64, conv_size, scope='convlayer2', stride =2, 
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
#    print(output.get_shape())
    output = tf.contrib.layers.conv2d(
        output, 128, conv_size, scope='convlayer3', stride =2, padding='VALID',
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True}) 
#    print(output.get_shape())   
    output = tf.contrib.layers.dropout(output, 0.9, scope='dropout1')
    output = tf.contrib.layers.flatten(output)
    return tf.contrib.layers.fully_connected(output, output_size, activation_fn=None)


def intermediate_decoder(input_tensor, batch,channel, hidden_size):
    # from batch*channel to batch*channle*d*d or(batch*channle*4*d)
    output = tf.expand_dims(input_tensor, 1)
    output = tf.expand_dims(output, 1)
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 64, deconv_size_second, scope='inter1', padding = 'VALID',
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())    
    output = tf.contrib.layers.conv2d(
        output, 128, deconv_size_second, scope='inter2', stride =1, padding='SAME',
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.transpose(output, perm=[0,3,1,2])
    output = tf.reshape(output,[batch, channel, 3*3])
    print(output.get_shape())
    output = tf.contrib.layers.fully_connected(output, hidden_size*4, activation_fn=None )
    print(output.get_shape())
    return output



def decoder(input_sensor):
#    output = tf.expand_dims(input_sensor ,1)
#    output = tf.expand_dims(output,1)
  #  output = input_sensor`
    output = tf.transpose(input_sensor, perm=[0, 2, 3 ,1])
    # print(output.get_shape())
    # output = tf.contrib.layers.conv2d_transpose(
    #     output, 256, deconv_size_second, scope='deconv1', padding='VALID',
    #     activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
    #     normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 128, deconv_size_second, scope='deconv1', stride = 2,
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 64, deconv_size_second, scope='deconv2', padding='VALID',
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})    
    output = tf.contrib.layers.conv2d_transpose(
        output, 32, deconv_size, scope='deconv3', stride = 2,
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 16, deconv_size, scope='deconv4', stride = 2,
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 3, deconv_size, scope='deconv5', stride=2,
        activation_fn=tf.nn.tanh, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())         
    return output
