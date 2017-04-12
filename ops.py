import tensorflow as tf

conv_size = 5
deconv_size_first = 4
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

def decoder(input_sensor):
#    output = tf.expand_dims(input_sensor ,1)
#    output = tf.expand_dims(output,1)
  #  output = input_sensor`
    output = tf.transpose(input_sensor, perm=[0, 2, 3 ,1])
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 128, deconv_size_first, scope='deconv1', padding='VALID',
        activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 64, deconv_size_first, scope='deconv2', stride = 2,
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
