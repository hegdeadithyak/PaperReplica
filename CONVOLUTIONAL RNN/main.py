import tensorflow as tf

def crnn(tensor, kernel_size, stride, out_channels, rnn_n_layers, rnn_type, bidirectional, w_std, padding, scope_name):
    with tf.variable_scope(scope_name):
        conv = tf.layers.conv1d(tensor, out_channels, kernel_size, stride, padding=padding, kernel_initializer=tf.random_normal_initializer(stddev=w_std))
        rnn = tf.keras.layers.RNN([rnn_type] * rnn_n_layers, return_sequences=True, go_backwards=bidirectional, kernel_initializer=tf.random_normal_initializer(stddev=w_std))
        return rnn(conv)

# crnn takes input in shape of [batch_size, height, width, channels] and returns output in shape of [batch_size, time_steps, out_channels] which is the output of the RNN layer. 