import tensorflow as tf


def Scaled_dot_product_attention(Q, K, V, mask=None):
    QK_mul = tf.linalg.matmul(Q, K)

    dk = tf.cast(K.shape(K)[-1], tf.float32)
    scaled_attention_logits = QK_mul / tf.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, V)

    return output


def Multi_Head_Attention(attention_heads):
    concatenated_attention = tf.concat(attention_heads, axis=-1)

    output = tf.keras.layers.Dense(units=64)(concatenated_attention)

    return output

def sublayer_connection(x, sublayer, dropout_rate=0.1):
    sublayer_output = sublayer(x)
    sublayer_output = tf.keras.layers.Dropout(rate=dropout_rate)(sublayer_output)

    return x + sublayer_output

