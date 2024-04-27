import tensorflow as tf

def Scaled_dot_product_attention(Q,K,V,mask = None):
    QK_mul = tf.linalg.matmul(Q,K)

    dk = tf.cast(K.shape(K)[-1],tf.float32)
    scaled_attention_logits = QK_mul / tf.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
  
    attention_weights = tf.nn.softmax(scaled_attention_logits,axis=-1)

    output = tf.matmul(attention_weights,V)

    return output