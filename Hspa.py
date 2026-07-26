import tensorflow as tf
from tensorflow.keras import layers, models
class SoftThresholding(tf.keras.layers.Layer):
    def __init__(self, top_k=128, **kwargs):
        super(SoftThresholding, self).__init__(**kwargs)
        self.top_k = top_k

    def call(self, scores):
            
        s = scores - tf.reduce_max(scores, axis=-1, keepdims=True)

        
        top_k_values, _ = tf.math.top_k(s, k=self.top_k)
        topk_cumsum = tf.cumsum(top_k_values, axis=-1) - 1
        arange = tf.range(1, self.top_k + 1, dtype=s.dtype)
        support = arange * top_k_values > topk_cumsum

        support_size = tf.reduce_sum(tf.cast(support, s.dtype), axis=-1, keepdims=True)
        idx = support_size - 1
        idx = tf.clip_by_value(idx, 0, self.top_k - 1)

        # Convert idx to int32 before using it in tf.gather
        idx = tf.cast(idx, tf.int32)

        tau = tf.gather(topk_cumsum, idx, batch_dims=1)
        # print(s.shape)
        # print(tau.shape)
        # tau = tf.reduce_mean(tau, axis=-1)  
        top_m = 10 
        top_m_values, _ = tf.math.top_k(tau, k=top_m)
        tau = tf.reduce_mean(top_m_values, axis=-1)


        tau = tau / support_size
        # print(s.shape)
        # print(tau.shape)
        output = tf.nn.relu(s - tau)
        return output


class HSPA(tf.keras.layers.Layer):
    def __init__(self, channels=32, reduction=2, top_k=128, **kwargs):
        super(HSPA, self).__init__(**kwargs)
        self.channels = channels
        self.reduced_channels = channels // reduction
        self.top_k = top_k

        self.conv_match1 = layers.Conv1D(self.reduced_channels, 1, activation='relu')
        self.conv_match2 = layers.Conv1D(self.reduced_channels, 1, activation='relu')
        self.conv_assembly = layers.Conv1D(channels, 1, activation='relu')
        self.threshold = SoftThresholding(top_k=top_k)

    def call(self, x):
        # x shape: (batch, seq_len, channels)
        #print("x shape:",x.shape)
        x1 = self.conv_match1(x)  # (B, L, C')
        x2 = self.conv_match2(x)  # (B, L, C')
        x_asm = self.conv_assembly(x)  # (B, L, C)

        # similarity score
        score = tf.matmul(x1, x2, transpose_b=True)  # (B, L, L)

        # apply thresholding
        #print("score shape:",score.shape)
        score = self.threshold(score)
        #print("score shape:",score.shape)
        # attention-weighted sum
        out = tf.matmul(score, x_asm)  # (B, L, C)
        #print("out shape:",out.shape)
        return out + x  # residual connection
