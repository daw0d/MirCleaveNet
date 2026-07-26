import tensorflow as tf
from tensorflow.keras import layers

class BottConv(layers.Layer):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding='same', bias=True):
        super(BottConv, self).__init__()
        
        mid_channels = max(mid_channels, 2)  # Ensure mid_channels is at least 2
        
        # Pointwise 1x1 Conv (1D)
        self.pointwise_1 = layers.Conv1D(mid_channels, 1, strides=stride, padding='same', use_bias=bias)
        
        # Depthwise Conv (1D)
        self.depthwise = layers.Conv1D(mid_channels, kernel_size, strides=stride, padding=padding, 
                                       groups=mid_channels, use_bias=False)
        
        # Pointwise 1x1 Conv (1D)
        self.pointwise_2 = layers.Conv1D(out_channels, 1, strides=stride, padding='same', use_bias=False)
        
    def call(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x

class GBCmodel(tf.keras.Model):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBCmodel, self).__init__()
        
        # Define projection blocks
        self.proj = BottConv(in_channels, in_channels, in_channels // 8, kernel_size=3, stride=1, padding='same')
        self.norm = layers.LayerNormalization()  # Layer normalization instead of InstanceNorm3d
        
        # Define non-linearity and second projection
        self.proj2 = BottConv(in_channels, in_channels, in_channels // 8, kernel_size=3, stride=1, padding='same')
        self.norm2 = layers.LayerNormalization()
        
        # Further projection blocks
        self.proj3 = BottConv(in_channels, in_channels, in_channels // 8, kernel_size=1, stride=1, padding='valid')
        self.norm3 = layers.LayerNormalization()
        
        self.proj4 = BottConv(in_channels, in_channels, in_channels // 8, kernel_size=1, stride=1, padding='valid')
        self.norm4 = layers.LayerNormalization()

        # Additional convolution layer to adjust residual connection channels
        self.residual_conv = layers.Conv1D(in_channels, 1, strides=1, padding='same')  # Adjust channels for residual

    def call(self, x):
        x_residual = x
        
        # First projection
        x1_1 = self.norm(x)
        x1_1 = tf.nn.swish(x1_1)
        x1_1 = self.proj(x1_1)

        # Second projection
        x1 = self.norm2(x1_1)
        x1 = tf.nn.swish(x1)
        x1 = self.proj2(x1)

        # Third projection
        x2 = self.norm3(x)
        x2 = tf.nn.swish(x2)
        x2 = self.proj3(x2)

        # Combine with multiplication (element-wise)
        x = x1 * x2
        x = self.norm4(x)
        x = tf.nn.swish(x)
        x = self.proj4(x)

        # Adjust the residual channel size to match the output of x
        x_residual = self.residual_conv(x_residual)

        return x + x_residual
    # def call(self, x):
    #     x_residual = x
        
    #     # First projection
    #     x1_1 = self.proj(x)
    #     x1_1 = self.norm(x1_1)
    #     x1_1 = tf.nn.swish(x1_1)

    #     # Second projection
    #     x1 = self.proj2(x1_1)
    #     x1 = self.norm2(x1)
    #     x1 = tf.nn.swish(x1)

    #     # Third projection
    #     x2 = self.proj3(x)
    #     x2 = self.norm3(x2)
    #     x2 = tf.nn.swish(x2)

    #     # Combine with multiplication (element-wise)
    #     x = x1 * x2
    #     x = self.proj4(x)
    #     x = self.norm4(x)
    #     x = tf.nn.swish(x)

    #     # Adjust the residual channel size to match the output of x
    #     x_residual = self.residual_conv(x_residual)

    #     return x + x_residual
