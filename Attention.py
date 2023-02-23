import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
import Mylayers


# This class is used in LatentStack.py
# the input shape and output shape are the same [B, H, W, C]
class Attention(tf.keras.layers.Layer):
    def __init__(self, num_channels, ratio_kq=8, ratio_v=8, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.conv1 = layers.Conv2D(filters=num_channels // ratio_kq,
                                   kernel_size=1, padding='VALID', use_bias=False)
        self.conv2 = layers.Conv2D(filters=num_channels // ratio_kq,
                                   kernel_size=1, padding='VALID', use_bias=False)
        self.conv3 = layers.Conv2D(filters=num_channels // ratio_v,
                                   kernel_size=1, padding='VALID', use_bias=False)
        self.conv4 = layers.Conv2D(filters=num_channels,
                                   kernel_size=1, padding='VALID', use_bias=False)
        # Learnable gain parameter gamma.
        # self._gamma = None
        self.gamma = tf.Variable(0., dtype=tf.float32, trainable=True, name='attention_gama')

    @tf.function
    def call(self, tensor):
        # Compute query, key and value using 1x1 convolutions.
        # query, key, value are tensors with shape [B, H, W, 24].
        query = self.conv1(tensor)
        key = self.conv2(tensor)
        value = self.conv3(tensor)

        # if self._gamma is None:
        #    self._gamma = tf.Variable(0., dtype=tf.float32, trainable=trainable, name='attention_gama')

        # Apply the attention operation.
        # here first out is tensor with shape [B, H, W, 72]
        out = Mylayers.ApplyAlongAxis(attention_ein_sum, axis=0)(query, key, value)
        out = self.gamma * self.conv4(out)
        # Residual connection. element-wise add.
        return out + tensor

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'num_channels': self.num_channels,
                       'ratio_kq': self.ratio_kq,
                       'ratio_v': self.ratio_v,
                       })
        return config

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.function
def attention_ein_sum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    # -1 inferred to be the size of h*w*c/(k.shape[-1])
    k = tf.reshape(k, [-1, k.shape[-1]])  # [h, w, c] -> [L, c]
    v = tf.reshape(v, [-1, v.shape[-1]])  # [h, w, c] -> [L, c]

    # Einstein summation corresponding to the query * key operation.
    beta = activations.softmax(tf.einsum('hwc, Lc->hwL', q, k), axis=-1)

    # Einstein summation corresponding to the attention * value operation.
    out = tf.einsum('hwL, Lc->hwc', beta, v)
    return out

