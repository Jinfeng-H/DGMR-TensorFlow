import keras.layers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import SpectralNormalization
import Attention
import LBlock


class LatentCondStack(tf.keras.layers.Layer):
    """Latent Conditioning Stack."""
    """Latent Conditioning Stack for the Sampler."""

    def __init__(self, **kwargs):
        super(LatentCondStack, self).__init__(**kwargs)
        self.conv1 = SpectralNormalization(layers.Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding="same"))
        self.lblock1 = LBlock.LBlock(output_channels=24)
        self.lblock2 = LBlock.LBlock(output_channels=48)
        self.lblock3 = LBlock.LBlock(output_channels=192)
        self.mini_attn_block = Attention.Attention(num_channels=192)
        self.lblock4 = LBlock.LBlock(output_channels=768)

    @tf.function
    def call(self, batch_size, resolution=(256, 256)):

        # Independent draws from a Normal distribution.
        h, w = resolution[0] // 32, resolution[1] // 32
        z = tf.random.normal(shape=[batch_size, h, w, 8])

        # 3x3 convolution.
        z = self.conv1(z)

        # Three L Blocks to increase the number of channels to 24, 48, 192.
        z = self.lblock1(z)
        z = self.lblock2(z)
        z = self.lblock3(z)

        # Spatial attention module.
        z = self.mini_attn_block(z)

        # L Block to increase the number of channels to 768.
        z = self.lblock4(z)
        # return tensor with shape [B, 8, 8, 768]
        return z

    def get_config(self):
        config = super(LatentCondStack, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
